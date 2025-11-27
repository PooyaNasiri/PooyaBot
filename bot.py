import os
import logging
import json
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import threading
from flask import Flask

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()

# Retrieve Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Validate Keys
if not all(
    [TELEGRAM_TOKEN, GOOGLE_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY, GITHUB_TOKEN]
):
    print("CRITICAL ERROR: One or more API keys are missing in .env file.")
    exit(1)

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- IMPORTS ---
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from github import Github
from pinecone import Pinecone

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# Smart Import for Pinecone VectorStore
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    try:
        from langchain_pinecone import Pinecone as PineconeVectorStore
    except ImportError:
        from langchain_community.vectorstores import Pinecone as PineconeVectorStore

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# --- 2. INITIALIZE SERVICES ---

# A. Initialize Pinecone Client
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("pooya-bot")
except Exception as e:
    logger.error(f"Failed to connect to Pinecone: {e}")
    exit(1)

# B. Initialize Embeddings & VectorStore
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

try:
    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
except TypeError:
    vectorstore = PineconeVectorStore(index_name="pooya-bot", embedding=embeddings)

# --- 3. DEFINE TOOLS ---


@tool
def check_my_memory(query: str):
    """
    ALWAYS use this FIRST. Search here for Pooya's personal opinions,
    past projects, resume, biography, or specific advice he has written.
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        if not docs:
            return "No specific personal memory found."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error reading memory: {e}"


@tool
def check_github_activity(query: str):
    """
    Use this to see what Pooya is coding RIGHT NOW.
    Returns his latest repositories and commit messages.
    """
    try:
        g = Github(GITHUB_TOKEN)
        user = g.get_user()
        repos = user.get_repos(sort="updated", direction="desc")[:3]

        info = []
        for repo in repos:
            desc = repo.description if repo.description else "No description"
            info.append(f"Repo: {repo.name} | URL: {repo.html_url} | Desc: {desc}")

        if not info:
            return "No recent public repositories found."
        return "\n".join(info)
    except Exception as e:
        return f"Error connecting to GitHub: {e}"


# --- UPDATED: GENERAL WEB SEARCH TOOL ---
# We use the raw Tavily tool directly so the LLM can search for ANYTHING (Weather, News, Pooya's LinkedIn)
web_search_tool = TavilySearchResults(max_results=3)
web_search_tool.name = "web_search"
web_search_tool.description = "Use this to search the internet for live info (Weather, News) or public info about Pooya (LinkedIn/Instagram)."

tools = [check_my_memory, check_github_activity, web_search_tool]

# --- 4. THE BRAIN (AGENT SETUP) ---

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.7
)  # Slightly higher temp for better "improv"
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chatbot_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Graph Construction
workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()

# --- 5. TELEGRAM HANDLERS ---


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id

    if not user_text or not user_text.strip():
        await context.bot.send_message(
            chat_id=chat_id, text="Please send a valid text message."
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # --- UPDATED SYSTEM PROMPT ---
    system_message = """You are the AI Digital Twin of **Pooya Nasiri**. You are a developer, a mentor, and a digital secretary.

    **CORE IDENTITY & TONE:**
    1. **The Persona:** You are friendly, tech-savvy, and warm, but your defining characteristic is **SARCASM**. You love a good joke, you are witty, and you often poke fun at the user's questions before helping them.
    2. **The "Not an AI" Rule:** NEVER say "As an AI". You are the digital essence of Pooya.
    3. **Third Person ONLY:** You must **ALWAYS** speak about Pooya in the third person.
       - WRONG: "I think you should..." or "I suggest..."
       - RIGHT: "Pooya suggests that you..." or "Pooya thinks this is a terrible idea, but he would say..."
    4. **Language Protocol:** - **English:** Refer to him as "Pooya".
       - **Persian (Farsi):** If the user speaks Farsi, you MUST respond in Farsi. Refer to him as "پویا". Be just as sarcastic and helpful in Farsi.

    **TOOL USAGE STRATEGY:**
    - **Step 1 (Memory):** Always use `check_my_memory` first. See if Pooya has actual past experiences, resume details, or specific opinions on the topic.
    - **Step 2 (Live Info):** Use `web_search` if the user asks about the weather, current news, or wants to know what Pooya is up to on LinkedIn/Instagram.
    - **Step 3 (Code):** Use `check_github_activity` ONLY for questions about coding, repositories, or technical stacks.

    **THE "NEVER GIVE UP" RULE (CRITICAL):**
    - If the tools return "No results" (e.g., for "How to fix a broken heart" or "How to sleep better"), **DO NOT** say "Pooya doesn't know" or "I have no information."
    - **INSTEAD:** Improvise! Use general knowledge but **frame it through Pooya's personality.**
       - *Example:* "Pooya hasn't committed code for 'better sleep' to GitHub yet, but he would probably tell you to stop staring at blue light and close your tabs."
       - *Example (Farsi):* "پویا متخصص قلب نیست، اما پیشنهاد می‌کند که..."
    - **Always provide an answer**, even if it's a sarcastic life tip based on general logic.
    """

    messages = [SystemMessage(content=system_message), HumanMessage(content=user_text)]

    try:
        # Invoke the Agent
        result = app_graph.invoke(
            {"messages": messages}, config={"recursion_limit": 10}
        )

        final_message = result["messages"][-1]

        # Robust Text Extraction
        if isinstance(final_message.content, list) and len(final_message.content) > 0:
            if (
                isinstance(final_message.content[0], dict)
                and "text" in final_message.content[0]
            ):
                bot_reply = final_message.content[0]["text"]
            else:
                bot_reply = str(final_message.content)
        elif isinstance(final_message.content, str):
            bot_reply = final_message.content
        else:
            bot_reply = "I'm thinking, but I couldn't formulate a text response."

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        bot_reply = "Sorry, I encountered an internal error. Please try again."

    await context.bot.send_message(chat_id=chat_id, text=bot_reply)




# --- 6. FLASK SERVER FOR CLOUD DEPLOYMENT ---
app = Flask(__name__)

@app.route('/')
def health_check():
    return "PooyaBot is running!", 200

def run_flask():
    # Render assigns a port automatically via the 'PORT' env var
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    # 1. Start the Flask Server in a background thread (Keep-Alive)
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # 2. Start the Telegram Bot (Main Process)
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", lambda u, c: c.bot.send_message(chat_id=u.effective_chat.id, text="Hi! I'm AI Pooya. Ask me anything.")))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print(f"✅ Pooya Bot is Online and Ready! (Listening on port {os.environ.get('PORT', 8080)})")
    application.run_polling()

################### for local testing ####################
# if __name__ == "__main__":
#     application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

#     application.add_handler(
#         CommandHandler(
#             "start",
#             lambda u, c: c.bot.send_message(
#                 chat_id=u.effective_chat.id, text="Hi! I'm AI Pooya. Ask me anything."
#             ),
#         )
#     )
#     application.add_handler(
#         MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
#     )

#     print("✅ Pooya Bot is Online and Ready!")
#     application.run_polling()
