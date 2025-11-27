import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# Load environment variables
load_dotenv()

def ingest():
    print("--- 1. Loading Your 'Brain' (Data) ---")
    # Load TXT and PDF files from 'data' folder
    pdf_loader = DirectoryLoader('./data', glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
    
    docs = pdf_loader.load() + txt_loader.load()
    
    if not docs:
        print("No data found in 'data/' folder!")
        return

    print(f"--- 2. Splitting {len(docs)} documents into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("--- 3. Saving to Pinecone Memory ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Push to Pinecone
    PineconeVectorStore.from_documents(
        documents=splits, 
        embedding=embeddings, 
        index_name="pooya-bot"
    )
    print("--- Done! Your Digital Twin has learned. ---")

if __name__ == "__main__":
    # For local running, set keys here or in .env
    # os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_KEY"
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_KEY"
    ingest()