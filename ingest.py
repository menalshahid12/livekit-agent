import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# PASTE YOUR KEY DIRECTLY HERE
MY_GOOGLE_KEY = "AIzaSyAL-GXL4CJx75SHLLMwGyE8KhOwVKNWR0Y"

def rebuild_db():
    data_path = './data'
    if not os.path.exists(data_path):
        print("Error: 'data' folder not found!")
        return
        
    documents = []
    print("Loading documents using Brute Force mode...")

    for file in os.listdir(data_path):
        if file.endswith('.txt'):
            full_path = os.path.join(data_path, file)
            # This 'latin-1' + 'ignore' combo is impossible to crash
            with open(full_path, 'r', encoding='latin-1', errors='ignore') as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": file}))

    print(f"Loaded {len(documents)} files. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # We pass the key DIRECTLY into the class
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=MY_GOOGLE_KEY
    )

    print(f"Creating ChromaDB from {len(docs)} chunks...")
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    print("✅ SUCCESS! Your brain (chroma_db) is ready.")

if __name__ == "__main__":
    rebuild_db()