# ingest.py
import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings

VECTOR_STORE = os.getenv("VECTOR_STORE", "pinecone").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert OPENAI_API_KEY, "Set OPENAI_API_KEY in .env"

# get documents from a folder or individual file path
def load_documents(data_path):
    p = Path(data_path)
    if p.is_dir():
        # try to load pdf/txt
        loader = DirectoryLoader(str(p), glob="**/*.*", loader_cls_map={
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
        })
        docs = loader.load()
    else:
        ext = p.suffix.lower()
        if ext == ".pdf":
            docs = PyPDFLoader(str(p)).load()
        else:
            docs = TextLoader(str(p)).load()
    return docs

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def build_chroma(chunks):
    import chromadb
    from chromadb.config import Settings
    from langchain.vectorstores import Chroma

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, client=client, persist_directory=persist_dir)
    vectordb.persist()
    print("Chroma persist done at:", persist_dir)
    return vectordb

def build_pinecone(chunks):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    pine_api_key = os.getenv("PINECONE_API_KEY")
    pine_env = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")

    assert pine_api_key and pine_env, "Set PINECONE_API_KEY and PINECONE_ENVIRONMENT in .env when using Pinecone"

    pinecone.init(api_key=pine_api_key, environment=pine_env)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # create index if not exists (use embedding dim 1536 which matches many OpenAI embeddings)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536, metric="cosine")
        print(f"Created Pinecone index {index_name}")

    index = pinecone.Index(index_name)
    vectordb = Pinecone(index, embeddings.embed_query, "text")
    vectordb.add_documents(chunks)
    print("Pinecone upsert done to index:", index_name)
    return vectordb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to file or folder to ingest (pdf, txt, etc.)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    docs = load_documents(args.data_path)
    chunks = split_docs(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    print(f"Loaded {len(docs)} doc(s) -> {len(chunks)} chunks")

    if VECTOR_STORE == "chroma":
        build_chroma(chunks)
    elif VECTOR_STORE == "pinecone":
        build_pinecone(chunks)
    else:
        raise ValueError("VECTOR_STORE must be 'pinecone' or 'chroma'")
