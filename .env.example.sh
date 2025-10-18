OPENAI_API_KEY= your-openai-api-key
OPENAI_MODEL=gpt-4o-mini   # or gpt-4o, gpt-4-turbo, etc.
EMBEDDING_MODEL=text-embedding-3-small

# Vector store choice: 'pinecone' or 'chroma'
VECTOR_STORE=pinecone

# Pinecone credentials (if using Pinecone)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=rag-index

# Chroma (if using Chroma)
CHROMA_PERSIST_DIR=./chroma_data

# Streamlit
STREAMLIT_SERVER_PORT=8501
