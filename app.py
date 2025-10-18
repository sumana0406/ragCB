# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

VECTOR_STORE = os.getenv("VECTOR_STORE", "pinecone").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot — Pinecone / Chroma")

if "history" not in st.session_state:
    st.session_state.history = []

def get_retriever():
    if VECTOR_STORE == "chroma":
        import chromadb
        from chromadb.config import Settings
        from langchain.vectorstores import Chroma
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=os.getenv("CHROMA_PERSIST_DIR","./chroma_data")))
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectordb = Chroma(client=client, persist_directory=os.getenv("CHROMA_PERSIST_DIR","./chroma_data"), embedding_function=embeddings)
        return vectordb.as_retriever(search_kwargs={"k": 4})
    elif VECTOR_STORE == "pinecone":
        import pinecone
        from langchain.vectorstores import Pinecone
        pine_api_key = os.getenv("PINECONE_API_KEY")
        pine_env = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
        pinecone.init(api_key=pine_api_key, environment=pine_env)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        index = pinecone.Index(index_name)
        vectordb = Pinecone(index, embeddings.embed_query, "text")
        return vectordb.as_retriever(search_kwargs={"k": 4})
    else:
        raise ValueError("VECTOR_STORE must be 'pinecone' or 'chroma'")

@st.cache_resource
def build_chain():
    retriever = get_retriever()
    llm = OpenAI(model_name=OPENAI_MODEL, temperature=0.0)  # deterministic
    # Basic prompt to combine retrieved context and question
    prompt = PromptTemplate.from_template(
        """You are an assistant that answers user questions using the provided context.
        If answer is not in context, say you don't know and provide best effort using general knowledge.

        Context:
        {context}

        Question:
        {question}

        Answer concisely and list sources if available.
        """
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

qa_chain = build_chain()

with st.sidebar:
    st.header("Settings")
    st.write(f"Vector store: **{VECTOR_STORE}**")
    st.write(f"OpenAI model: **{OPENAI_MODEL}**")
    if VECTOR_STORE == "chroma":
        st.write("Chroma persist dir: " + os.getenv("CHROMA_PERSIST_DIR","./chroma_data"))
    else:
        st.write("Pinecone index: " + os.getenv("PINECONE_INDEX_NAME","rag-index"))

query = st.text_input("Ask a question about your documents", key="input")
if st.button("Send") and query:
    with st.spinner("Retrieving..."):
        res = qa_chain({"query": query})
    answer = res.get("result") or res.get("answer")
    sources = res.get("source_documents", [])
    st.session_state.history.append((query, answer, sources))

# display history
for q,a,sources in reversed(st.session_state.history[-10:]):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    if sources:
        st.markdown("**Sources:**")
        for doc in sources[:4]:
            metadata = getattr(doc, "metadata", {})
            txt_preview = (doc.page_content[:400] + "...") if len(doc.page_content) > 400 else doc.page_content
            st.markdown(f"- {metadata.get('source','unknown')} — {txt_preview}")
    st.markdown("---")
