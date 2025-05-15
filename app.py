import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@st.cache_data(show_spinner=True)
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        "embeddings/esp_full_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True  # tambahkan ini
    )
    return vectorstore


st.title("⚡ ESP Hybrid Q&A Tool")

vectorstore = load_vectorstore()

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # batasi 3 dokumen untuk prompt supaya gak overload

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = st.text_input("Tanya apa saja tentang data ESP kamu:")

if query:
    with st.spinner("Mencari jawaban..."):
        answer = qa_chain.run(query)
        st.markdown(f"### Jawaban:\n{answer}")

    # Contoh visualisasi sederhana kalau ada kata kunci tertentu
    if any(word in query.lower() for word in ["vendor", "run life", "performance"]):
        df = pd.read_csv("data/esp_well_installations_cleaned.csv")
        st.subheader("Grafik Rata-rata Run Life per Vendor")
        avg_run = df.groupby("vendor")["run"].mean().sort_values(ascending=False)
        st.bar_chart(avg_run)
