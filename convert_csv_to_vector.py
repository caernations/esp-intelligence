import os
import pandas as pd
import openai
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
import faiss

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load ketiga CSV
wells_df = pd.read_csv("data/wells_cleaned.csv")
installations_df = pd.read_csv("data/esp_well_installations_cleaned.csv")
failure_cause_df = pd.read_csv("data/esp_failure_cause_cleaned.csv")

def safe_str(val):
    return str(val) if pd.notna(val) else "NA"

def row_to_narrative(row):
    parts = []
    for col in row.index:
        parts.append(f"{col}: {safe_str(row[col])}")
    return ", ".join(parts)

# Buat narasi per row untuk ketiga dataset
wells_texts = wells_df.apply(row_to_narrative, axis=1).tolist()
installations_texts = installations_df.apply(row_to_narrative, axis=1).tolist()
failure_texts = failure_cause_df.apply(row_to_narrative, axis=1).tolist()

# Gabung semua teks jadi satu list
all_texts = wells_texts + installations_texts + failure_texts

# Split teks jadi chunk kecil supaya embed lebih efektif
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20,
    separators=["\n\n", "\n", ".", ",", " ", ""]
)

documents = []
for text in all_texts:
    docs = text_splitter.create_documents([text])
    documents.extend(docs)

print(f"Total chunks to embed: {len(documents)}")

# Batch embed semua chunk dengan OpenAI API (model ada-002)
def batch_embed_texts(texts, batch_size=1000):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        for embedding_obj in response.data:
            embeddings.append(embedding_obj.embedding)
        print(f"Embedded batch {i//batch_size + 1} / {(len(texts) + batch_size -1)//batch_size}")
    return embeddings

texts = [doc.page_content for doc in documents]
embeddings = batch_embed_texts(texts, batch_size=1000)

embeddings_np = np.array(embeddings, dtype=np.float32)

# Buat index FAISS manual dari embeddings
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Mapping dari index ke dokumen (docstore)
docstore = {i: doc for i, doc in enumerate(documents)}

# Setup embedding_function untuk FAISS wrapper
embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Buat vectorstore FAISS lengkap
vectorstore = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id={i: i for i in range(len(documents))},
    embedding_function=embedding_function,
)

# Simpan vectorstore ke folder embeddings/
os.makedirs("embeddings", exist_ok=True)
vectorstore.save_local("embeddings/esp_full_vectorstore")

print("✅ Semua embedding selesai dan vectorstore disimpan di 'embeddings/esp_full_vectorstore'")
