import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from constants import *
import os

if not os.path.exists(VECTOR_STORE_PATH):
    print("Creating FAISS vector store...")
    df = pd.read_csv(CSV_PATH)
    df['text'] = df['p_id'].astype(str) + " " + df['name'] + " " + df['description'] + " " + df['price'].astype(str) + " " + df['img']
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    documents = []
    for _, row in df.iterrows():
        chunks = splitter.split_text(row["text"])
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "p_id": row["p_id"],
                    "name": row["name"],
                    "brand": row.get("brand", ""),
                    "price": row["price"]
                }
            ))

    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("FAISS vector store created.")
else:
    print("FAISS vector store already exists.")