import os
import json
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 🔹 Function to extract text from nested JSON
def extract_text_from_json(data, prefix=""):
    text_chunks = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix} -> {key}" if prefix else key
            text_chunks.extend(extract_text_from_json(value, new_prefix))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_prefix = f"{prefix} [{idx}]"
            text_chunks.extend(extract_text_from_json(item, new_prefix))
    else:
        text_chunks.append(f"{prefix}: {data}")
    return text_chunks

# 🔹 Function to create/load FAISS vector store
def create_vector_store(folder_path, vectorstore_path="bylaws_vector_index"):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_texts.extend(extract_text_from_json(data))

    structured_text = "\n".join(all_texts)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
    split_docs = text_splitter.create_documents([structured_text])
    
    for doc in split_docs:
        doc.metadata = {"source": "bylaw"}

    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(vectorstore_path)
    return vector_store



