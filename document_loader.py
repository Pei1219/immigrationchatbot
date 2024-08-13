import os
import logging
import json
from typing import List
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_core.documents import Document
from default_config import parse_arguments

args = parse_arguments()

def load_documents_into_database(model_name: str, documents_path: str, chunk_size: int, chunk_overlap: int) -> Chroma:
    logging.info("Initializing text splitter")
    json_splitter = RecursiveJsonSplitter(max_chunk_size=300)

    logging.info("Loading documents")
    raw_documents = load_documents(documents_path)
    
    logging.info("Splitting documents into chunks")
    documents = []
    chunk_counter = 0  # Counter to keep track of the number of chunks
    for doc in raw_documents:
        try:
            json_data = json.loads(doc.page_content)  # Assuming the content of the document is JSON
            split_chunks = json_splitter.split_json(json_data=json_data)
            # Convert each chunk back into a Document object
            for chunk in split_chunks:
                chunk_counter += 1
                chunk_content = json.dumps(chunk)
                documents.append(Document(page_content=chunk_content))
                # Print each chunk
                print(f"Chunk {chunk_counter}: {chunk_content}")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from document: {e}")
    
    logging.info(f"Total document chunks created: {len(documents)}")

    logging.info("Creating embeddings and loading documents into Chroma")
    if model_name == "OpenAI":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=args.openai_key, openai_api_base=args.openai_base)
    else:
        embeddings = OllamaEmbeddings(model=model_name)
    
    db = Chroma.from_documents(documents, embeddings)
    
    logging.info("Documents loaded into Chroma successfully")
    return db

def load_documents(path: str) -> List[Document]:
    if not os.path.exists(path):
        logging.error(f"The specified path does not exist: {path}")
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    docs = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.endswith(".json"):  # 确保只处理文件，并且是 .json 文件
            logging.info(f"Loading {file_path}")
            loader = JSONLoader(file_path=file_path, jq_schema='.sections[]', text_content=False)
            docs.extend(loader.load())
    
    logging.info(f"Total documents loaded: {len(docs)}")
    return docs
