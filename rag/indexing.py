from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path("./../.env"))

BASE_URL = os.getenv("BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
K_SAMPLE = os.getenv("K_SAMPLE")

if any(x is None for x in [BASE_URL, EMBEDDING_MODEL, K_SAMPLE]):
    print("Env is None")
    exit()
K_SAMPLE = int(K_SAMPLE) #type: ignore


class LocalEmbeddings(Embeddings):
    def __init__(self, model, base_url):
        self.client = OpenAI(
            base_url=base_url,
            api_key="none"
        )
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [data.embedding for data in response.data]

def chunk_recursive(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", ".", ",", " "]
    )
    chunks = splitter.split_text(text)
    return chunks

def chunk_document(document, title, id):
    chunks = []
    for page in document:
        chunks_tmp = chunk_recursive(page.page_content, chunk_size=500, chunk_overlap=100)
        for chunk in chunks_tmp:
            id+=1
            chunks.append(Document(page_content=chunk,id=id, metadata={"title":title}))
    return (chunks,id)

def process_documents(documents):
    chunks = []
    id = 0
    for doc in documents:
        chunks_ret, id = chunk_document(doc, doc[0].metadata['title'][:-12], id)
        chunks.extend(chunks_ret)
    return chunks


def load_documents():
    dir_path = "./../data/"
    docs = []

    for filename in os.listdir(dir_path):
        loader = PyPDFLoader(dir_path + filename)
        pages = loader.load()
        docs.append(pages)
    return docs


def main():

    embeddings = LocalEmbeddings(EMBEDDING_MODEL, BASE_URL)

    vector_store = Chroma(
        collection_name="animated_movies",
        embedding_function=embeddings,
        persist_directory="./../chroma_db"
    )
    vector_store.reset_collection()

    docs = load_documents()
    chunks = process_documents(docs)
    vector_store.add_documents(chunks)

if __name__ == "__main__":
    main()