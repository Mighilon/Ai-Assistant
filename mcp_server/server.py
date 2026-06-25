from langchain_core.embeddings import Embeddings
from collections import defaultdict
from langchain_chroma import Chroma
from fastmcp import FastMCP
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import httpx
import os

SUMMARIES_DIR = Path("./../summaries_movies")

load_dotenv(dotenv_path=Path("./../.env"))
BASE_URL = os.getenv("BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
K_SAMPLE = os.getenv("K_SAMPLE")
MCP_URL = os.getenv("MCP_URL")
MODEL = os.getenv("MODEL")
MULTI_QUERY = os.getenv("MULTI_QUERY")

if any(x is None for x in [BASE_URL, EMBEDDING_MODEL, K_SAMPLE, MCP_URL, MODEL, MULTI_QUERY]):
    print("Env is None")
    exit()
K_SAMPLE = max(int(K_SAMPLE)-1,0) #type: ignore
MULTI_QUERY = int(MULTI_QUERY) #type: ignore


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

embeddings = LocalEmbeddings(EMBEDDING_MODEL, BASE_URL)

vector_store = Chroma(
    collection_name="animated_movies",
    embedding_function=embeddings,
    persist_directory="./../chroma_db"
)

def process_query(query, vector_store):
    results = vector_store.similarity_search(query, k=K_SAMPLE)
    return results

def rank_deduplicate(queries):
    groups = defaultdict(list)
    for res in queries:
        groups[res.metadata["title"]].append(res.page_content)
    sorted_groups = sorted(
        groups.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    return sorted_groups[0]

async def fetch_omd(client: httpx.AsyncClient, movie_name: str) -> Optional[dict]:
    try:
        url = "http://www.omdbapi.com/"
        params = {
            "t": movie_name,
            "apikey": "ac7ede43"
        }
        
        response = await client.get(url, params=params, timeout=5.0)
        data = response.json()
        return data
        
    except Exception as e:
        pass
    
    return None

llm = OpenAI(
    base_url=BASE_URL,
    api_key="none"
)
def generate_queries(question: str) -> list[str]:
    prompt = f"Generate {MULTI_QUERY} different search queries for: \n\n{question} \n\nReturn one query per line and ONLY the generated queries."

    response = llm.responses.create(
        model=MODEL, #type: ignore
        input=prompt
    ).output_text

    res = [question]
    res.extend(response.splitlines())
    return res

mcp = FastMCP("mcp_server", auth=None)

@mcp.tool()
def similarity_search(query: str):
    """Performs semantic vector search on a movie documents database to retrieve similar info about movies, reviews, plot summaries, and metadata."""
    
    queries = generate_queries(query)

    all_docs = []

    for q in queries:
        docs = process_query(q, vector_store)
        all_docs.extend(docs)

    final_docs = rank_deduplicate(all_docs)
    return final_docs


@mcp.tool()
async def movie_ratings(movie: str) -> dict:
    """Retrives the ratings for a specified movie."""
    result = {
        "movie": movie,
        "scores": {
            "imdb": None,
            "rotten_tomatoes": None,
            "metacritic": None,
        },
        "error": None
    }
    
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            data = await fetch_omd(client, movie)
            if data == None:
                result["error"] = "Error: Couldn't find the indicated movie title!"
                return result
            if "Ratings" not in data:
                return result
            for rating in data["Ratings"]:
                if rating["Source"]=="Metacritic":
                    result["scores"]["metacritic"] = rating["Value"]
                if rating["Source"]=="Rotten Tomatoes":
                    result["scores"]["rotten_tomatoes"] = rating["Value"]
                if rating["Source"]=="Internet Movie Database":
                    result["scores"]["imdb"] = rating["Value"]

    except Exception as e:
        result["error"] = f"Error fetching scores: {str(e)}"
    
    return result

@mcp.tool()
def get_movie_summary(movie_name: str) -> str:
    """Movie summary and plot information"""
    file_path = SUMMARIES_DIR / f"{movie_name}.txt"

    if not file_path.exists():
        return f"Movie '{movie_name}' not found."

    return file_path.read_text(encoding="utf-8")

@mcp.tool()
def list_movies() -> str:
    """List of movies available"""
    files = sorted(SUMMARIES_DIR.glob("*.txt"))

    if not files:
        return "No movies found."

    return "\n".join(file.stem for file in files)

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=9000)