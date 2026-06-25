## Domain Selection  
First I'm sorry for not respecting the hard deadline.  
Here is the link to the presentation [link](https://www.youtube.com/watch?v=kOAgyhhmoQs).  
  
My selected domain is *Animated movies*. I wanted to chose something simple, so I chose that. The main reason was that I didn't want to get things complicated, from the start.
The collection of the documents was easy, because I just downloaded some Wikipedia articles.  
## RAG Implementation  
Documents: My document collection consists in 10 Wikipedia articles, as PDF.  
  
Chunking: For chunking I used overlap semantic with 500 characters per chunk, and overlap 100. I decided to use 500 characters because the documents have a lot of text. And I didn't want to lose context easily.  
  
Vector store: For vector store I used Chroma, it looked simple and easy to start with it, and so it was.  The vector database is stored in a file. I didn't want to make a server for it, for simplicity. The database is loaded in the `FastMCP` server, and lives there.  
  
Retrieval: For retrieval part I implemented a function which searches the vector database, groups the vectors by document of origin, remove duplicated, and returns the *majority* result, the group with the most vectors.  
I integrated this function as a tool in the MCP server.  
  
Generation: I used local Embedder and LLM. For LLM I tried a smaller model ~1B, for curiosity, but it was to week so I switched to a 4B model, which performed well, with a good speed.  A larger model loses a lot of speed.  
  
Advanced technique: I used multi-query. When the model calls the function `similarity_search`, I generate other queries, and I query the database with the original and the generated queries for chunks of documents. And at the end I filter all the returned chunks as explained above.  
For this case I have some env variables:  
`K_SAMPLE` - defines how many chunks the vector database returns per query.  
`MULTI_QUERY` - how many queries it generates, it also includes the original query   
A function call it return at most `K_SAMPLE`\*`MULTI_QUERY` chunks. I have set the variables to a small numbers, because the 4B model doesn't handle a lot of chunks, that also takes into consideration that a chunk has 500 chars.  
  
## MCP Server Development  
Tools:  
`similatiry_search` - it performs queries on the vector database  
`movie_ratings` - retrieves the ratings for a specified movie. This tool is connected to `OMDb API`  which returns information about specific movies.  
`get_movie_summary` - returns prewritten movie summary and plot information. Actually this should be a resource, but I had some troubles with the `@mcp.resource()`, and I made it as a tool.  
`list_movies` - returns the list of available movies.  
  
## Evaluation  
The results of the evaluation are in the `evaluation` folder. From 11 tests 2 are wrong. The test are done with a fresh context every time. If we use the same context the accuracy will drop. Sometimes saying to the model to search, or to look in it's resources helps.  
  
Bellow is an example. The question can be considered very specific, but even so it returns the right  answer, because of the RAG system. The normal 4B LLM can't answer that, it hallucinates.  
  
```txt
Prompt:
How many individual bunny characters intended to have the 'Bambi' film?  

Response:
Originally, the film was intended to have six individual bunny characters, similar to the dwarfs in Snow White. However, the concept evolved to include five generic rabbits and one rabbit with a distinct coloring and dental feature. The initial cast included Thumper (a pink-nosed cottontail), Flower (a skunk), and Faline.    
```

One of the wrong responses are:  
```txt
Prompt:
How long is the runtime of Toy Story?  

Response:
The runtime of Toy Story is 95 minutes.  
```
That is wrong. The actual answer is 81 minutes. The database contains the correct information, but it fails to retrieve it.  
  
The model utilises the tools correctly. When asked about the ratings of a movie it returns the correct ratings. A normal model, without access to the tools doesn't return correct answers.  
```
Prompt:
What are the ratings of Bambi movie

Response:
The movie *Bambi* has an average rating of 7.3 out of 10 on IMDb, 91% on Rotten Tomatoes, and 91 out of 100 on Metacritic.
*Correct* usage of the tool
```

## How to run  
First you need to fill the `.env` file with the right values. Bellow is the necessary variables, and their corresponding description.   
If you don't want to get an `OMDb_API` key you can just comment the function `get_movie_summary` from `./mcp_server/server.py`.  
  
After that you need to load the vector database.  
```bash
python ./rag/indexing.py
```
  
Once the database is loaded, you can start the MCP server.  
```bash
python ./mcp_server/server.py
```
  
In a new terminal you can start the AI assistant  
```bash
python ./rag/retrieval.py
```
  
In the `examples` folder there are some examples of conversation with the AI assistant.  

## Other stuff  
### Env Variables  
```bash
BASE_URL="http://172.24.192.1:1234/v1" #The endpoint of the models, mine is not localhost because of WSL
EMBEDDING_MODEL="text-embedding-embeddinggemma-300m"
MODEL="google/gemma-3-4b"
MCP_URL="http://127.0.0.1:9000/mcp" #The endpoint of the MCP server
K_SAMPLE=5 #How many chunks returns the vector db per direct query
MULTI_QUERY=2 #The number of generated queries when acessing the vector db (the original query is included) 
MAX_MESSAGES=20 #The max length of the LLM conversation 
OMD_API_KEY="########" #The api key from OMDb
```
### System prompt
```
You are a movie information assistant designed to help users discover, learn about, and discuss films.

Guidelines:
- Use available tools proactively when queries require current information, data verification, or external sources
- If a tool returns insufficient results, refine your query and retry once before acknowledging limitations
- If a user's query appears to contain spelling or logical errors, clarify rather than silently correct—ask before assuming
- Prioritize accuracy: verify results are adequate before responding; if gaps remain, use additional tools
- Provide natural, transparent responses; avoid mentioning tool mechanics or implementation details
- Keep responses concise while ensuring completeness and clarity
- When uncertain, acknowledge limitations honestly rather than speculating

Be concise but complete."""
```