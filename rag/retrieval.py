from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI
from pathlib import Path
import asyncio
import os
import json

load_dotenv(dotenv_path=Path("./../.env"))

BASE_URL = os.getenv("BASE_URL")
MCP_URL = os.getenv("MCP_URL")
MODEL = os.getenv("MODEL")
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", '10'))

if any(x is None for x in [BASE_URL, MCP_URL, MODEL]):
    print("Env is None")
    exit()


SYSTEM_PROMPT="""
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

llm = OpenAI(
    base_url=BASE_URL,
    api_key="none"
)

async def get_tools(mcp: Client):
    raw = await mcp.list_tools()

    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {
                    "type": "object",
                    "properties": {}
                }
            }
        }
        for t in raw
    ]

mcp = Client(MCP_URL)  #type: ignore
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
]
async def agent(user_input: str):
    global messages

    async with mcp:
        tools = await get_tools(mcp)
        messages.append({"role": "user", "content": user_input})

        while True:
            response = llm.chat.completions.create(
                model=MODEL, #type: ignore
                messages=messages, #type: ignore
                tools=tools, #type: ignore
                tool_choice="auto"
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content #type: ignore
                })
                return msg.content

            messages.append(msg) #type: ignore

            for tc in msg.tool_calls:

                name = tc.function.name #type: ignore
                
                args = json.loads(tc.function.arguments) #type: ignore
                result = await mcp.call_tool(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result.content)
                })

async def main():
    global messages
    
    while len(messages) < MAX_MESSAGES:
        prompt = input("\033[32mYour prompt:\033[0m \n") 
        if prompt == 'exit':
            return
        res = await agent(prompt)
        print("\033[33mOutput: \033[0m")
        print(res)

    # data = {}
    # with open('./../evaluation/test_queries.json', 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    # for query in data["queries"]: 
    #     messages = [ {"role": "system", "content": SYSTEM_PROMPT}, ]
    #     answer = await agent(query)
    #     print(query)
    #     print(answer)
    #     print()

if __name__ == "__main__":
    asyncio.run(main())