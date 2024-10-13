from ollama import AsyncClient
from rag import RagFile

from repository import run_search_query
from gemini import formatPrompt

import json


rag = RagFile('../credenciamento.pdf')

async def connect_to_aioprompt():
    client = AsyncClient(host="http://localhost:11435")
    
    return client

async def chat(message):
    client = await connect_to_aioprompt()

    query = formatPrompt(message, run_search_query(message))

    stream =  await client.chat(model='llama3', messages=[{"role": "user", "content": query}], stream=True, options={'num_gpu':1})
    
    async for part in  stream:
            print(f"Streamed part: {json.dumps(part)}")
            # decoded_part = json.loads(part)
            yield f"{json.dumps(part)}\n"

