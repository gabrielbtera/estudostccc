# %%
from typing import List
from vectordb import Memory


import pymupdf4llm
import asyncio
import json

# %%

def formatar(txt):
  txt['text'] = f" $$ pag{str(txt['metadata']['page'])} arq-{txt['metadata']['file_path']} \n\n\n" + txt['text']
  
  return txt



# %%
md_text = pymupdf4llm.to_markdown("credenciamento.pdf", page_chunks=True, write_images=False)

md_text = [formatar(i)  for i in md_text]


brute = ''
for i in md_text:
  brute += i['text']

# %%
# md_tex_2t = pymupdf4llm.to_markdown("credenciamento.pdf", page_chunks=False, write_images=False)


from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter



padrao = [('$$', 'pagina')]

md_spliter = MarkdownHeaderTextSplitter(headers_to_split_on=padrao)

sections = md_spliter.split_text(brute)




# %%
# chunk_size = 20
# chunk_overlap = 3

# text = 'Once you have your data in Markdown format you are ready to chunk/split it and supply it to your LLM, for example, if this is LangChain then do the following:'

# splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# chunks = splitter.split_text(text)


# %%
memory = Memory(chunking_strategy={"mode": 'sliding_window', "window_size": 500, "overlap": 8})




# %%

for i in sections:
  memory.save( i.page_content, i.metadata)


q =  memory.search('Como fazer um requerimento', top_n=10)

# q
  


# %%
vector = [i['chunk'] for i in memory.search('Como fazer um requerimento', top_n=10)]

t = ' Use o texto seguinte como base:'
for i in vector:
  t += i


def searchQuery(prompt):
  vector = [i['chunk'] for i in memory.search(prompt, top_n=10)]

  print(prompt)

  t = f'{prompt} (formate em markdown e use o texto seguinte como base): '
  for i in vector:
    t += i
  
  print('construiu a query', t)
  
  return t
   

# %%


# %%
from openai import OpenAI
from dotenv import load_dotenv
from IPython.display import display_markdown


import os

load_dotenv()
'''
ask = "Como fazer um requerimento"

messages = [{"role": "system", "content": chunk} for chunk in vector]


client = OpenAI(api_key='ollama', base_url='http://localhost:11435/v1/')

response = client.chat.completions.create(
  model='llama3',
  messages = [
  
    {"role": "user", "content": t},
  ]
)

print(response.choices[0].message.content)
'''
import time

inicio = time.time()
# %%
from ollama import Client
from ollama import AsyncClient

async def connect_to_aioprompt():
    client = AsyncClient(host="http://localhost:11435")
    
    return client

async def chat(message):
    client = await connect_to_aioprompt()

    stream =  await client.chat(model='llama3', messages=[{"role": "user", "content": message}], stream=True,options=
                                {"seed": 123,"top_k": 20,
                                 "top_p": 0.9,
                                 "temperature": 0})
    
    async for part in  stream:
        yield part



async def chat2(message):
    client = await connect_to_aioprompt()

    query = searchQuery(message)
    # query = message

    stream =  await client.chat(model='llama3', messages=[{"role": "user", "content": query}], stream=True, options={'num_gpu':1})
    
    async for part in  stream:
            print(f"Streamed part: {json.dumps(part)}")
            # decoded_part = json.loads(part)
            yield f"{json.dumps(part)}\n"
        
            
    

#for chunck in stream:
#  print(chunck['message']['content'], end='', flush=True)


minutos_decimal = (time.time() - inicio) / 60

minutos_inteiros = int(minutos_decimal)

segundos_decimal = (minutos_decimal - minutos_inteiros) * 60

segundos = int(segundos_decimal)

print(f"\n\n{minutos_inteiros} minutos e {segundos} segundos")

'''
# %%
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# %%


load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)




model = genai.GenerativeModel('gemini-1.0-pro')

response = model.generate_content(t)

to_markdown(response.text)


# %% [markdown]
# - bert eh encoded nao decoded.
# - olhar modelos da apple - pegar link com gabes (open elm) https://huggingface.co/apple/OpenELM
# - 


'''