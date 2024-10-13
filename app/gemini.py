import google.generativeai as genai
import json

import random


import os
from dotenv import load_dotenv
load_dotenv()
from repository import run_search_query

from rag import RagFile

rag = RagFile('../credenciamento.pdf')


api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

def randomSizeChunck():
  return random.randint(1,10)


def formatPrompt(prompt, query):
  metadata = 'Ao final de cada texto usado vai ter algo assim: ------ arquivo: ["nome_arquivo"] ao final da resposta diga quais arquivos foram usados'
  t = f'pergunta: {prompt} (lembre-se que voce responde pela secretaria do estado da fazenda de sergipe -site: https://www.sefaz.se.gov.br/SitePages/default.aspx. {metadata}  formate em markdown e use o texto seguinte como base):' 
  return t + query


async def getResponseGemini(query):
  model = genai.GenerativeModel('gemini-1.0-pro')

  search = formatPrompt(query, run_search_query(query))


  response = await model.generate_content_async(search, stream=True)

  async for chunk in response:
    dic_chunck = {'response': chunk.text} 
    yield f'{json.dumps(dic_chunck)}\n'
