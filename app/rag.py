from typing import List
from vectordb import Memory

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import pymupdf4llm
import asyncio
import json


class RagFile:

  def __init__(self, name_file: str) -> None:
    self.name_file = name_file
    self.memory =  Memory(chunking_strategy={"mode": 'sliding_window', "window_size": 500, "overlap": 8})
    self.__save_memory()
  
  def searchQuery(self,prompt):
    vector = [i['chunk'] for i in self.memory.search(prompt, top_n=10)]

    t = f'{prompt} (formate em markdown e use o texto seguinte como base): '
    for i in vector:
      t += i
    
    return t

  def __convert_pdf_for_text(self):
    file = self.name_file
    
    md_text = pymupdf4llm.to_markdown(file, page_chunks=True, write_images=False)

    md_text = [self.__format(i)  for i in md_text]

    brute = ''
    for i in md_text:
      brute += i['text'] 
    return brute
  
  def __format(self,chunck):
    chunck['text'] = f" $$ pag{str(chunck['metadata']['page'])} arq-{chunck['metadata']['file_path']} \n\n\n" + chunck['text']
    return chunck
  
  def __textSpliterMarkdown(self):
    str_file = self.__convert_pdf_for_text()

    padrao = [('$$', 'pagina')]
    md_spliter = MarkdownHeaderTextSplitter(headers_to_split_on=padrao)

    return md_spliter.split_text(str_file)
  
  def __save_memory(self):
    sections = self.__textSpliterMarkdown()

    for i in sections:
      self.memory.save( i.page_content, i.metadata)
  

