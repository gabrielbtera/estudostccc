# https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/

from langchain.text_splitter import MarkdownTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import pymupdf4llm
from os.path import isfile, join
from pypdf import PdfWriter
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

PDF_PATH = "./pdf"
SINGLE_PDF_PATH = "./pdf/ecc.pdf"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_MODEL = "models/text-embedding-004"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
ATLAS_CONNECTION_URL = "mongodb://user:pass@localhost:27019/?directConnection=true"
ATLAS_DB_NAME = "search_db"
ATLAS_COLLECTION_NAME = "search_collection"
ATLAS_SEARCH_INDEX_NAME = "vsearch"
QUESTION = "Como preencher um requerimento de credenciamento?"

def load_merged_docs():
    doc_list = [join(PDF_PATH, f) for f in os.listdir(PDF_PATH) if isfile(join(PDF_PATH, f))]
    print(f"Doc list: {doc_list}")
    merger = PdfWriter()
    for file_path in doc_list:
        merger.append(file_path)
    merger.write(join(PDF_PATH, "merged_docs.pdf"))
    merger.close()

    md_text = pymupdf4llm.to_markdown(join(PDF_PATH, "merged_docs.pdf"), page_chunks=True)
    text_splitter = MarkdownTextSplitter()
    #text_splitter = SemanticChunker(setup_gemini_embeddings())
    return text_splitter.create_documents([i['text'] for i in md_text], [i['metadata'] for i in md_text])

def load_docs_from_directory():
    doc_list = [join(PDF_PATH, f) for f in os.listdir(PDF_PATH) if isfile(join(PDF_PATH, f))]
    print(f"Doc list: {doc_list}")
    
    all_docs_list = []
    for file_path in doc_list:
        all_docs_list += pymupdf4llm.to_markdown(file_path, page_chunks=True)
    print(f"all_docs_list lenght: {len(all_docs_list)}")
    
    text_splitter = MarkdownTextSplitter()
    #text_splitter = SemanticChunker(setup_gemini_embeddings())
    return text_splitter.create_documents([i['text'] for i in all_docs_list], [i['metadata'] for i in all_docs_list])

def load_single_doc():
    md_text = pymupdf4llm.to_markdown(SINGLE_PDF_PATH, page_chunks=True)
    print(type(md_text))
    text_splitter = MarkdownTextSplitter()
    return text_splitter.create_documents([i['text'] for i in md_text], [i['metadata'] for i in md_text])

def setup_gemini_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY
    )
    return embeddings

def setup_atlas(embeddings):
    client = MongoClient(ATLAS_CONNECTION_URL)
    atlas_collection = client[ATLAS_DB_NAME][ATLAS_COLLECTION_NAME]
    
    print(f'Db exists: {len([a for a in client.list_database_names() if a == "search_db"]) == 1}')
    if len([a for a in client.list_database_names() if a == "search_db"]) == 1: #database exists
        return True
    else:
        docs = load_docs_from_directory()
        #docs = load_merged_docs()
        #docs = load_single_doc()
        
        print("Inserting docs...")
        MongoDBAtlasVectorSearch.from_documents(
            documents = docs,
            embedding = embeddings,
            collection = atlas_collection,
            index_name = ATLAS_SEARCH_INDEX_NAME
        )
        
        return setup_search_index_if_not_exists(atlas_collection)


def setup_search_index_if_not_exists(atlas_collection):
    # OBS.: for text_embedding_004, dimensions = 768

    search_index_definition = {
        "definition": {
            "mappings": {
                "dynamic": True, 
                "fields": {
                    "embedding" : {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector"
                    }
                }
            }
        },
        "name": ATLAS_SEARCH_INDEX_NAME
    }

    search_index = atlas_collection.list_search_indexes().try_next()
    if search_index is None:
        print("Creating search index...")
        atlas_collection.create_search_index(search_index_definition)
        return False
    else:
        #Armengo caso o indice não funcione de primeira
        if search_index['status'] == "DOES_NOT_EXIST":
            atlas_collection.drop_search_index(ATLAS_SEARCH_INDEX_NAME)
            atlas_collection.create_search_index(search_index_definition)
            return False
        return True

def drop_database():
    client = MongoClient(ATLAS_CONNECTION_URL)
    client.drop_database(name_or_database=ATLAS_DB_NAME)
    print("Database dropped")

def check_index():
    client = MongoClient(ATLAS_CONNECTION_URL)
    atlas_collection = client[ATLAS_DB_NAME][ATLAS_COLLECTION_NAME]
    print(f"Index: {atlas_collection.list_search_indexes().try_next()}")

def run_search_query(question):
    embeddings = setup_gemini_embeddings()
    run_query = setup_atlas(embeddings)
    if run_query:
        print("Running the search query...")
        docsearch = MongoDBAtlasVectorSearch.from_connection_string(
            ATLAS_CONNECTION_URL,
            ATLAS_DB_NAME + "." + ATLAS_COLLECTION_NAME,
            embeddings,
            index_name=ATLAS_SEARCH_INDEX_NAME,
        )
        
        results = docsearch.similarity_search(question, k=10)
        # for doc in results:
        #     print(str(doc))
        # print(results)

        results_list = [doc.page_content + '------ arquivo: [' +doc.metadata.get('file_path') + ']' for doc in results]
        
        return ' '.join(results_list)
       
    else:
        print("Wait until the index gets synced with the dataset")

# drop_database()
#check_index()
run_search_query(QUESTION)