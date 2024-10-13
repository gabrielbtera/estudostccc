from gemini import getResponseGemini
from ollama_run import chat
import asyncio
from quart_cors import cors
import json
from quart import Quart, request, stream_with_context, make_response


app = Quart(__name__)
cors(app, allow_origin="http://localhost:4200") 


def setHeaders(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:4200'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'


@app.route('/gemini', methods=['POST'])
async def get_response_gemini():
    data = await request.get_json()
    prompt = data.get('prompt')
    parametro = data.get('parametro')

    response = await make_response(getResponseGemini(prompt))
    response.content_type = 'application/json' 
    return response


@app.route('/', methods=['POST'])
async def stream_response():
    data = await request.get_json()
    prompt = data.get('prompt')
    parametro = data.get('parametro')
    

    try: 
        
        response = await make_response(chat(prompt)) 
        response.timeout = None
        setHeaders(response)
    
        response.content_type = 'application/json' 
        return response
        
    except:
        error_response = {"error": 'Aconteceu um problema no servidor, tente mais tarde!'}
        
        response = await make_response(error_response, 500)
        print(response)
        setHeaders(response)
        response.content_type = 'application/json'
        return response



    
    
    
if __name__ == '__main__':
    asyncio.run(app.run(port=8000))

