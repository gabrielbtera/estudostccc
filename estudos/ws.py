import asyncio
import websockets

from ragteste import chat

async def echo(websocket, path):
    try:
        
        ping_interval = 5  
        while True:
            print('conectou')

            message = await websocket.recv()

            print('recebeu a msg')

            async for chunk in chat(message):
                msg = chunk['message']['content']
                
                await websocket.send(f"{msg}")

            await asyncio.sleep(0)
            if not websocket.closed:
                await websocket.ping()
            else:
                break  

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error: {e}")


async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())
