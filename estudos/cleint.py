import asyncio
import websockets
import time
import os
import platform

def clear_console():
    current_os = platform.system()
    
   
    if current_os == 'Windows':
        os.system('cls')
    else:
        os.system('clear')
async def connect_to_server():
    uri = "ws://localhost:8765"  
    temp_msg = ''
    flag = False
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to server")

                while True:

                    if flag:
                        msg = temp_msg
                        flag = False
                    else:
                        msg = input('send msg: ')
                    if not msg:
                        break
                    
                    await websocket.send(msg)
                    response = await websocket.recv()
                    stri = ''
                    while len(response):
                        stri += response
                        clear_console()
                        print(stri)
                        response = await websocket.recv()
                        

        except websockets.exceptions.ConnectionClosedError:
            print("Connection to server closed, reconnecting...")
            flag = True
            temp_msg = msg
            await asyncio.sleep(0)  
        except Exception as e:
            print(f"Error during connection: {e}")
            break  

async def main():
    await connect_to_server()

if __name__ == "__main__":
    asyncio.run(main())
