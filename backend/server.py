import asyncio
import websockets
import numpy as np
from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='client/build')

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

async def audio_conversion(websocket, path):
    try:
        async for message in websocket:
            audio_data = np.frombuffer(message, dtype=np.int16)
            print(f"Received data: {audio_data}")

            # 音声変換処理（例：単純な増幅）
            converted_audio_data = audio_data * 2
            converted_audio_data = converted_audio_data.astype(np.int16)

            # 変換後の音声データを送信
            await websocket.send(converted_audio_data.tobytes())
    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except Exception as e:
        print(f"Error during audio conversion: {e}")

start_server = websockets.serve(audio_conversion, "0.0.0.0", 8765)

if __name__ == "__main__":
    # Flaskサーバーを別スレッドで実行
    # from threading import Thread
    # flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    # flask_thread.start()

    # WebSocketサーバーを実行
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
