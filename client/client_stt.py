import fire
import asyncio
import pyaudio
import audioop
import websockets
from multiprocessing import Process, Pipe


FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
INPUT_RATE = 16000
OUTPUT_RATE = 22050


async def send_audio(websocket, pipe):
    try:
        while True:
            data = pipe.recv()
            await websocket.send(data)
            await asyncio.sleep(0)
    except Exception as e:
        print(f"Error Sending: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def receive_text(websocket, pipe):
    try:
        while True:
            data = await websocket.recv()
            pipe.send(data)
    except Exception as e:
        print(f"Error Receiving: {e}")


async def stt(url, pipe1, pipe2):
    try:
        async with websockets.connect(url) as websocket:
            await asyncio.gather(
                send_audio(websocket, pipe1), 
                receive_text(websocket, pipe2)
            )
    except Exception as e:
        print(e)


def process_audio(url, pipe1, pipe2):
    asyncio.run(stt(url, pipe1, pipe2))


def record_audio(pipe):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format = FORMAT, channels = CHANNELS, rate = INPUT_RATE, frames_per_buffer = CHUNK, input = True)
        while True:
            data = stream.read(CHUNK)
            pipe.send(data)
    except Exception as e:
        print(f"Error Recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def print_text(pipe):
    try:
        while True:
            data = pipe.recv()
            print(data)
    except Exception as e:
        print(f"Error Print: {e}")


def main(url, 
         speaker_verify = "mingr.z", threshold = 0.6, language_check = True, 
         use_itn = True, add_punctuations = True, use_corrector = False):
    try:
        params = []
        if speaker_verify is not None:
            params.append(f"speaker_verify={speaker_verify}")
        if threshold is not None:
            params.append(f"threshold={threshold}")
        if language_check is not None:
            params.append(f"language_check={language_check}")
        if use_itn is not None:
            params.append(f"use_itn={use_itn}")
        if add_punctuations is not None:
            params.append(f"add_punctuations={add_punctuations}")
        if use_corrector is not None:
            params.append(f"use_corrector={use_corrector}")
        if len(params) > 0:
            params_url = url + "?" + "&".join(params)
        else:
            params_url = url
        print("*"*20 + "  Config  " + "*"*20)
        print("\n".join(params))
        print("*"*15 + "   Start Recoding   " + "*"*15)
        pipe1 = Pipe()
        pipe2 = Pipe()
        p1 = Process(target = record_audio, args = (pipe1[0], ))
        p2 = Process(target = process_audio, args = (params_url, pipe1[1], pipe2[0]))
        p3 = Process(target = print_text, args = (pipe2[1], ))
        p1.start()
        p2.start()
        p3.start()
    except Exception as e:
        print(f"Error Process: {e}")
    finally:
        p1.join()
        p2.join()
        p3.join()


if __name__ == "__main__":
    fire.Fire(main)

