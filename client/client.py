import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s \n %(message)s \n')
logger = logging.getLogger()

import os
import fire
import traceback
import re
import json
import zhon
import string
import pyaudio
import audioop
import httpx
import websockets
from openai import OpenAI
import asyncio
from multiprocessing import Process, Pipe, SimpleQueue


FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
INPUT_RATE = 16000
OUTPUT_RATE = 22050


class Recorder:

    async def read(self, ):
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format = FORMAT, channels = CHANNELS, rate = INPUT_RATE, frames_per_buffer = CHUNK, input = True)
            while True:
                data = stream.read(CHUNK)
                yield data
        except Exception as e:
            print(f"[Recorder] Error: {e}")
            traceback.print_exc()
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def run(self, pipe_from_stt):
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format = FORMAT, channels = CHANNELS, rate = INPUT_RATE, frames_per_buffer = CHUNK, input = True)
            while True:
                data = stream.read(CHUNK)
                pipe_from_stt.send(data)
        except Exception as e:
            print(f"[Recorder] Error: {e}")
            traceback.print_exc()
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


class Player:

    def run(self, pipe_from_tts):
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format = FORMAT, channels = CHANNELS, rate = OUTPUT_RATE, frames_per_buffer = CHUNK, output = True)
            while True:
                data = pipe_from_tts.recv()
                stream.write(data)
        except Exception as e:
            print(f"[Player] Error: {e}")
            traceback.print_exc()
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


class STTClient:

    def __init__(self, config):
        try: self.url = config.pop("url")
        except: raise Exception("Missing STT url!")
        self.params_url = self.url + "?"
        for key, value in config.items():
            self.params_url += f"&{key}={value}"
        self.stop_at_vad_start = config.get("speaker_verify") is not None
        self.recoder = Recorder()

    async def send_audio(self, websocket, pipe_from_recoder):
        while True:
            data = pipe_from_recoder.recv()
            await websocket.send(data)
            await asyncio.sleep(0)

    async def receive_text(self, websocket, queue_to_llm):
        while True:
            data = await websocket.recv()
            logger.info(f"[STT] {data}")
            if data == "[VAD Start]":
                if self.stop_at_vad_start:
                    queue_to_llm.put("") # then LLM will send Stop signal to TTS
            else:
                text = json.loads(data)["text"]
                if text:
                    queue_to_llm.put(text)

    async def stt(self, pipe_from_recoder, queue_to_llm):
        async with websockets.connect(self.params_url) as websocket:
            await asyncio.gather(
                self.send_audio(websocket, pipe_from_recoder), 
                self.receive_text(websocket, queue_to_llm)
            )

    def run(self, pipe_from_recoder, queue_to_llm):
        try:
            asyncio.run(self.stt(pipe_from_recoder, queue_to_llm))
        except Exception as e:
            print(f"[STT] Error: {e}")
            traceback.print_exc()


class LLMClient:

    def __init__(self, config):
        self.base_url = config.pop("base_url")
        self.api_key = config.pop("api_key")
        self.model = config.pop("model")
        self.temperature = config.get("temperature", 1.0)
        self.max_tokens = config.get("max_tokens", 128)
        self.system = config.get("system")
        if self.system:
            self.messages = [{"role": "system", "content": self.system}]
        else:
            self.messages = []
        self.client = OpenAI(base_url = self.base_url, api_key = self.api_key)

    def cut_sentence(self, prefix, delta):
        stops = ['．', '！', '？', '｡', '。', '?', '!', '~', '—', '…', '...', '\n', '\r']
        if delta:
            pos = -1
            for char in stops:
                pos = max(pos, delta.rfind(char))
            if pos >= 0:
                sent = prefix + delta[: pos+1]
                prefix = delta[pos+1: ]
            else:
                sent = None
                prefix += delta
        else: sent = None
        return sent, prefix

    def llm(self, queue_from_stt, queue_from_tts, queue_to_tts):
        user = ""
        while True:
            while not queue_from_stt.empty() or len(user) == 0:
                user += queue_from_stt.get()
                queue_to_tts.put(None)
            assistant = ""
            while not queue_from_tts.empty():
                assistant += queue_from_tts.get()
            if assistant:
                self.messages.append({"role": "assistant", "content": assistant})
            self.messages.append({"role": "user", "content": user})
            logger.info(f"[LLM] User: {user}")
            user = ""
            request = self.client.chat.completions.create(
                model = self.model, messages = self.messages, 
                temperature = self.temperature, max_tokens = self.max_tokens, 
                stream = True)
            if not queue_from_stt.empty():
                queue_to_tts.put(None)
                request.response.close()
                continue
            prefix = ""
            for chunk in request:
                if not queue_from_stt.empty():
                    queue_to_tts.put(None)
                    request.response.close()
                    break
                delta = chunk.choices[0].delta.content
                sent, prefix = self.cut_sentence(prefix, delta)
                if sent:
                    logger.info(f"[LLM] Sent: {sent}")
                    queue_to_tts.put(sent)
            if prefix:
                logger.info(f"[LLM] Prefix: {prefix}")
                queue_to_tts.put(prefix)

    def run(self, queue_from_stt, queue_from_tts, queue_to_tts):
        try:
            self.llm(queue_from_stt, queue_from_tts, queue_to_tts)
        except Exception as e:
            print(f"[LLM] Error: {e}")
            traceback.print_exc()


class TTSClient:

    def __init__(self, config):
        self.url = config.pop("url")
        self.speaker = config.pop("speaker")
        self.stream = config.pop("stream")
        self.client = httpx.AsyncClient(timeout = None)

    async def request(self, sent):
        #data = {"text": sent, "speaker": self.speaker, "stream": self.stream}
        data = {"tts_text": sent, "spk_id": self.speaker}
        req = self.client.build_request("GET", self.url, data = data)
        logger.info(f"[TTS] Request: {sent}")
        return await self.client.send(req, stream = True)

    async def requests(self, sents):
        tasks = [self.request(sent) for sent in sents]
        return await asyncio.gather(*tasks)

    def clean_sent(self, sent):
        punctuations = zhon.hanzi.punctuation+string.punctuation
        keep_punctuations = ['．', '！', '？', '｡', '。', '?', '!', '，', ',', '；', ';', '.']
        sent = sent.strip()
        if re.search(f"[^{punctuations}\s]", sent):
            for char in punctuations:
                if char not in keep_punctuations:
                    sent = sent.replace(char, " ")
            sent = re.sub("\s+", " ", sent)
            return sent
        else: return None

    def group_sents(self, sents):
        grouped_sents = []
        cleaned_sents = []
        original_sents = []
        for sent in sents:
            original_sents.append(sent)
            cleaned_sent = self.clean_sent(sent)
            if cleaned_sent:
                grouped_sents.append("".join(original_sents))
                cleaned_sents.append(cleaned_sent)
                original_sents = []
        return grouped_sents, cleaned_sents, original_sents

    def check(self, sents, queue_from_llm):
        while not queue_from_llm.empty():
            sent = queue_from_llm.get()
            if sent: sents.append(sent)
            else: return False, []
        return True, sents

    async def tts(self, queue_from_llm, queue_to_llm, pipe_to_player):
        sents = []
        while True:
            while not queue_from_llm.empty() or len(sents) == 0:
                sent = queue_from_llm.get()
                if sent: sents.append(sent)
                else: sents = []
            grouped_sents, cleaned_sents, sents = self.group_sents(sents)
            responses = await self.requests(cleaned_sents)
            continue_tts, sents = self.check(sents, queue_from_llm)
            for grouped_sent, cleaned_sent, response in zip(grouped_sents, cleaned_sents, responses):
                if continue_tts:
                    logger.info(f"[TTS] Start: {grouped_sent}")
                    async for chunk in response.aiter_bytes(chunk_size = CHUNK):
                        continue_tts, sents = self.check(sents, queue_from_llm)
                        if continue_tts:
                            pipe_to_player.send(chunk)
                            new_grouped_sents, new_cleaned_sents, sents = self.group_sents(sents)
                            if len(new_cleaned_sents) > 0:
                                new_responses = await self.requests(new_cleaned_sents)
                                grouped_sents.extend(new_grouped_sents)
                                cleaned_sents.extend(new_cleaned_sents)
                                responses.extend(new_responses)
                        else:
                            logger.info(f"[TTS] Break: {grouped_sent}")
                            break
                    else:
                        logger.info(f"[TTS] Finished: {grouped_sent}")
                        queue_to_llm.put(grouped_sent)
                await response.aclose()

    def run(self, queue_from_llm, queue_to_llm, pipe_to_player):
        try:
            asyncio.run(self.tts(queue_from_llm, queue_to_llm, pipe_to_player))
        except Exception as e:
            print(f"[TTS] Error: {e}")
            traceback.print_exc()


def run_recorder(pipe_to_stt):
    recorder = Recorder()
    recorder.run(pipe_to_stt)


def run_player(pipe_from_tts):
    player = Player()
    player.run(pipe_from_tts)


def run_stt(config, pipe_from_recoder, queue_to_llm):
    stt = STTClient(config)
    stt.run(pipe_from_recoder, queue_to_llm)


def run_llm(config, queue_from_stt, queue_from_tts, queue_to_tts):
    llm = LLMClient(config)
    llm.run(queue_from_stt, queue_from_tts, queue_to_tts)


def run_tts(config, queue_from_llm, queue_to_llm, pipe_to_player):
    tts = TTSClient(config)
    tts.run(queue_from_llm, queue_to_llm, pipe_to_player)


def main(config_path):
    logger.info("*"*20 + "  Config  " + "*"*20)
    with open(config_path, "r", encoding = "utf-8") as f:
        config = json.load(f)
    logger.info(json.dumps(config, ensure_ascii = False, indent = 2))
    logger.info("*"*50)
    pipe_from_recoder_to_stt = Pipe()
    pipe_from_tts_to_player = Pipe()
    queue_from_stt_to_llm = SimpleQueue()
    queue_from_llm_to_tts = SimpleQueue()
    queue_from_tts_to_llm = SimpleQueue()
    p1 = Process(target = run_recorder, args = (pipe_from_recoder_to_stt[0], ))
    p2 = Process(target = run_player, args = (pipe_from_tts_to_player[1], ))
    p3 = Process(target = run_stt, args = (config["stt"], pipe_from_recoder_to_stt[1], queue_from_stt_to_llm))
    p4 = Process(target = run_llm, args = (config["llm"], queue_from_stt_to_llm, queue_from_tts_to_llm, queue_from_llm_to_tts))
    p5 = Process(target = run_tts, args = (config["tts"], queue_from_llm_to_tts, queue_from_tts_to_llm, pipe_from_tts_to_player[0]))
    processes = [p1, p2, p3, p4, p5]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    fire.Fire(main)


