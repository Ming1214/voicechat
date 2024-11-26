import re
import json
import zhon
import string
import httpx
import asyncio
import numpy as np


INPUT_CHANNELS = 1           # 接收信号的通道数
INPUT_SAMPLE_RATE = 16000    # 接收信号的采样率
INPUT_CHUNK_SIZE = 1024      # 每次接收的帧数

OUTPUT_CHANNELS = 1          # 发送信号的通道数
OUTPUT_SAMPLE_RATE = 22050   # 发送信号的采样率
OUTPUT_CHUNK_SIZE = 1024     # 每次发送的帧数

VAD_TIME_INTERVAL = 200      # VAD 检测时间长度 (ms)

# 每次接收的字节数
INPUT_BYTE_SIZE = INPUT_CHANNELS*INPUT_CHUNK_SIZE*2

# 每次发送的字节数
OUTPUT_BYTE_SIZE = OUTPUT_CHANNELS*OUTPUT_CHUNK_SIZE*2

# VAD 检测时间长度转化为帧数
VAD_CHUNK_SIZE = int(VAD_TIME_INTERVAL*INPUT_SAMPLE_RATE/1000)

# VAD 检测帧数转化为字节数
VAD_BYTE_SIZE = VAD_CHUNK_SIZE*INPUT_CHANNELS*2


class Input:

    def __init__(self, websocket, inp_queue, logger = None):
        self.websocket = websocket
        self.inp_queue = inp_queue
        self.logger = logger

    async def run(self, ):
        while True:
            data = await self.websocket.receive_bytes()
            await self.inp_queue.put(data)
            if self.logger:
                if isinstance(data, bytes):
                    self.logger.info(f"[Input] Received {len(data)} bytes")
                if isinstance(data, str):
                    self.logger.info(f"[Input] Received text: {data}")


class Output:

    def __init__(self, websocket, outp_queue, logger = None):
        self.websocket = websocket
        self.outp_queue = outp_queue
        self.logger = logger

    async def run(self, ):
        while True:
            data = await self.outp_queue.get()
            self.outp_queue.task_done()
            await self.websocket.send_bytes(data)
            if self.logger:
                if isinstance(data, bytes):
                    self.logger.info(f"[Output] Sent {len(data)} bytes")
                if isinstance(data, str):
                    self.logger.info(f"[Output] Sent text: {data}")


class VAD:

    def __init__(self, model, inp_queue, outp_queue_asr, outp_queue, logger = None):
        self.model = model
        self.inp_queue = inp_queue
        self.outp_queue_asr = outp_queue_asr
        self.cache = {}
        self.offset = 0
        self.logger = logger

    async def run(self, ):
        audio_buffer = b""
        speech_buffer = b""
        start, end = -1, -1
        while True:
            chunk = await self.inp_queue.get()
            self.inp_queue.task_done()
            audio_buffer += chunk
            while len(audio_buffer) >= VAD_BYTE_SIZE:
                vad_chunk = audio_buffer[: VAD_BYTE_SIZE]
                small_chunks = await self.model.infer(vad_chunk, VAD_TIME_INTERVAL, self.cache, self.offset)
                self.offset += VAD_TIME_INTERVAL
                audio_buffer = audio_buffer[VAD_BYTE_SIZE: ]
                speech_buffer += vad_chunk
                for tag, i, j, s, e, t in small_chunks:
                    self.logger.info(f"[VAD] vad chunk: {tag}, {i}: {j}, {s}: {e}, {t}")
                    if s >= 0:
                        start = i+len(speech_buffer)-VAD_BYTE_SIZE
                        await self.outp_queue.put("[VAD Start]")
                        if self.logger:
                            self.logger.info(f"[VAD] vad start")
                    if e >= 0:
                        end = j+len(speech_buffer)-VAD_BYTE_SIZE
                        if self.logger:
                            self.logger.info(f"[VAD] vad end")
                    if 0 <= start <= end:
                        speech = speech_buffer[start: end]
                        await self.outp_queue_asr.put(speech)
                        speech_buffer = speech_buffer[end: ]
                        start, end = -1, -1
                        if self.logger:
                            self.logger.info(f"[VAD] vad length: {len(speech)/INPUT_CHANNELS/INPUT_SAMPLE_RATE/2*1000:.0f} ms")


class ASR:

    def __init__(self, model, inp_queue_vad, outp_queue, logger = None, 
                 speaker_verify = None, threshold = 0.5, language_check = True, 
                 use_itn = True, add_punctuations = True, use_corrector = True):
        self.model = model
        self.inp_queue_vad = inp_queue_vad
        self.outp_queue = outp_queue
        self.logger = logger
        self.speaker_verify = speaker_verify
        self.threshold = threshold
        self.language_check = language_check
        self.use_itn = use_itn
        self.add_punctuations = add_punctuations
        self.use_corrector = use_corrector

    async def run(self, ):
        while True:
            speech = await self.inp_queue_vad.get()
            self.inp_queue_vad.task_done()
            result = await self.model.infer(speech, 
                                            self.speaker_verify, self.threshold, self.language_check, 
                                            self.use_itn, self.add_punctuations, self.use_corrector)
            await self.outp_queue.put(json.dumps(result, ensure_ascii = False, indent = 2))
            if self.logger:
                if result["speaker_verify_info"] is not None:
                    self.logger.info(f"[ASR] speaker verify info: {result['speaker_verify_info']}")
                if result["check_language"] is not None:
                    self.logger.info(f"[ASR] check language: {result['check_language']}")
                if result["raw_text"]:
                    self.logger.info(f"[ASR] raw text: {result['raw_text']}")
                if result["text"]:
                    self.logger.info(f"[ASR] text: {result['text']}")

