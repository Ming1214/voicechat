import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

import traceback

import os
import asyncio
import numpy as np

import uvicorn
from urllib.parse import parse_qs
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.models import VADModel, ASRModel
from src.pipeline import Input, VAD, ASR, Output


app = FastAPI()


MODEL_ROOT_PATH = "Path_to_Model's_Dir"
vad_model_path = MODEL_ROOT_PATH + "speech_fsmn_vad_zh-cn-16k-common-pytorch"
spk_model_path = MODEL_ROOT_PATH + "speech_campplus_sv_zh_en_16k-common_advanced"
asr_model_path = MODEL_ROOT_PATH + "SenseVoiceSmall"
pun_model_path = MODEL_ROOT_PATH + "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
speakers_path = "./speakers"
vad_model = VADModel(vad_model_path)
asr_model = ASRModel(spk_model_path, asr_model_path, pun_model_path, speakers_path)


@app.websocket("/stt")
async def websocket_endpoint(websocket: WebSocket):
    try:
        params = websocket.scope['query_string'].decode()
        logger.info(f"Params String: {params}")
        params = parse_qs(params)
        logger.info(f"Params: {params}")
        speaker_verify = params.get('speaker_verify', [None])[0]
        threshold = float(params.get('threshold', ['0.5'])[0])
        language_check = params.get('language_check', ['true'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        use_itn = params.get('use_itn', ['true'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        add_punctuations = params.get('add_punctuations', ['true'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        use_corrector = params.get('use_corrector', ['true'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        await websocket.accept()
        
        inp_queue = asyncio.Queue()
        outp_queue = asyncio.Queue()
        queue_from_vad_to_asr = asyncio.Queue()

        inp = Input(websocket, inp_queue)
        outp = Output(websocket, outp_queue)
        vad = VAD(vad_model, inp_queue, queue_from_vad_to_asr, logger = logger)
        asr = ASR(asr_model, queue_from_vad_to_asr, outp_queue, logger = logger, 
                  speaker_verify = speaker_verify, threshold = threshold, language_check = language_check, 
                  use_itn = use_itn, add_punctuations = add_punctuations, use_corrector = use_corrector)
        await asyncio.gather(inp.run(), outp.run(), vad.run(), asr.run())

        queues = [inp_queue, outp_queue, queue_from_vad_to_asr]
        for queue in queues:
            await queue.join()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"Unexpected error: {e}")
        logger.error(error)
        await websocket.close()
    finally:
        logger.info("Cleaned up resources after WebSocket disconnect")


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 7016)

