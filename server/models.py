import os
import re
import base64
import httpx
import asyncio
from openai import AsyncOpenAI

import torch
import soundfile
import numpy as np
from funasr import AutoModel
from pycorrector import Corrector


class VADModel:

    def __init__(self, model_path, max_end_silence_time = 200, speech_noise_thres = 0.8):
        self.model = AutoModel(
            model = model_path, 
            max_end_silence_time = max_end_silence_time, 
            speech_noise_thres = speech_noise_thres, 
            disable_pbar = True
        )

    def cut(self, chunk, chunk_size, offset, start, end):
        start_or_end = None
        if start < 0:
            i = -1
            start_or_end = False # 语音结束
        else:
            i = len(chunk)*(start-offset)//chunk_size
            if i % 2 == 1:
                i -= 1
        if end < 0:
            j = -1
            start_or_end = True # 语音开始
        else:
            j = len(chunk)*(end-offset)//chunk_size
            if j % 2 == 1:
                j += 1
        return (start_or_end, i, j, start, end, offset)

    async def infer(self, chunk, chunk_size, cache, offset):
        res = self.model.generate(input = chunk, chunk_size = chunk_size, cache = cache, is_final = False)
        chunks = []
        for start, end in res[0]["value"]:
            chunks.append(self.cut(chunk, chunk_size, offset, start, end))
        return chunks


class ASRModel:

    def __init__(self, spk_model_path, asr_model_path, pun_model_path, speakers_path):
        self.spk_model = AutoModel(model = spk_model_path, disable_pbar = True)
        self.asr_model = AutoModel(model = asr_model_path, disable_pbar = True)
        self.pun_model = AutoModel(model = pun_model_path, disable_pbar = True)
        self.corrector = Corrector()
        self.reg_spks = self.reg_spks_init(speakers_path)

    def reg_spks_init(self, speakers_path):
        reg_spks = {}
        for file in os.listdir(speakers_path):
            file_path = os.path.join(speakers_path, file)
            spk, base = os.path.splitext(os.path.basename(file))
            if os.path.isfile(file_path) and base in [".wav"]:
                data, sample_rate = soundfile.read(file_path, dtype = "float32")
                embedding = self.spk_model.generate(data)[0]["spk_embedding"].flatten()
                embedding /= torch.norm(embedding)
                reg_spks[spk] = {
                    "data": data,
                    "sample_rate": sample_rate, 
                    "embedding": embedding
                }
        return reg_spks

    def format_text_and_patterns(self, text):
        regex = r"<\|[^\|]*\|>"
        emoji_dict = {
        	"<|nospeech|><|Event_UNK|>": "❓",
        	"<|HAPPY|>": "😊", "<|SAD|>": "😔", "<|ANGRY|>": "😡",
        	"<|BGM|>": "🎼", "<|Applause|>": "👏", "<|Laughter|>": "😀",
        	"<|FEARFUL|>": "😰", "<|DISGUSTED|>": "🤢", "<|SURPRISED|>": "😮",
        	"<|Cry|>": "😭", "<|Sneeze|>": "🤧", "<|Cough|>": "😷",
        }
        patterns = re.findall(regex, text)
        patterns = "".join(patterns)
        for pattern, emoji in emoji_dict.items():
            patterns = patterns.replace(pattern, emoji)
        patterns = re.sub(regex, "", patterns)
        texts = re.split(regex, text)
        texts = [text.strip() for text in texts if text.strip()]
        text = " ".join(texts)
        return text, patterns

    async def infer(self, speech, 
                    speaker_verify = None, threshold = 0.5, language_check = True, 
                    use_itn = True, add_punctuations = True, use_corrector = True):
        result = {}
        result["speaker_verify_result"] = None
        result["speaker_verify_info"] = None
        result["scores"] = None
        result["check_language"] = None
        result["corrector"] = None
        if speaker_verify:
            targets = []
            for spk in self.reg_spks:
                if spk.startswith(speaker_verify):
                    targets.append(spk)
            if len(targets) == 0:
                result["speaker_verify_info"] = f"{speaker_verify} not found!"
                result["raw_text"] = ""
                result["text"] = ""
                return result
            source = self.spk_model.generate(speech)[0]["spk_embedding"].flatten()
            source /= torch.norm(source)
            sims, max_spk, max_sim = {}, None, 0.
            for spk in targets:
                target = self.reg_spks[spk]["embedding"]
                sim = float((1.+torch.sum(source*target))/2)
                sims[spk] = np.round(sim, 3)
                if sim > max_sim:
                    max_sim = sim
                    max_spk = spk
                if sim >= threshold:
                    result["speaker_verify_result"] = True
                    result["speaker_verify_info"] = f"{speaker_verify} hit with {spk}: score_from_{spk} = {sim:.2f} >= {threshold:.2f}"
                    result["scores"] = sims
                    break
            else:
                result["speaker_verify_result"] = False
                result["speaker_verify_info"] = f"{speaker_verify} not hit: max_score_from_{max_spk} = {max_sim:.2f} < {threshold:.2f}"
                result["scores"] = sims
                result["raw_text"] = ""
                result["text"] = ""
                return result
        text = self.asr_model.generate(speech, use_itn = use_itn)[0]["text"]
        result["raw_text"] = text
        if language_check:
            if not ("<|zh|>" in text or "<|en|>" in text): # 只允许中文和英文
                result["check_language"] = False
                result["text"] = ""
                return result
            else:
                result["check_language"] = True
        text, patterns = self.format_text_and_patterns(text)
        if text and add_punctuations and not use_itn:
            text = self.pun_model.generate(text)[0]["text"]
        if text and use_corrector:
            res = self.corrector.correct(text)
            result["corrector"] = res["errors"]
            text = res["target"]
        text = text + patterns
        result["text"] = text
        return result

