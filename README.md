# VoiceChat

本项目是本人为了熟悉 STT、LLM、TTS 之间的对接与协同机制而创建的，支持说话人判断、流式语音识别、流式语音输出、实时打断。

### 模型

#### STT

- 语音端点检测模型：speech_fsmn_vad_zh-cn-16k-common-pytorch
- 说话人判断模型：speech_campplus_sv_zh_en_16k-common_advanced
- ASR 模型：SenseVoiceSmall
- 标点模型：punc_ct-transformer_zh-cn-common-vocab272727-pytorch（SenceVoiceSmall 可以自带标点，因此该模型非必须）

#### LLM

- openai 的各 chat 模型
- 可以调用 openai api 的模型（例如采用 vLLM openai api server 部署的模型）

#### TTS

- CosyVoice-300M-SFT

### 部署

- STT server 请用本项目代码进行部署
- TTS server 请按照 CosyVoice 官方提供的 FastAPI server 进行部署：[CosyVoice/runtime/python/fastapi/server.py at main · FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/python/fastapi/server.py)
- LLM 可以使用 OpenAI 模型，或者自己的模型，推荐 vLLM openai api server 部署
- client_sst.py 提供 SST 单任务测试
- client_tts.py 提供 TTS 单任务测试
- client.py 为完整语音对话入口

### 致谢

- STT 部分的流式处理方法借鉴了项目：[0x5446/api4sensevoice: API and websocket server for sensevoice. It has inherited some enhanced features, such as VAD detection, real-time streaming recognition, and speaker verification.](https://github.com/0x5446/api4sensevoice)
- CosyVoice 可以使用更多音色：[CosyVoice 170+音色免费分享，真的强！生成式语音中当之无愧的No.1!生成的语音韵律和参考音频有关系，比如中间的停顿。_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Qb421E74S?vd_source=ac4afcff4782934e9da19b5e64438db8&spm_id_from=333.788.videopod.sections)