import fire
import asyncio
import aioconsole
import pyaudio
import httpx
import requests
import numpy as np
import traceback

""" speaker 名录
CyberWon分享,仅供学习
请勿商用,别干违法
中文女
中文男
日语男
粤语女
英文女
英文男
韩语女
云健
云夏
云希
云扬
云杰
云枫
云泽
云皓
云野
晓伊
晓双
晓墨
晓悠
晓晓
晓柔
晓梦
晓涵
晓甄
晓睿
晓秋
晓萱
晓辰
晓颜
AzureCN
AzureEN
AzureJA
tts-1
云逸 多语言
剪映
晓宇 多语言
晓晓 多语言
晓辰 多语言
三月七
丹恒
佩拉
停云
克拉拉
刃
卡芙卡
卢卡
可可利亚
史瓦罗
姬子
娜塔莎
寒鸦
尾巴
布洛妮娅
希儿
希露瓦
帕姆
开拓者(女)
开拓者(男)
彦卿
托帕&账账
景元
杰帕德
桂乃芬
桑博
流萤
玲可
瓦尔特
白露
真理医生
砂金
符玄
米沙
素裳
罗刹
艾丝妲
花火
藿藿
虎克
螺丝咕姆
银枝
银狼
镜流
阮•梅
阿兰
雪衣
青雀
驭空
中立
开心
难过
黄泉
黑塔
黑天鹅
七七
丽莎
久岐忍
九条裟罗
云堇
五郎
优菈
八重神子
凝光
凯亚
凯瑟琳
刻晴
北斗
卡维
可莉
嘉明
坎蒂丝
夏沃蕾
夏洛蒂
多莉
夜兰
奥兹
妮露
娜维娅
安柏
宵宫
戴因斯雷布
托马
提纳里
早柚
林尼
枫原万叶
柯莱
派蒙-兴奋说话
派蒙-吞吞吐吐
派蒙-平静
派蒙-很激动
派蒙-疑惑
流浪者
温迪
烟绯
珊瑚宫心海
珐露珊
班尼特
琳妮特
琴
瑶瑶
甘雨
申鹤
白术
砂糖
神里绫人
神里绫华
空
米卡
纳西妲
绮良良
罗莎莉亚
胡桃
艾尔海森
芭芭拉
荒泷一斗
荧
莫娜
莱依拉
莱欧斯利
菲米尼
菲谢尔
萍姥姥
行秋
诺艾尔
赛诺
辛焱
达达利亚
迪卢克
迪奥娜
迪娜泽黛
迪希雅
那维莱特
重云
钟离
闲云
阿贝多
雷泽
雷电将军
香菱
魈
鹿野院平藏
"""


ASYNC = True
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024


async def input_text(queue_texts):
    while True:
        text = await aioconsole.ainput("Input text: ")
        await queue_texts.put(text)


async def send_request(url, speaker, stream, client, queue_texts, queue_responses):
    while True:
        text = await queue_texts.get()
        queue_texts.task_done()
        #data = {"text": text, "speaker": speaker, "stream": stream}
        data = {"tts_text": text, "spk_id": speaker}
        if not ASYNC:
            response = requests.get(url, data = data, stream = True)
        else:
            req = client.build_request("GET", url, data = data)
            response = await client.send(req, stream = True)
        await queue_responses.put(response)


async def receive_audio(queue_responses):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format = FORMAT, channels = CHANNELS, rate = RATE, frames_per_buffer = CHUNK, output = True)
        while True:
            response = await queue_responses.get()
            queue_responses.task_done()
            if not ASYNC:
                for chunk in response.iter_content(chunk_size = 2*CHANNELS*CHUNK):
                    stream.write(chunk)
            else:
                async for chunk in response.aiter_bytes(chunk_size = 2*CHANNELS*CHUNK):
                    stream.write(chunk)
                await response.aclose()
    except Exception as e:
        #print(e)
        traceback.print_exc()
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        raise Exception("Error: receive_audio")


async def run(url, speaker, stream):
    try:
        client = httpx.AsyncClient(timeout = None)
        queue_texts = asyncio.Queue()
        queue_responses = asyncio.Queue()
        await asyncio.gather(
                input_text(queue_texts),
                send_request(url, speaker, stream, client, queue_texts, queue_responses), 
                receive_audio(queue_responses)
        )
    except Exception as e:
        #print(e)
        traceback.print_exc()
    finally:
        await client.aclose()


def main(url, speaker = "中文女", stream = False):
    asyncio.run(run(url, speaker, stream))


if __name__ == "__main__":
    fire.Fire(main)



