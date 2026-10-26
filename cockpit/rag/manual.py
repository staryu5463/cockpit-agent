"""Owner's-manual retrieval tool (RAG module).

A tiny, self-written manual corpus plus a dependency-free BM25 over Chinese
character bigrams. The interesting part is not the retriever: it is letting
the agent decide *when* a request needs the manual and when it needs a
vehicle command.
"""
from __future__ import annotations

import math
from collections import Counter

MANUAL = [
    ("man01", "胎压报警灯", "仪表盘出现胎压报警灯（马蹄形内带感叹号）表示至少一个轮胎气压偏低。请尽快减速并在安全处停车检查，按车门内侧标签上的标准气压充气。补气后行驶几公里，报警灯会自动熄灭。"),
    ("man02", "充电口打开", "停车并解锁车辆后，按压充电口盖后部即可弹开；也可以在中控屏“车辆设置-充电”中点击“打开充电口”。低温环境下充电口盖可能结冰，请先除冰再操作。"),
    ("man03", "雨刮更换", "抬起雨刮臂至垂直位置，按住雨刮片与雨刮臂连接处的卡扣，沿雨刮臂方向滑出旧雨刮片，再将新雨刮片推入直到听到咔嗒声。建议每 6 至 12 个月更换一次。"),
    ("man04", "前挡风除雾", "按下空调面板上的前除霜按钮，系统会自动开启外循环、提高风量并对准前挡风玻璃吹风。同时打开压缩机（A/C）可以更快除去雾气。"),
    ("man05", "儿童锁", "儿童锁位于后门内侧边缘。将拨杆拨到锁止位置后，后门只能从车外打开。也可以在中控屏“车辆设置-门锁”中开启电子儿童锁。"),
    ("man06", "保养周期", "常规保养建议每 10000 公里或 12 个月进行一次，以先到者为准，内容包括更换机油、机油滤清器和检查制动系统。新能源车型需定期检查电池冷却液。"),
    ("man07", "蓝牙连接", "在中控屏“设置-蓝牙”中打开蓝牙并选择“添加设备”，在手机上搜索车辆名称并配对。首次连接需在两端确认配对码。最多可保存 5 台设备。"),
    ("man08", "疲劳提醒", "车辆会根据方向盘操作和驾驶时长判断疲劳程度。连续驾驶超过 4 小时或检测到疲劳特征时，仪表会提示休息。可在“驾驶辅助-疲劳提醒”中调整灵敏度。"),
    ("man09", "发动机故障灯", "发动机故障灯常亮表示排放或发动机控制系统存在故障，应尽快到服务站检查；如果故障灯闪烁，说明可能存在严重失火，应立即降低车速并尽快停车。"),
    ("man10", "后视镜加热", "开启后挡风除霜时，外后视镜加热会同时启动，约 15 分钟后自动关闭，用于去除镜面上的雾气和薄冰。"),
    ("man11", "座椅记忆", "调整好座椅位置后，按住座椅侧面的 SET 键，再按 1 或 2 号键即可保存。下次按对应数字键，座椅会自动恢复到保存的位置。"),
    ("man12", "一键启动失灵", "如果钥匙电量不足导致无法启动，可将钥匙贴近转向柱下方的感应区域，再踩下制动踏板并按下启动按钮。"),
]


def _bigrams(text):
    text = "".join(ch for ch in text if not ch.isspace())
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [_bigrams(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = _bigrams(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


_INDEX = BM25([title + " " + body for _, title, body in MANUAL])


def search_manual(query: str, k: int = 3):
    scores = _INDEX.scores(query)
    order = sorted(range(len(MANUAL)), key=lambda i: -scores[i])[:k]
    return [{"id": MANUAL[i][0], "title": MANUAL[i][1], "text": MANUAL[i][2],
             "score": round(scores[i], 3)} for i in order if scores[i] > 0]


# Questions that should be answered from the manual, by family (D = expression OOD).
MANUAL_QUESTIONS = {
    "man01": {"A": ["胎压灯亮了怎么办"], "B": ["仪表盘上胎压报警是什么意思"], "C": ["轮胎气压低的提示灯亮了"], "D": ["那个马蹄形的黄灯一直亮着咋回事"]},
    "man02": {"A": ["充电口怎么打开"], "B": ["怎么打开充电口盖"], "C": ["充电口盖打不开"], "D": ["我要充电，口子在哪开"]},
    "man03": {"A": ["雨刮怎么更换"], "B": ["怎么换雨刮片"], "C": ["雨刮片多久换一次"], "D": ["刮水器那个胶条怎么拆"]},
    "man04": {"A": ["前挡风起雾怎么办"], "B": ["怎么给前挡风除雾"], "C": ["前挡玻璃全是雾"], "D": ["前面玻璃看不清了，雾蒙蒙的"]},
    "man05": {"A": ["儿童锁怎么开"], "B": ["怎么打开儿童锁"], "C": ["后门儿童锁在哪里"], "D": ["孩子老开后门，怎么锁住"]},
    "man06": {"A": ["多久保养一次"], "B": ["保养周期是多少"], "C": ["保养要做哪些项目"], "D": ["车子跑了一万公里了要不要去店里"]},
    "man07": {"A": ["蓝牙怎么连接"], "B": ["手机怎么连车机蓝牙"], "C": ["蓝牙配对怎么操作"], "D": ["手机连不上车，配对在哪弄"]},
    "man09": {"A": ["发动机故障灯亮了怎么办"], "B": ["发动机故障灯闪烁是什么意思"], "C": ["故障灯常亮还能开吗"], "D": ["仪表上那个发动机图标亮了要紧吗"]},
    "man11": {"A": ["座椅记忆怎么设置"], "B": ["怎么保存座椅位置"], "C": ["座椅记忆功能怎么用"], "D": ["每次换人开都要重新调座椅，能记住吗"]},
}
