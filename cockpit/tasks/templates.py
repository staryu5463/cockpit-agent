"""Natural-language templates, grouped into families.

Family "D" of every primitive is never used for training: it is the
expression-OOD test set ("did the model learn the skill, or the sentence?").
"""
from __future__ import annotations

ZONE_WORDS = {
    "driver": ["主驾", "驾驶位", "我这边"],
    "passenger": ["副驾", "副驾驶", "副驾那边"],
    "all": ["", "全车"],
}
NUM_CN = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六"}

TEMPLATES = {
    "AC_ON": {
        "A": ["打开空调", "空调打开"],
        "B": ["开一下空调", "帮我开空调"],
        "C": ["空调开起来", "把空调开了"],
        "D": ["把冷气开开", "整点冷气"],
    },
    "AC_OFF": {
        "A": ["关闭空调", "空调关掉"],
        "B": ["关一下空调", "帮我关空调"],
        "C": ["空调关了吧", "不用开空调了"],
        "D": ["冷气别开了", "冷气停了吧"],
    },
    "TEMP_ABS": {
        "A": ["把{zone}空调温度调到{v}度", "{zone}温度调到{v}度"],
        "B": ["{zone}温度设成{v}度", "{zone}空调设置为{v}度"],
        "C": ["{zone}空调{v}度", "{zone}{v}度"],
        "D": ["麻烦帮我把{zone}那边弄到{v}度吧"],
    },
    # {dir}: 低/高, {n}: Chinese number; "_dflt" variants use the default step
    "TEMP_REL": {
        "A": ["{zone}温度再{dir}{n}度", "{zone}温度调{dir}{n}度"],
        "B": ["{zone}空调{dir}{n}度", "{zone}再调{dir}{n}度"],
        "C": ["{zone}有点{feel}，调{dir}{n}度", "{zone}{feel}，温度{dir}{n}度"],
        "D": ["感觉{zone}{feel}了，弄{dir}{n}度"],
    },
    "TEMP_REL_dflt": {
        "A": ["{zone}温度再{dir}一点"],
        "B": ["{zone}空调调{dir}一点"],
        "C": ["{zone}有点{feel}"],
        "D": ["{zone}好像{feel}了些"],
    },
    "VOL_ABS": {
        "A": ["音量调到{v}", "把音量调到{v}"],
        "B": ["声音开到{v}", "音量设成{v}"],
        "C": ["音量{v}", "声音{v}"],
        "D": ["音量给我弄成{v}吧"],
    },
    "VOL_REL": {
        "A": ["音量调{dir}{n}格", "声音调{dir}{n}格"],
        "B": ["声音再{dir}{n}格", "音量再{dir}{n}格"],
        "C": ["音量{dir}{n}格", "声音{dir}{n}个格"],
        "D": ["音量往{dir}拧{n}格"],
    },
    "VOL_REL_dflt": {
        "A": ["声音{dir}一点", "音量{dir}一点"],
        "B": ["音量调{dir}点", "声音再{dir}点"],
        "C": ["{feel}"],
        "D": ["声音有点{dirfeel}"],
    },
    "NAV_NEAREST": {
        "A": ["导航去最近的{cat}", "导航到最近的{cat}"],
        "B": ["带我去附近的{cat}", "去附近的{cat}"],
        "C": ["找个最近的{cat}", "最近的{cat}在哪，导航过去"],
        "D": ["附近哪有{cat}，带我过去"],
    },
    "NAV_BRAND": {
        "A": ["导航到{brand}", "导航去{brand}"],
        "B": ["去{brand}", "我要去{brand}"],
        "C": ["带我去{brand}", "开去{brand}"],
        "D": ["往{brand}那边走"],
    },
    "PLAY_SONG": {
        "A": ["播放{artist}的{title}", "放{artist}的{title}"],
        "B": ["放一首{title}", "播放{title}"],
        "C": ["来首{artist}的{title}", "来一首{title}"],
        "D": ["想听{title}"],
    },
    "PLAY_ARTIST": {
        "A": ["放{artist}的歌", "播放{artist}的歌"],
        "B": ["来点{artist}的歌", "放首{artist}的"],
        "C": ["来点{artist}", "播放{artist}的歌曲"],
        "D": ["我想听{artist}"],
    },
}

CHAT_TEXTS = {
    "A": ["今天好累啊", "哈哈哈太好笑了", "刚才那个电影真不错"],
    "B": ["你叫什么名字", "给我讲个笑话", "你觉得明天会下雨吗"],
    "C": ["唉，堵死了", "这首歌的歌词好有意思", "周末去哪玩好呢"],
    "D": ["老板今天又骂我了", "你说人生有什么意义"],
}

CONNECTORS = {
    "A": ["，然后", "，再"],
    "B": ["，顺便", "；"],
    "C": ["，另外", "，同时"],
    "D": ["，对了，"],
}
