from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from text_filters import is_noise_message

INPUT_CSV = Path("message.csv")
OUTPUT_JSON = Path("content_topic_analysis.json")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

# 内容类型（约 10 类）
CONTENT_KEYWORDS: dict[str, list[str]] = {
    "求助": ["怎么", "如何", "帮忙", "求助", "有知道", "问一下", "谁有"],
    "分享": ["分享", "推荐", "安利", "链接", "好用", "体验"],
    "吐槽": ["吐槽", "无语", "好烦", "坑", "差评", "垃圾"],
    "交易": ["出售", "转让", "卖", "买", "交易", "团购", "闲置", "出一", "收一"],
    "资讯": ["通知", "公告", "发布", "新闻", "资讯", "更新", "官宣"],
    "生活": ["食堂", "寝室", "宿舍", "外卖", "洗澡", "排队", "水卡", "电费"],
    "学习": ["作业", "考试", "复习", "课程", "教材", "ppt", "题", "答案", "考研"],
    "活动": ["活动", "聚会", "招募", "报名", "比赛", "竞赛", "讲座", "宣讲"],
    "求职": ["兼职", "招聘", "实习", "校招", "面试", "岗位", "投简历"],
    "问候": ["谢谢", "辛苦", "欢迎", "早上好", "晚安", "抱歉", "sorry"],
}

# 话题（约 6 类）
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "饮食": ["吃", "饭", "食堂", "外卖", "奶茶", "夜宵", "早餐", "午饭", "晚饭"],
    "学习": ["作业", "考试", "复习", "课堂", "老师", "成绩", "课程", "考研"],
    "运动": ["跑步", "健身", "篮球", "足球", "羽毛球", "游泳", "骑车", "跳绳"],
    "兼职求职": ["兼职", "招聘", "实习", "校招", "简历", "offer", "岗位"],
    "情感社交": ["表白", "脱单", "情侣", "约饭", "聚会", "社交", "朋友"],
    "生活服务": ["宿舍", "寝室", "快递", "洗澡", "排队", "电费", "水卡", "维修"],
}

# 聊天形式检测
FORMAT_PATTERNS = {
    "图片": [r"\[图片\]", ".jpg", ".png", ".jpeg", ".gif"],
    "语音": [r"\[语音\]", ".amr", ".silk", ".mp3"],
    "视频": [r"\[视频\]", ".mp4", ".mov", ".avi"],
    "表情包": [r"\[表情", r"\[动画表情", "emoji"],
}

STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "我们",
    "你们",
    "他们",
    "以及",
    "然后",
    "还有",
    "就是",
    "不是",
    "可以",
    "知道",
    "欢迎",
    "不会",
    "谢谢",
    "好的",
    "可以吗",
    "一下",
    "有没有",
    "一下吗",
    "有没有人",
    "还有人",
    "请问",
    "真的",
    "那个",
    "这个",
    "一个",
    "吗",
    "啊",
    "吧",
    "哦",
    "嘛",
    "啦",
    "呀",
    "啥",
    "啥子",
}


def match_category(text: str, mapping: dict[str, list[str]], default: str) -> str:
    for category, kws in mapping.items():
        if any(kw in text for kw in kws):
            return category
    return default


def detect_format(text: str) -> str:
    lower = text.lower()
    for fmt, patterns in FORMAT_PATTERNS.items():
        for pat in patterns:
            if pat.startswith("["):
                if re.search(pat, lower):
                    return fmt
            elif pat in lower:
                return fmt
    return "文字"


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if not chunk or chunk in STOPWORDS:
            continue
        if chunk.isdigit():
            continue
        if len(chunk) == 1 and re.match(r"[a-z]", chunk):
            continue
        tokens.append(chunk)
    return tokens


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    content_counter = Counter()
    format_counter = Counter()
    topic_counter = Counter()
    topic_by_day: defaultdict[str, Counter[str]] = defaultdict(Counter)
    word_counter = Counter()

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            content = (row.get("message_content") or "").strip()
            if not content or is_noise_message(content):
                continue
            normalized = content.lower()

            content_cat = match_category(normalized, CONTENT_KEYWORDS, "其他")
            content_counter[content_cat] += 1

            topic_cat = match_category(normalized, TOPIC_KEYWORDS, "其他")
            topic_counter[topic_cat] += 1

            fmt = detect_format(content)
            format_counter[fmt] += 1

            ts_str = row.get("message_time_utc")
            if ts_str:
                try:
                    day = datetime.strptime(ts_str, TIMESTAMP_FMT).strftime("%Y-%m-%d")
                    topic_by_day[day][topic_cat] += 1
                except ValueError:
                    pass

            word_counter.update(tokenize(content))

    data = {
        "contentTypes": [{"name": k, "count": v} for k, v in content_counter.most_common()],
        "formats": [{"name": k, "count": v} for k, v in format_counter.most_common()],
        "topicsTotals": [{"name": k, "count": v} for k, v in topic_counter.most_common()],
        "topicsByDay": [
            {"day": day, "topics": dict(counter)} for day, counter in sorted(topic_by_day.items())
        ],
        "wordcloud": [
            {"name": k, "value": v}
            for k, v in word_counter.most_common(120)
            if k not in STOPWORDS
        ],
    }

    OUTPUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
