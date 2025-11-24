#!/usr/bin/env python3
"""
Generate a conversational intelligence report focused on topic depth,
temporal rhythms, scene association and sentiment from message.csv.
The script reads the CSV once and emits a self-contained HTML file
that visualises the requested indicators.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# Optional Chinese tokenizer. The code falls back to regex based splits
# when jieba is not available in the environment.
try:  # pragma: no cover - optional dependency
    import jieba  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade
    jieba = None

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "message.csv"
OUTPUT_PATH = REPO_ROOT / "conversation_insights.html"
JSON_OUTPUT_PATH = REPO_ROOT / "conversation_insights.json"

STOPWORDS = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "我们",
    "你",
    "你们",
    "他",
    "她",
    "它",
    "他们",
    "她们",
    "而且",
    "以及",
    "并且",
    "还有",
    "这",
    "那",
    "一个",
    "一些",
    "可以",
    "要",
    "有",
    "没",
    "没有",
    "就是",
    "然后",
    "但是",
    "如果",
    "因为",
    "所以",
    "或者",
    "还是",
    "非常",
    "比较",
    "已经",
    "的话",
    "这么",
    "那么",
    "这样",
    "那样",
    "嘛",
    "啊",
    "哦",
    "恩",
    "嗯",
    "吧",
    "呢",
    "啦",
    "噢",
    "喔",
    "与",
    "及",
    "其",
    "并",
    "等",
    "以及",
    "大家",
    "一下",
    "什么",
    "怎么",
    "的话",
    "就是",
    "还是",
    "那个",
    "这个",
    "这里",
    "那里",
    "不会",
    "可能",
    "应该",
    "需要",
    "用于",
    "以及",
    "由于",
    "因此",
    "为了",
    "通过",
    "自己",
    "一样",
    "同时",
    "以及",
    "主要",
    "其中",
    "关于",
    "这些",
    "那些",
    "不少",
    "很多",
    "非常",
    "还是",
    "个",
    "与",
    "or",
    "the",
    "and",
    "for",
    "you",
    "are",
    "but",
}

POSITIVE_WORDS = {
    "开心",
    "高兴",
    "满意",
    "喜欢",
    "不错",
    "可以",
    "棒",
    "好",
    "稳",
    "靠谱",
    "舒适",
    "期待",
    "欢迎",
    "感谢",
    "祝贺",
    "支持",
    "顺利",
    "加油",
}

NEGATIVE_WORDS = {
    "难",
    "不好",
    "不行",
    "糟",
    "烦",
    "担心",
    "麻烦",
    "郁闷",
    "痛",
    "差",
    "问题",
    "风险",
    "讨厌",
    "哭",
    "崩溃",
    "着急",
    "失败",
    "生气",
    "气",
    "晕",
    "无语",
    "尴尬",
    "恐怖",
}

CQ_PATTERN = re.compile(r"\[CQ:[^\]]+\]")
URL_PATTERN = re.compile(r"https?://\S+")
LETTER_PATTERN = re.compile(r"[A-Za-z]+")
CHINESE_WORD_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def clean_message(text: str) -> str:
    text = CQ_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    return text.replace("\n", " ").strip()


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = clean_message(text)
    if not text:
        return []
    tokens: Iterable[str]
    if jieba:  # pragma: no branch - optional path
        tokens = (tok.strip().lower() for tok in jieba.cut(text))
    else:
        chinese_tokens = CHINESE_WORD_PATTERN.findall(text)
        alpha_tokens = [tok.lower() for tok in LETTER_PATTERN.findall(text)]
        tokens = list(chinese_tokens) + alpha_tokens
    filtered: List[str] = []
    for token in tokens:
        if not token:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) == 1:
            continue
        filtered.append(token)
    return filtered


def build_topic_stats(messages: pd.Series) -> Tuple[List[Tuple[str, int]], float, int]:
    counter: Counter[str] = Counter()
    total_tokens = 0
    for message in messages.dropna():
        tokens = tokenize(message)
        if not tokens:
            continue
        counter.update(tokens)
        total_tokens += len(tokens)
    top_topics = counter.most_common(10)
    index = 0.0
    if total_tokens:
        index = sum(count for _, count in top_topics) / total_tokens * 100
    return top_topics, round(index, 2), total_tokens


def compute_hourly_counts(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    hourly_counts = df.groupby(df["message_time_utc"].dt.hour).size()
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
    labels = [f"{hour:02d}:00" for hour in hourly_counts.index]
    return labels, hourly_counts.tolist()


def compute_weekday_counts(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_counts = df.groupby(df["message_time_utc"].dt.weekday).size()
    weekday_counts = weekday_counts.reindex(range(7), fill_value=0)
    labels = [weekday_map[idx] for idx in weekday_counts.index]
    return labels, weekday_counts.tolist()


def compute_monthly_counts(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    monthly_counts = df.groupby(df["message_time_utc"].dt.to_period("M")).size().sort_index()
    labels = [str(period) for period in monthly_counts.index]
    return labels, monthly_counts.tolist()


def compute_scene_distribution(df: pd.DataFrame) -> Tuple[List[str], List[int], List[float]]:
    counts = df.groupby("session_type").size().sort_values(ascending=False)
    labels = counts.index.tolist()
    values = counts.tolist()
    total = counts.sum() or 1
    shares = [round(value / total * 100, 2) for value in values]
    return labels, values, shares


def compute_depth_index(df: pd.DataFrame) -> dict:
    session_counts = df.groupby("session_qq").size()
    if session_counts.empty:
        return {
            "index": 0,
            "avg_per_session": 0,
            "median_per_session": 0,
            "rich_session_ratio": 0,
            "top_sessions": [],
        }
    avg_msgs = session_counts.mean()
    median_msgs = session_counts.median()
    rich_ratio = float((session_counts >= 5).mean())
    index = (
        0.5 * (avg_msgs / (avg_msgs + 5))
        + 0.3 * (median_msgs / (median_msgs + 5))
        + 0.2 * rich_ratio
    )
    top_sessions = session_counts.sort_values(ascending=False).head(5).reset_index()
    top_records = [
        {"session": row["session_qq"], "messages": int(row[0])}
        for _, row in top_sessions.iterrows()
    ]
    return {
        "index": round(index * 100, 2),
        "avg_per_session": round(float(avg_msgs), 2),
        "median_per_session": round(float(median_msgs), 2),
        "rich_session_ratio": round(rich_ratio * 100, 2),
        "top_sessions": top_records,
    }


def compute_sentiment(messages: pd.Series) -> dict:
    distribution = Counter({"positive": 0, "neutral": 0, "negative": 0})
    scores: List[float] = []
    for raw in messages.dropna():
        tokens = tokenize(raw)
        if not tokens:
            distribution["neutral"] += 1
            scores.append(0.0)
            continue
        pos = sum(1 for token in tokens if token in POSITIVE_WORDS)
        neg = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            distribution["neutral"] += 1
            scores.append(0.0)
            continue
        score = (pos - neg) / total
        scores.append(score)
        if score > 0:
            distribution["positive"] += 1
        elif score < 0:
            distribution["negative"] += 1
        else:
            distribution["neutral"] += 1
    mean_score = sum(scores) / len(scores) if scores else 0
    index = round((mean_score + 1) / 2 * 100, 2)
    total_msgs = max(sum(distribution.values()), 1)
    shares = {k: round(v / total_msgs * 100, 2) for k, v in distribution.items()}
    return {
        "index": index,
        "mean_score": round(mean_score, 3),
        "distribution": distribution,
        "shares": shares,
    }


def build_html(context: dict) -> str:
    pretty_topics = [
        {
            "keyword": keyword,
            "count": count,
            "share": round(count / max(context["topic"]["total_tokens"], 1) * 100, 2),
        }
        for keyword, count in context["topic"]["top_keywords"]
    ]
    context = {**context, "pretty_topics": pretty_topics}
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>聊天核心话题与节奏洞察</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      font-family: "Noto Sans SC", "PingFang SC", "Microsoft Yahei", sans-serif;
      color: #1c1c1c;
      background: #f5f7fb;
    }}
    body {{
      margin: 0;
      padding: 32px;
      line-height: 1.65;
    }}
    h1 {{
      margin-bottom: 0.5rem;
    }}
    .subtitle {{
      color: #667085;
      margin-bottom: 2rem;
    }}
    section {{
      background: #fff;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 1rem;
    }}
    .card {{
      padding: 16px;
      background: #f9fafb;
      border-radius: 12px;
      border: 1px solid #e4e7ec;
    }}
    .card h3 {{
      margin: 0;
      font-size: 0.95rem;
      color: #475467;
    }}
    .card .value {{
      font-size: 1.75rem;
      font-weight: 600;
      color: #0f172a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px;
      border-bottom: 1px solid #f1f5f9;
    }}
    th {{
      background: #f8fafc;
      color: #475467;
      font-size: 0.9rem;
    }}
    canvas {{
      max-width: 100%;
    }}
    .chart-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 24px;
    }}
    .note {{
      font-size: 0.9rem;
      color: #475467;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>聊天核心话题与节奏洞察</h1>
  <p class="subtitle">数据来源：message.csv，样本量 {context["meta"]["message_count"]} 条消息</p>

  <section>
    <h2>话题高频度指数（核心话题）</h2>
    <div class="cards">
      <div class="card">
        <h3>核心话题覆盖率</h3>
        <div class="value">{context["topic"]["index"]}%</div>
        <p class="note">Top10 关键词占全部有效词频的比例</p>
      </div>
      <div class="card">
        <h3>有效分词总量</h3>
        <div class="value">{context["topic"]["total_tokens"]}</div>
        <p class="note">去除停用词与噪声后的计数</p>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>关键词</th>
          <th>出现次数</th>
          <th>占比</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f"<tr><td>{row['keyword']}</td><td>{row['count']}</td><td>{row['share']}%</td></tr>" for row in pretty_topics)}
      </tbody>
    </table>
    <p class="note">{context["topic"]["narrative"]}</p>
  </section>

  <section>
    <h2>聊天时间分布（高峰期 / 周期性）</h2>
    <div class="cards">
      <div class="card">
        <h3>最活跃小时</h3>
        <div class="value">{context["temporal"]["peak_hour"]["label"]}</div>
        <p class="note">消息量 {context["temporal"]["peak_hour"]["count"]} 条</p>
      </div>
      <div class="card">
        <h3>峰值日期</h3>
        <div class="value">{context["temporal"]["peak_day"]["date"]}</div>
        <p class="note">消息量 {context["temporal"]["peak_day"]["count"]} 条</p>
      </div>
      <div class="card">
        <h3>周内节奏</h3>
        <div class="value">{context["temporal"]["weekday_pattern"]}</div>
        <p class="note">周末/工作日对比</p>
      </div>
    </div>
    <div class="chart-row">
      <canvas id="hourly-chart"></canvas>
      <canvas id="weekday-chart"></canvas>
      <canvas id="monthly-chart"></canvas>
    </div>
    <p class="note">{context["temporal"]["narrative"]}</p>
  </section>

  <section>
    <h2>社交场景关联度</h2>
    <div class="cards">
      <div class="card">
        <h3>主要场景</h3>
        <div class="value">{context["scene"]["top_scene"]["label"]}</div>
        <p class="note">占比 {context["scene"]["top_scene"]["share"]}%</p>
      </div>
      <div class="card">
        <h3>多场景覆盖</h3>
        <div class="value">{context["scene"]["scene_count"]} 种</div>
        <p class="note">session_type 字段识别的聊天类型</p>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>场景类型</th>
          <th>消息量</th>
          <th>占比</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f"<tr><td>{label}</td><td>{count}</td><td>{share}%</td></tr>" for label, count, share in context["scene"]["table"])}
      </tbody>
    </table>
    <p class="note">{context["scene"]["narrative"]}</p>
  </section>

  <section>
    <h2>话题互动深度指数</h2>
    <div class="cards">
      <div class="card">
        <h3>深度指数</h3>
        <div class="value">{context["interaction"]["index"]}</div>
        <p class="note">综合均值 / 中位数 / 丰富会话占比</p>
      </div>
      <div class="card">
        <h3>平均每会话消息</h3>
        <div class="value">{context["interaction"]["avg_per_session"]}</div>
        <p class="note">全部会话粒度</p>
      </div>
      <div class="card">
        <h3>中位值</h3>
        <div class="value">{context["interaction"]["median_per_session"]}</div>
        <p class="note">对极端活跃会话去敏感</p>
      </div>
      <div class="card">
        <h3>≥5条会话占比</h3>
        <div class="value">{context["interaction"]["rich_session_ratio"]}%</div>
        <p class="note">多轮互动覆盖情况</p>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>高互动会话</th>
          <th>消息量</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f"<tr><td>{row['session']}</td><td>{row['messages']}</td></tr>" for row in context["interaction"]["top_sessions"])}
      </tbody>
    </table>
    <p class="note">{context["interaction"]["narrative"]}</p>
  </section>

  <section>
    <h2>情感价值导向指数</h2>
    <div class="cards">
      <div class="card">
        <h3>情感指数</h3>
        <div class="value">{context["sentiment"]["index"]}</div>
        <p class="note">-100（负向）~ 100（正向）</p>
      </div>
      <div class="card">
        <h3>平均得分</h3>
        <div class="value">{context["sentiment"]["mean_score"]}</div>
        <p class="note">基于极性词典的归一化评分</p>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>情感极性</th>
          <th>消息量</th>
          <th>占比</th>
        </tr>
      </thead>
      <tbody>
        {"".join(f"<tr><td>{label}</td><td>{context['sentiment']['distribution'][label]}</td><td>{context['sentiment']['shares'][label]}%</td></tr>" for label in ["positive", "neutral", "negative"])}
      </tbody>
    </table>
    <p class="note">{context["sentiment"]["narrative"]}</p>
  </section>

  <script>
    const hourlyLabels = {json.dumps(context["temporal"]["hourly"]["labels"], ensure_ascii=False)};
    const hourlyData = {json.dumps(context["temporal"]["hourly"]["values"], ensure_ascii=False)};
    const weekdayLabels = {json.dumps(context["temporal"]["weekday"]["labels"], ensure_ascii=False)};
    const weekdayData = {json.dumps(context["temporal"]["weekday"]["values"], ensure_ascii=False)};
    const monthlyLabels = {json.dumps(context["temporal"]["monthly"]["labels"], ensure_ascii=False)};
    const monthlyData = {json.dumps(context["temporal"]["monthly"]["values"], ensure_ascii=False)};

    new Chart(document.getElementById("hourly-chart"), {{
      type: "line",
      data: {{
        labels: hourlyLabels,
        datasets: [{{
          label: "小时消息量",
          data: hourlyData,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.1)",
          tension: 0.3,
          fill: true,
        }}]
      }},
      options: {{
        plugins: {{
          legend: {{ display: false }}
        }},
        scales: {{
          y: {{
            beginAtZero: true
          }}
        }}
      }}
    }});

    new Chart(document.getElementById("weekday-chart"), {{
      type: "bar",
      data: {{
        labels: weekdayLabels,
        datasets: [{{
          label: "周内分布",
          data: weekdayData,
          backgroundColor: "#22c55e",
        }}]
      }},
      options: {{
        plugins: {{
          legend: {{ display: false }}
        }},
        scales: {{
          y: {{
            beginAtZero: true
          }}
        }}
      }}
    }});

    new Chart(document.getElementById("monthly-chart"), {{
      type: "bar",
      data: {{
        labels: monthlyLabels,
        datasets: [{{
          label: "月度消息量",
          data: monthlyData,
          backgroundColor: "#f97316",
        }}]
      }},
      options: {{
        plugins: {{
          legend: {{ display: false }}
        }},
        scales: {{
          y: {{
            beginAtZero: true
          }}
        }}
      }}
    }});
  </script>
</body>
</html>"""


def build_context(df: pd.DataFrame) -> dict:
    topic_keywords, topic_index, total_tokens = build_topic_stats(df["message_content"])
    hour_labels, hour_values = compute_hourly_counts(df)
    weekday_labels, weekday_values = compute_weekday_counts(df)
    month_labels, month_values = compute_monthly_counts(df)
    scene_labels, scene_values, scene_shares = compute_scene_distribution(df)

    scene_table = list(zip(scene_labels, scene_values, scene_shares))
    top_scene = (
        {"label": scene_labels[0], "share": scene_shares[0]}
        if scene_labels
        else {"label": "无数据", "share": 0}
    )

    hour_series = pd.Series(hour_values, index=hour_labels)
    peak_hour_label = hour_series.idxmax()
    peak_hour_count = int(hour_series.max()) if not hour_series.empty else 0

    daily_counts = df.groupby(df["message_time_utc"].dt.date).size()
    if not daily_counts.empty:
        peak_day = {"date": daily_counts.idxmax().isoformat(), "count": int(daily_counts.max())}
    else:
        peak_day = {"date": "无", "count": 0}

    weekday_series = pd.Series(weekday_values, index=weekday_labels)
    weekday_pattern = (
        weekday_series.idxmax() if not weekday_series.empty else "无"
    )

    weekday_workday = sum(weekday_values[:5])
    weekday_weekend = sum(weekday_values[5:])
    weekend_ratio = weekday_weekend / weekday_workday if weekday_workday else 0.0
    weekend_percent = round(weekend_ratio * 100, 1)
    temporal_narrative = (
        f"{peak_hour_label} 为全天高峰；{peak_day['date']} 是整体峰值日，周内以 {weekday_pattern} 为重心，"
        f"周末活跃度约为工作日的 {weekend_percent}%。"
    )

    if topic_keywords:
        top_terms = "、".join([kw for kw, _ in topic_keywords[:3]])
        topic_narrative = f"核心词 {top_terms} 占据前列，表现为任务广播与咨询型对话为主。"
    else:
        topic_narrative = "暂无显著关键词。"

    if len(scene_labels) >= 2:
        scene_narrative = (
            f"{scene_labels[0]} 占比 {scene_shares[0]}%，显著高于 {scene_labels[1]} "
            f"({scene_shares[1]}%)，其余场景仅提供补充触达。"
        )
    elif scene_labels:
        scene_narrative = f"{scene_labels[0]} 为唯一场景，占比 {scene_shares[0]}%。"
    else:
        scene_narrative = "暂无场景分布数据。"

    interaction_stats = compute_depth_index(df)
    interaction_narrative = (
        f"平均每会话 {interaction_stats['avg_per_session']} 条、≥5条会话占比 "
        f"{interaction_stats['rich_session_ratio']}%，深度指数 {interaction_stats['index']}。"
    )

    sentiment_data = compute_sentiment(df["message_content"])
    sentiment_narrative = (
        f"情绪以中性陈述为主（占比 {sentiment_data['shares']['neutral']}%），"
        f"正向 {sentiment_data['shares']['positive']}%，负向 {sentiment_data['shares']['negative']}%。"
    )

    return {
        "meta": {
            "message_count": int(len(df)),
        },
        "topic": {
            "index": topic_index,
            "total_tokens": total_tokens,
            "top_keywords": topic_keywords,
            "narrative": topic_narrative,
        },
        "temporal": {
            "peak_hour": {"label": peak_hour_label, "count": peak_hour_count},
            "peak_day": peak_day,
            "weekday_pattern": weekday_pattern,
            "weekend_ratio": round(weekend_ratio, 3),
            "weekend_percent": weekend_percent,
            "hourly": {"labels": hour_labels, "values": hour_values},
            "weekday": {"labels": weekday_labels, "values": weekday_values},
            "monthly": {"labels": month_labels, "values": month_values},
            "narrative": temporal_narrative,
        },
        "scene": {
            "table": scene_table,
            "top_scene": top_scene,
            "scene_count": len(scene_labels),
            "narrative": scene_narrative,
        },
        "interaction": {**interaction_stats, "narrative": interaction_narrative},
        "sentiment": {**sentiment_data, "narrative": sentiment_narrative},
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Cannot find {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["message_time_utc"], keep_default_na=False)
    df["message_time_utc"] = pd.to_datetime(df["message_time_utc"], errors="coerce")
    df = df.dropna(subset=["message_time_utc"])
    context = build_context(df)
    JSON_OUTPUT_PATH.write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html = build_html(context)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
