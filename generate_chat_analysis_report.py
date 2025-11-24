import base64
import io
import json
import re
from collections import Counter, defaultdict
from datetime import timedelta

import jieba
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# Basic Chinese stopwords and punctuation.
STOPWORDS = {
    "", " ", "\n", "\t", "的", "了", "呢", "吗", "哦", "啊", "吧", "你", "我", "他", "她",
    "他们", "我们", "和", "或", "在", "是", "都", "就", "还", "很", "也", "有", "没",
    "没有", "一个", "这个", "那个", "如果", "但是", "因为", "所以", "而且", "以及",
    "可能", "可以", "还是", "然后", "已经", "觉得", "感觉", "需要", "请", "谢谢",
    "不是", "什么", "怎么", "这样", "知道", "吧", "吗", "哈", "呀",
}
PUNCS = r"[\\s\\d\\W_]+"


def tokenize(text: str):
    text = re.sub(PUNCS, " ", str(text)).strip()
    tokens = [t for t in jieba.lcut(text) if t and t not in STOPWORDS and len(t) > 1]
    return tokens


def top_terms_from_matrix(vectorizer, matrix, top_n=10):
    terms = np.array(vectorizer.get_feature_names_out())
    weights = np.asarray(matrix).ravel()
    top_idx = weights.argsort()[::-1][:top_n]
    return [(terms[i], float(weights[i])) for i in top_idx]


def build_topic_models(texts, n_topics=6, max_features=5000):
    sample_size = min(40000, len(texts))
    sample_texts = texts.sample(sample_size, random_state=42) if len(texts) > sample_size else texts
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=max_features)
    tfidf = vectorizer.fit_transform(sample_texts)

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    components = svd.fit_transform(tfidf)
    topic_terms = []
    for i, comp in enumerate(svd.components_):
        topic_terms.append({
            "topic": f"主题 {i + 1}",
            "terms": top_terms_from_matrix(vectorizer, comp, top_n=10),
        })

    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(tfidf)
    label_counts = Counter(labels)
    clusters = [
        {"topic": f"簇 {i + 1}", "size": int(label_counts.get(i, 0))}
        for i in range(n_topics)
    ]

    keyword_weights = np.asarray(tfidf.mean(axis=0)).ravel()
    top_keywords = top_terms_from_matrix(vectorizer, keyword_weights, top_n=15)

    return {
        "topic_terms": topic_terms,
        "clusters": clusters,
        "top_keywords": top_keywords,
    }


def classify_messages(texts):
    rules = {
        "求助": ["求助", "怎么", "如何", "请教", "有没有", "帮忙", "问题", "bug", "报错"],
        "技术交流": ["代码", "python", "模型", "算法", "数据", "训练", "接口", "部署", "测试"],
        "闲聊": ["哈哈", "表情", "闲聊", "聊天", "早", "晚安", "吹水", "水群"],
        "广告": ["广告", "优惠", "推广", "红包", "扫码", "链接", "报名", "活动"],
        "争吵": ["垃圾", "气死", "举报", "无语", "吵", "骂", "闭嘴", "滚"],
        "通知": ["通知", "会议", "安排", "发布", "更新", "版本", "上线", "公告"],
    }
    counts = Counter()
    labels = []
    for text in texts:
        t = str(text)
        assigned = None
        for label, kws in rules.items():
            if any(k in t for k in kws):
                assigned = label
                break
        labels.append(assigned or "未分类")
        counts[assigned or "未分类"] += 1
    return counts, labels


def sentiment_flags(texts):
    pos_words = {"赞", "优秀", "棒", "牛", "开心", "爽", "喜欢", "感谢", "支持", "稳"}
    neg_words = {"差", "烦", "讨厌", "气", "生气", "崩", "失败", "垃圾", "麻烦", "痛苦", "哭"}
    dist = Counter()
    flags = []
    for text in texts:
        toks = tokenize(text)
        pos = sum(1 for t in toks if t in pos_words)
        neg = sum(1 for t in toks if t in neg_words)
        if pos > neg:
            label = "正向"
        elif neg > pos:
            label = "负向"
        else:
            label = "中性"
        dist[label] += 1
        flags.append(label)
    return dist, flags


def toxicity_flags(texts):
    toxic_words = {"垃圾", "蠢", "傻", "闭嘴", "滚", "骗子", "诈骗", "违规", "广告"}
    dist = Counter()
    flags = []
    for text in texts:
        toks = tokenize(text)
        has_toxic = any(t in toxic_words for t in toks)
        label = "可疑" if has_toxic else "正常"
        dist[label] += 1
        flags.append(label)
    return dist, flags


def intent_flags(texts):
    intents = []
    dist = Counter()
    for text in texts:
        t = str(text)
        if t.endswith("?") or "?" in t or "吗" in t or "？" in t:
            label = "问句/求助"
        elif any(k in t for k in ["通知", "安排", "发布", "上线", "会议"]):
            label = "通知/命令"
        elif any(k in t for k in ["谢谢", "感谢", "收到"]):
            label = "回应"
        else:
            label = "闲聊/其他"
        intents.append(label)
        dist[label] += 1
    return dist, intents


def build_threads(df):
    df = df.sort_values(["session_qq", "message_time"])
    last_time = {}
    thread_id = 0
    thread_lengths = []
    current_len = {}
    thread_id_map = []
    for row in df.itertuples():
        prev_time = last_time.get(row.session_qq)
        new_thread = prev_time is None or (row.message_time - prev_time) > timedelta(minutes=10)
        if new_thread:
            if row.session_qq in current_len:
                thread_lengths.append(current_len[row.session_qq])
            thread_id += 1
            current_len[row.session_qq] = 0
        current_len[row.session_qq] += 1
        last_time[row.session_qq] = row.message_time
        thread_id_map.append(thread_id)
    thread_lengths.extend(current_len.values())
    df = df.copy()
    df["thread_id"] = thread_id_map
    thread_stats = {
        "threads": int(df["thread_id"].nunique()),
        "avg_length": float(np.mean(thread_lengths)),
        "median_length": float(np.median(thread_lengths)),
        "longest": int(max(thread_lengths)),
    }
    return df, thread_stats


def plot_bar(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 3))
    items = list(data.items())
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    ax.bar(labels, values, color="#3d7edb")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_line(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 3))
    series.plot(ax=ax, color="#43a047")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_table(rows, headers):
    html = ["<table class='table'>"]
    html.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
    for row in rows:
        html.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def main():
    df = pd.read_csv("message.csv")
    df["message_time"] = pd.to_datetime(df["message_time_utc"])
    df = df.sort_values("message_time")
    df["text"] = df["message_content"].fillna("").astype(str)

    # 过滤掉包含 QQ CQ 标记的非文本消息
    df = df[~df["text"].str.contains("cq", case=False, na=False)]

    basic_stats = {
        "messages": len(df),
        "users": df["user_name"].nunique(),
        "sessions": df["session_qq"].nunique(),
        "span": f"{df['message_time'].min()} ~ {df['message_time'].max()}",
    }

    topic_data = build_topic_models(df["text"])

    classification_counts, classifications = classify_messages(df["text"])
    sentiment_counts, sentiments = sentiment_flags(df["text"])
    toxicity_counts, toxicity = toxicity_flags(df["text"])
    intent_counts, intents = intent_flags(df["text"])

    df["classification"] = classifications
    df["sentiment"] = sentiments
    df["toxicity"] = toxicity
    df["intent"] = intents

    df["hour"] = df["message_time"].dt.hour
    df["date"] = df["message_time"].dt.date
    df["weekday"] = df["message_time"].dt.day_name()
    df["month"] = df["message_time"].dt.to_period("M").astype(str)

    top_users = df["user_name"].value_counts().head(10)
    hour_counts = df["hour"].value_counts().sort_index()
    date_counts = df["date"].value_counts().sort_index()
    weekday_counts = df["weekday"].value_counts()
    month_counts = df["month"].value_counts().sort_index()

    peak_hour = int(hour_counts.idxmax()) if len(hour_counts) else None
    peak_weekday = str(weekday_counts.idxmax()) if len(weekday_counts) else None
    peak_month = str(month_counts.idxmax()) if len(month_counts) else None

    # Simple forecast for next 7 days using linear fit.
    day_index = np.arange(len(date_counts))
    coeffs = np.polyfit(day_index, date_counts.values, 1)
    slope, intercept = coeffs
    next_days = [int(max(intercept + slope * (len(day_index) + i), 0)) for i in range(1, 8)]
    forecast = {
        "slope": float(slope),
        "predicted_next_7_days_total": int(sum(next_days)),
        "daily_forecast": next_days,
    }

    # Social network edges via sequential replies inside each session.
    edges = defaultdict(int)
    starter_counts = Counter()
    reply_counts = Counter()
    prev_user = {}
    for row in df.itertuples():
        last_user = prev_user.get(row.session_qq)
        if last_user and last_user != row.user_name:
            edges[(last_user, row.user_name)] += 1
            reply_counts[row.user_name] += 1
        else:
            starter_counts[row.user_name] += 1
        prev_user[row.session_qq] = row.user_name
    top_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)[:10]
    initiators = starter_counts.most_common(10)
    responders = reply_counts.most_common(10)
    social_correlation_index = round(sum(edges.values()) / max(len(df), 1), 4)

    # Graph metrics: PageRank & 社区发现
    G = nx.DiGraph()
    for (src, tgt), w in edges.items():
        G.add_edge(src, tgt, weight=w)
    pagerank = nx.pagerank(G, weight="weight") if len(G) > 0 else {}
    pagerank_top = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    communities = []
    if len(G) > 0:
        undirected = G.to_undirected()
        comms = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
        for i, c in enumerate(comms[:6]):
            comm_users = list(c)
            sub_edges = sum(w for (a, b), w in edges.items() if a in c and b in c)
            communities.append((f"社区{i+1}", len(comm_users), sub_edges, ", ".join(comm_users[:8])))

    # Top keywords增长趋势（对前5关键词做线性回归斜率）
    growth_rows = []
    for kw, _score in topic_data["top_keywords"][:5]:
        series = df[df["text"].str.contains(kw, na=False)]["date"].value_counts().sort_index()
        if len(series) < 3:
            continue
        idx = np.arange(len(series))
        slope, intercept = np.polyfit(idx, series.values, 1)
        growth_rows.append((kw, float(slope), int(series.iloc[-1])))
    growth_rows = sorted(growth_rows, key=lambda x: x[1], reverse=True)

    df_with_threads, thread_stats = build_threads(df)
    topic_depth_index = thread_stats.get("avg_length")

    # Cross-group keyword tendencies.
    group_keywords = []
    for name, group in df.groupby("session_name"):
        texts = group["text"]
        vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=1000)
        try:
            tfidf = vectorizer.fit_transform(texts)
            weights = np.asarray(tfidf.mean(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()
            top_idx = weights.argsort()[::-1][:5]
            top_terms = ", ".join(terms[i] for i in top_idx)
        except ValueError:
            top_terms = ""
        group_keywords.append((name, top_terms, len(group)))
    group_keywords = sorted(group_keywords, key=lambda x: x[2], reverse=True)[:10]

    # Topic trend: frequency of top keyword over time.
    trend_keyword = topic_data["top_keywords"][0][0] if topic_data["top_keywords"] else None
    trend_series = None
    if trend_keyword:
        trend_series = df[df["text"].str.contains(trend_keyword, na=False)]["date"].value_counts().sort_index()

    # Plots
    charts = {
        "top_users": plot_bar(top_users.to_dict(), "最活跃用户 Top10", "用户", "消息数"),
        "hours": plot_bar(hour_counts.to_dict(), "按小时活跃度", "小时", "消息数"),
        "weekday": plot_bar(weekday_counts.to_dict(), "按星期活跃度", "星期", "消息数"),
        "daily": plot_line(date_counts, "每日消息量", "日期", "消息数"),
    }
    if trend_series is not None and len(trend_series) > 0:
        charts["trend"] = plot_line(trend_series, f"关键词「{trend_keyword}」趋势", "日期", "频次")

    # HTML rendering
    html_parts = []
    html_parts.append(
        "<!doctype html><html><head><meta charset='utf-8'><title>群聊分析报告</title>"
        "<style>body{font-family:Arial,Helvetica,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px;}"
        "h1,h2{color:#f1f5f9;} .card{background:#111827;border:1px solid #1f2937;padding:16px;margin-bottom:18px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.3);} "
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;} "
        ".table{width:100%;border-collapse:collapse;font-size:13px;} .table th,.table td{border-bottom:1px solid #1f2937;padding:6px;text-align:left;} "
        ".pill{display:inline-block;background:#1d4ed8;color:#e0f2fe;padding:4px 10px;border-radius:20px;margin-right:6px;font-size:12px;} "
        "img{max-width:100%;height:auto;border:1px solid #1f2937;border-radius:8px;} "
        "</style></head><body>"
    )
    html_parts.append("<h1>群聊分析总览</h1>")
    html_parts.append("<div class='grid'>")
    html_parts.append(
        f"<div class='card'><h2>基础数据</h2>"
        f"<p>消息总量：<strong>{basic_stats['messages']}</strong></p>"
        f"<p>活跃用户数：<strong>{basic_stats['users']}</strong></p>"
        f"<p>群/会话数：<strong>{basic_stats['sessions']}</strong></p>"
        f"<p>时间范围：<strong>{basic_stats['span']}</strong></p></div>"
    )
    html_parts.append(
        "<div class='card'><h2>分类占比</h2>"
        + "".join(f"<span class='pill'>{k}: {v}</span>" for k, v in classification_counts.items())
        + "<h2>情绪占比</h2>"
        + "".join(f"<span class='pill'>{k}: {v}</span>" for k, v in sentiment_counts.items())
        + "<h2>毒性检测</h2>"
        + "".join(f"<span class='pill'>{k}: {v}</span>" for k, v in toxicity_counts.items())
        + "</div>"
    )
    html_parts.append(
        "<div class='card'><h2>意图识别</h2>"
        + "".join(f"<span class='pill'>{k}: {v}</span>" for k, v in intent_counts.items())
        + "<h2>线程化统计</h2>"
        + "".join(f"<span class='pill'>{k}: {v}</span>" for k, v in thread_stats.items())
        + "</div>"
    )
    html_parts.append("</div>")  # end grid

    html_parts.append("<div class='card'><h2>主题提取与话题簇</h2>")
    rows = []
    for t in topic_data["topic_terms"]:
        term_str = ", ".join([w for w, _ in t["terms"]])
        rows.append((t["topic"], term_str))
    html_parts.append(render_table(rows, ["主题", "代表词"]))
    html_parts.append(render_table([(c["topic"], c["size"]) for c in topic_data["clusters"]], ["簇", "消息数"]))
    html_parts.append("</div>")

    html_parts.append("<div class='card'><h2>关键词提取 (TF-IDF)</h2>")
    kw_rows = [(w, f"{score:.4f}") for w, score in topic_data["top_keywords"]]
    html_parts.append(render_table(kw_rows, ["关键词", "权重"]))
    html_parts.append("</div>")

    html_parts.append("<div class='card'><h2>活跃度分析</h2><div class='grid'>")
    for key, title in [("top_users", "活跃用户"), ("hours", "小时分布"), ("weekday", "星期分布"), ("daily", "每日趋势")]:
        html_parts.append(f"<div><img alt='{title}' src='data:image/png;base64,{charts[key]}'></div>")
    if "trend" in charts:
        html_parts.append(f"<div><img alt='关键词趋势' src='data:image/png;base64,{charts['trend']}'></div>")
    html_parts.append("</div></div>")

    html_parts.append("<div class='card'><h2>社交网络与角色</h2>")
    edge_rows = [(f"{a} → {b}", w) for (a, b), w in top_edges]
    html_parts.append(render_table(edge_rows, ["用户对", "交互次数"]))
    html_parts.append("<h3>话题发起者 Top10</h3>")
    html_parts.append(render_table(initiators, ["用户", "开场次数"]))
    html_parts.append("<h3>回应者 Top10</h3>")
    html_parts.append(render_table(responders, ["用户", "回应次数"]))
    html_parts.append("</div>")

    html_parts.append("<div class='card'><h2>跨群关键词画像</h2>")
    html_parts.append(render_table(group_keywords, ["群/会话", "高频关键词", "消息数"]))
    html_parts.append("</div>")

    html_parts.append("<div class='card'><h2>社交网络 - PageRank & 社区</h2>")
    if pagerank_top:
        html_parts.append(render_table([(u, f"{s:.4f}") for u, s in pagerank_top], ["用户", "PageRank"]))
    if communities:
        html_parts.append(render_table(communities, ["社区", "成员数", "内部互动", "示例成员"]))
    html_parts.append("</div>")

    if growth_rows:
        html_parts.append("<div class='card'><h2>关键词趋势预测</h2>")
        html_parts.append(render_table(growth_rows, ["关键词", "增长斜率", "最近一天频次"]))
        html_parts.append("</div>")

    html_parts.append("<div class='card'><h2>预测</h2>")
    html_parts.append(
        f"<p>活跃度线性趋势斜率（每日变化）：<strong>{forecast['slope']:.2f}</strong></p>"
        f"<p>预计未来7天总消息量：<strong>{forecast['predicted_next_7_days_total']}</strong></p>"
        f"<p>逐日预测：{forecast['daily_forecast']}</p>"
    )
    html_parts.append("</div>")

    html_parts.append("</body></html>")
    with open("analysis_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    summary = {
        "basic_stats": basic_stats,
        "classification_counts": classification_counts,
        "sentiment_counts": sentiment_counts,
        "toxicity_counts": toxicity_counts,
        "intent_counts": intent_counts,
        "thread_stats": thread_stats,
        "forecast": forecast,
        "time_peaks": {
            "hour": peak_hour,
            "weekday": peak_weekday,
            "month": peak_month,
        },
        "topic_frequency_index": float(topic_data["top_keywords"][0][1]) if topic_data["top_keywords"] else None,
        "social_correlation_index": social_correlation_index,
        "topic_depth_index": topic_depth_index,
        "emotional_orientation_index": round(
            (sentiment_counts.get("正向", 0) - sentiment_counts.get("负向", 0))
            / max(
                sentiment_counts.get("正向", 0)
                + sentiment_counts.get("负向", 0)
                + sentiment_counts.get("中性", 0),
                1,
            ),
            4,
        ),
    }
    with open("analysis_report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
