from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from text_filters import is_noise_message

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "message.csv"
USER_ACTIVITY_CSV = BASE_DIR / "user_activity_summary.csv"
SESSION_HEALTH_CSV = BASE_DIR / "session_health_summary.csv"
OUTPUT_JSON = BASE_DIR / "kmeans_insights.json"

MAX_CONTENT_DOCS = 6000
CONTENT_VOCAB = 400
CONTENT_CLUSTERS = 6

USER_MIN_MESSAGES = 80
USER_CLUSTERS = 4

SESSION_VOCAB = 150
SESSION_CLUSTERS = 3

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
    "一个",
    "这个",
    "那个",
    "啊",
    "吗",
    "吧",
    "哦",
    "啦",
    "呀",
    "么",
    "啥",
    "了么",
    "噢",
}

FORMAT_PATTERNS = {
    "图片": [r"\[图片\]", ".jpg", ".jpeg", ".png", ".gif"],
    "语音": [r"\[语音\]", ".amr", ".silk", ".mp3"],
    "视频": [r"\[视频\]", ".mp4", ".mov", ".avi"],
    "表情包": [r"\[表情", r"\[动画表情", "emoji"],
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if not chunk or chunk in STOPWORDS:
            continue
        if chunk.isdigit():
            continue
        tokens.append(chunk)
    return tokens


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


def tfidf_vectors(
    docs: List[Counter[str]], vocab_size: int
) -> tuple[list[list[float]], list[str], dict[str, float]]:
    df = Counter()
    for counter in docs:
        df.update(counter.keys())

    vocab = [token for token, _ in df.most_common(vocab_size)]
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    idf: dict[str, float] = {}
    total_docs = len(docs)
    for token in vocab:
        idf[token] = math.log((1 + total_docs) / (1 + df[token])) + 1

    vectors: list[list[float]] = []
    for counter in docs:
        total_terms = sum(counter.values()) or 1
        vec = [0.0] * len(vocab)
        for token, count in counter.items():
            idx = vocab_index.get(token)
            if idx is None:
                continue
            tf = count / total_terms
            vec[idx] = tf * idf[token]
        vectors.append(vec)
    return vectors, vocab, idf


def kmeans(
    vectors: list[list[float]], k: int, max_iter: int = 25, seed: int = 42
) -> list[int]:
    random.seed(seed)
    if not vectors:
        return []
    centroids = [vec[:] for vec in random.sample(vectors, min(k, len(vectors)))]
    assignments = [0] * len(vectors)

    def distance(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    for _ in range(max_iter):
        # Assign step
        changed = False
        for i, vec in enumerate(vectors):
            best_idx = 0
            best_dist = float("inf")
            for idx, centroid in enumerate(centroids):
                dist = distance(vec, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if assignments[i] != best_idx:
                assignments[i] = best_idx
                changed = True
        if not changed:
            break

        # Update step
        new_centroids = [[0.0] * len(vectors[0]) for _ in centroids]
        counts = [0] * len(centroids)
        for assignment, vec in zip(assignments, vectors):
            counts[assignment] += 1
            for idx, value in enumerate(vec):
                new_centroids[assignment][idx] += value
        for idx, centroid in enumerate(new_centroids):
            if counts[idx] == 0:
                new_centroids[idx] = vectors[random.randint(0, len(vectors) - 1)][:]  # pragma: no cover - fallback
            else:
                new_centroids[idx] = [
                    value / counts[idx] for value in centroid
                ]
        centroids = new_centroids
    return assignments


def label_content_cluster(top_keywords: list[tuple[str, int]], index: int) -> str:
    keywords = {kw for kw, _ in top_keywords[:5]}
    if {"兼职", "招聘", "实习"} & keywords:
        return "兼职招聘类"
    if {"考试", "作业", "课程", "考研"} & keywords:
        return "学习考试类"
    if {"吐槽", "无语", "坑", "垃圾"} & keywords:
        return "吐槽闲聊类"
    if {"通知", "官宣", "活动", "报名"} & keywords:
        return "活动资讯类"
    if {"表白", "约饭", "朋友"} & keywords:
        return "社交互动类"
    return f"内容簇{index + 1}"


def label_user_cluster(metrics: dict[str, float]) -> str:
    if metrics["night_share"] >= 0.35:
        return "深夜闲聊型"
    if metrics["messages_per_day"] >= 150 and metrics["group_share"] > 0.8:
        return "活动组织型"
    if metrics["media_share"] >= 0.4:
        return "资讯搬运型"
    if metrics["messages_per_day"] <= 40 and metrics["group_share"] < 0.5:
        return "轻量私聊型"
    return "日常互动型"


def label_session_cluster(avg_metrics: dict[str, float]) -> str:
    if avg_metrics["messages"] >= 10000 and avg_metrics["top_user_share"] >= 0.3:
        return "超高活跃但集中"
    if avg_metrics["messages"] >= 3000 and avg_metrics["gini"] <= 0.65:
        return "平稳交流"
    return "待唤醒/长尾"


def normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    min_v = min(values)
    max_v = max(values)
    if math.isclose(min_v, max_v):
        return [0.5 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    random.seed(42)
    content_docs: list[Counter[str]] = []
    content_texts: list[str] = []
    content_seen = 0

    user_extra = defaultdict(
        lambda: {
            "total": 0,
            "night": 0,
            "media": 0,
            "length": 0,
            "hour_sum": 0,
        }
    )
    session_tokens: dict[str, Counter[str]] = defaultdict(Counter)

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row.get("message_content") or "").strip()
            if not text:
                continue
            if is_noise_message(text):
                continue
            tokens = tokenize(text)
            if tokens:
                if len(content_docs) < MAX_CONTENT_DOCS:
                    content_docs.append(Counter(tokens))
                    content_texts.append(text[:160])
                else:
                    content_seen += 1
                    idx = random.randint(0, content_seen)
                    if idx < MAX_CONTENT_DOCS:
                        content_docs[idx] = Counter(tokens)
                        content_texts[idx] = text[:160]
                content_seen += 1

            session_id = row["session_qq"]
            session_tokens[session_id].update(tokens)

            user = row["user_qq"]
            stats = user_extra[user]
            stats["total"] += 1
            stats["length"] += len(text)
            fmt = detect_format(text)
            if fmt != "文字":
                stats["media"] += 1
            ts = row.get("message_time_utc")
            if ts:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    hour = dt.hour
                    stats["hour_sum"] += hour
                    if 0 <= hour < 5:
                        stats["night"] += 1
                except ValueError:
                    pass

    vectors, vocab, _ = tfidf_vectors(content_docs, CONTENT_VOCAB)
    content_assignments = kmeans(vectors, CONTENT_CLUSTERS)

    content_clusters: list[list[int]] = [[] for _ in range(max(content_assignments, default=-1) + 1)]
    for idx, assignment in enumerate(content_assignments):
        if assignment >= len(content_clusters):
            content_clusters.extend([] for _ in range(assignment - len(content_clusters) + 1))
        content_clusters[assignment].append(idx)

    content_results = []
    for cluster_idx, indices in enumerate(content_clusters):
        if not indices:
            continue
        keyword_counter = Counter()
        samples: list[str] = []
        for idx in indices:
            keyword_counter.update(content_docs[idx])
            if len(samples) < 3:
                samples.append(content_texts[idx])
        top_keywords = keyword_counter.most_common(8)
        label = label_content_cluster(top_keywords, cluster_idx)
        content_results.append(
            {
                "label": label,
                "size": len(indices),
                "topKeywords": top_keywords,
                "sampleMessages": samples,
            }
        )

    # User clustering
    user_summary = {}
    if USER_ACTIVITY_CSV.exists():
        with USER_ACTIVITY_CSV.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                user_summary[row["user_qq"]] = row

    user_vectors: list[list[float]] = []
    user_metrics: list[dict[str, float]] = []
    users = []
    feature_lists: dict[str, list[float]] = defaultdict(list)

    for user, row in user_summary.items():
        total = int(row["total_messages"])
        if total < USER_MIN_MESSAGES:
            continue
        extra = user_extra.get(user)
        if not extra or extra["total"] == 0:
            continue

        group = int(row["group_messages"])
        private = int(row["private_messages"])
        group_share = group / total if total else 0
        media_share = extra["media"] / total
        night_share = extra["night"] / total
        avg_len = extra["length"] / total
        messages_per_day = float(row["messages_per_span_day"])
        active_days = int(row["active_days"])
        unique_sessions = int(row["unique_sessions"])

        metrics = {
            "log_total": math.log(total + 1),
            "group_share": group_share,
            "messages_per_day": messages_per_day,
            "media_share": media_share,
            "night_share": night_share,
            "avg_len": avg_len,
            "active_days": active_days,
            "unique_sessions": unique_sessions,
        }
        for key, value in metrics.items():
            feature_lists[key].append(value)
        user_metrics.append(metrics)
        users.append(user)

    normalized_vectors: list[list[float]] = []
    normalized_features: dict[str, list[float]] = {}
    for key, values in feature_lists.items():
        normalized_features[key] = normalize(values)

    for idx, metrics in enumerate(user_metrics):
        vec = [normalized_features[key][idx] for key in normalized_features]
        normalized_vectors.append(vec)

    user_assignments = kmeans(normalized_vectors, USER_CLUSTERS)

    user_cluster_data: list[dict] = []
    cluster_map: dict[int, list[int]] = defaultdict(list)
    for idx, assignment in enumerate(user_assignments):
        cluster_map[assignment].append(idx)

    for assignment, member_indices in sorted(cluster_map.items()):
        if not member_indices:
            continue
        agg = defaultdict(float)
        for idx in member_indices:
            for key, value in user_metrics[idx].items():
                agg[key] += value
        size = len(member_indices)
        avg_metrics = {key: value / size for key, value in agg.items()}
        label = label_user_cluster(
            {
                "night_share": avg_metrics["night_share"],
                "messages_per_day": avg_metrics["messages_per_day"],
                "group_share": avg_metrics["group_share"],
                "media_share": avg_metrics["media_share"],
            }
        )
        top_examples = [users[idx] for idx in sorted(member_indices, key=lambda i: user_metrics[i]["log_total"], reverse=True)[:5]]
        user_cluster_data.append(
            {
                "label": label,
                "size": size,
                "avgMetrics": avg_metrics,
                "exampleUsers": top_examples,
            }
        )

    # Session clustering
    session_rows = []
    if SESSION_HEALTH_CSV.exists():
        with SESSION_HEALTH_CSV.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                session_rows.append(row)

    session_rows = sorted(session_rows, key=lambda row: int(row["messages"]), reverse=True)[:800]
    session_features = []
    session_ids = []
    numeric_lists: dict[str, list[float]] = defaultdict(list)
    for row in session_rows:
        session_id = row["session_qq"]
        messages = int(row["messages"])
        unique_users = int(row["unique_users"])
        gini = float(row["gini_participation"])
        top_share = float(row["top_user_share"])
        per_day = float(row["messages_per_active_day"])
        quiet_days = int(row["quiet_days"])
        session_ids.append(session_id)
        metrics = {
            "messages": messages,
            "unique_users": unique_users,
            "gini": gini,
            "top_share": top_share,
            "per_day": per_day,
            "quiet_days": quiet_days,
        }
        for key, value in metrics.items():
            numeric_lists[key].append(value)
        session_features.append(metrics)

    normalized_numeric: dict[str, list[float]] = {
        key: normalize(values) for key, values in numeric_lists.items()
    }

    session_doc_list = [session_tokens.get(session_id, Counter()) for session_id in session_ids]
    session_vectors_kw, vocab_session, _ = tfidf_vectors(session_doc_list, SESSION_VOCAB)

    session_vectors: list[list[float]] = []
    for idx, _ in enumerate(session_ids):
        numeric_vec = [
            normalized_numeric["messages"][idx],
            normalized_numeric["unique_users"][idx],
            normalized_numeric["gini"][idx],
            normalized_numeric["top_share"][idx],
            normalized_numeric["per_day"][idx],
            normalized_numeric["quiet_days"][idx],
        ]
        session_vectors.append(numeric_vec + session_vectors_kw[idx])

    session_assignments = kmeans(session_vectors, SESSION_CLUSTERS)
    session_cluster_map: dict[int, list[int]] = defaultdict(list)
    for idx, assignment in enumerate(session_assignments):
        session_cluster_map[assignment].append(idx)

    session_results = []
    for assignment, member_indices in sorted(session_cluster_map.items()):
        if not member_indices:
            continue
        agg = defaultdict(float)
        keyword_counter = Counter()
        for idx in member_indices:
            metrics = session_features[idx]
            for key, value in metrics.items():
                agg[key] += value
            keyword_counter.update(session_doc_list[idx])
        size = len(member_indices)
        avg_metrics = {key: value / size for key, value in agg.items()}
        label = label_session_cluster(
            {
                "messages": avg_metrics["messages"],
                "top_user_share": avg_metrics["top_share"],
                "gini": avg_metrics["gini"],
            }
        )
        top_sessions = [session_ids[i] for i in member_indices[:5]]
        session_results.append(
            {
                "label": label,
                "size": size,
                "avgMetrics": avg_metrics,
                "topKeywords": keyword_counter.most_common(8),
                "exampleSessions": top_sessions,
            }
        )

    insights = {
        "contentClusters": content_results,
        "userClusters": user_cluster_data,
        "sessionClusters": session_results,
    }
    OUTPUT_JSON.write_text(json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"KMeans insights written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
