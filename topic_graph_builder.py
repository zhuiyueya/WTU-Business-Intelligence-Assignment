from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

INPUT_CSV = Path("message.csv")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
SLANG_XLSX = Path("网络热梗.xlsx")

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
MAX_GRAPH_NODES = 40
CO_OCCURRENCE_BUCKET_MINUTES = 5

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
    "这个",
    "那个",
    "就是",
    "一下",
    "一下子",
    "那个",
    "这个",
    "的话",
    "其实",
    "以及",
    "还有",
    "the",
    "and",
    "for",
    "with",
    "from",
    "you",
    "are",
    "that",
    "this",
}

try:  # Optional dependency for better词性识别
    from jieba import posseg as pseg  # type: ignore

    _JIEBA_AVAILABLE = True
except Exception:  # pragma: no cover - jieba 可能未安装
    pseg = None
    _JIEBA_AVAILABLE = False

_SLANG_CACHE: list[str] | None = None
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass
class Message:
    session_id: str
    session_name: str
    user: str
    user_name: str
    timestamp: datetime
    content: str
    content_lower: str


def load_group_messages(path: Path = INPUT_CSV) -> list[Message]:
    messages: list[Message] = []
    if not path.exists():
        return messages

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("session_type") or "").lower() != "group":
                continue
            ts_str = row.get("message_time_utc")
            if not ts_str:
                continue
            try:
                ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
            except ValueError:
                continue

            content = (row.get("message_content") or "").strip()
            if not content or _is_special_cq(content):
                continue
            lower = content.lower()
            session_id = row.get("session_qq") or ""
            session_name = row.get("session_name") or session_id
            user = row.get("user_qq") or ""
            user_name = row.get("user_name") or user

            messages.append(
                Message(
                    session_id=session_id,
                    session_name=session_name,
                    user=user,
                    user_name=user_name,
                    timestamp=ts,
                    content=content,
                    content_lower=lower,
                )
            )

    messages.sort(key=lambda m: (m.session_id, m.timestamp))
    return messages


def select_keywords(
    messages: Iterable[Message],
    *,
    top_n: int,
    min_count: int = 3,
    slang_tokens: list[str] | None = None,
) -> list[str]:
    slang_tokens = slang_tokens or []
    slang_lookup = {token.lower(): token for token in slang_tokens if token}
    slang_counter: Counter[str] = Counter()
    noun_counter: Counter[str] = Counter()

    for message in messages:
        chunks = list(_iter_basic_chunks(message.content))
        slang_hits = {
            slang_lookup[ch.lower()]
            for ch in chunks
            if ch.lower() in slang_lookup
        }
        for hit in slang_hits:
            slang_counter[hit] += 1

        noun_tokens = _extract_noun_tokens(message.content, slang_lookup)
        noun_counter.update(noun_tokens)

    candidates: list[tuple[str, bool, float, int]] = []
    for token, count in slang_counter.items():
        if count < min_count:
            continue
        candidates.append((token, True, count * 2.0, count))
    for token, count in noun_counter.items():
        if count < min_count:
            continue
        candidates.append((token, False, float(count), count))

    candidates.sort(key=lambda item: (item[2], item[3]), reverse=True)

    keywords: list[str] = []
    matched_slang = False
    for token, is_slang, _, _ in candidates:
        if token in keywords:
            continue
        keywords.append(token)
        matched_slang = matched_slang or is_slang
        if len(keywords) >= top_n:
            break

    if matched_slang:
        _log_slang_hits(slang_counter)
    return keywords


def build_topic_graphs(
    *,
    top_n: int = 3,
    window_minutes: int = 15,
    min_messages: int = 5,
    min_users: int = 3,
) -> list[dict]:
    messages = load_group_messages()
    if not messages:
        return []

    slang_tokens = load_slang_tokens()
    keywords = select_keywords(messages, top_n=top_n * 2, slang_tokens=slang_tokens)
    if not keywords:
        return []

    per_session: defaultdict[str, list[Message]] = defaultdict(list)
    for msg in messages:
        per_session[msg.session_id].append(msg)

    window_span = timedelta(minutes=window_minutes)
    all_windows: list[dict] = []

    for token in keywords:
        token_lower = token.lower()
        for session_id, items in per_session.items():
            hits = [m for m in items if token_lower in m.content_lower]
            if not hits:
                continue

            intervals: list[tuple[datetime, datetime]] = []
            for hit in hits:
                start = hit.timestamp - window_span
                end = hit.timestamp + window_span
                if intervals and start <= intervals[-1][1]:
                    prev_start, prev_end = intervals[-1]
                    intervals[-1] = (prev_start, max(prev_end, end))
                else:
                    intervals.append((start, end))

            for start, end in intervals:
                window_msgs = [m for m in items if start <= m.timestamp <= end]
                if len(window_msgs) < min_messages:
                    continue
                participants = {m.user for m in window_msgs}
                if len(participants) < min_users:
                    continue

                graph = _build_graph(window_msgs, token_lower)
                all_windows.append(
                    {
                        "id": f"{token}_{session_id}_{start:%Y%m%d%H%M}",
                        "keyword": token,
                        "sessionId": session_id,
                        "sessionName": window_msgs[0].session_name,
                        "start": start.isoformat(sep=" "),
                        "end": end.isoformat(sep=" "),
                        "messages": len(window_msgs),
                        "uniqueUsers": len(participants),
                        **graph,
                    }
                )

    all_windows.sort(key=lambda w: (w["messages"], w["uniqueUsers"]), reverse=True)
    selected = all_windows[:top_n]
    for window in selected:
        window["insight"] = _summarize_window(window)
    return selected


def _build_graph(window_msgs: list[Message], token_lower: str) -> dict:
    nodes: dict[str, dict] = {}
    edges: defaultdict[tuple[str, str], int] = defaultdict(int)
    bucket_participants: defaultdict[datetime, set[str]] = defaultdict(set)
    prev_user: str | None = None

    for msg in window_msgs:
        stats = nodes.setdefault(
            msg.user,
            {
                "id": msg.user,
                "label": msg.user_name,
                "messages": 0,
                "isTrigger": False,
            },
        )
        stats["messages"] += 1
        if token_lower and token_lower in msg.content_lower:
            stats["isTrigger"] = True

        if prev_user and prev_user != msg.user:
            key = tuple(sorted((prev_user, msg.user)))
            edges[key] += 1
        prev_user = msg.user

        bucket_key = _bucket_timestamp(msg.timestamp)
        bucket_participants[bucket_key].add(msg.user)

    for participants in bucket_participants.values():
        if len(participants) < 2:
            continue
        for src, tgt in combinations(sorted(participants), 2):
            edges[(src, tgt)] += 1

    node_list = list(nodes.values())
    edge_list = [
        {"source": src, "target": tgt, "weight": weight}
        for (src, tgt), weight in edges.items()
    ]

    if len(node_list) > MAX_GRAPH_NODES:
        top_ids = {
            node["id"]
            for node in sorted(node_list, key=lambda n: n["messages"], reverse=True)[
                :MAX_GRAPH_NODES
            ]
        }
        node_list = [node for node in node_list if node["id"] in top_ids]
        edge_list = [
            edge
            for edge in edge_list
            if edge["source"] in top_ids and edge["target"] in top_ids
        ]

    label_map = {node["id"]: node["label"] for node in node_list}
    top_connections = [
        {
            "source": label_map.get(edge["source"], edge["source"]),
            "target": label_map.get(edge["target"], edge["target"]),
            "weight": edge["weight"],
        }
        for edge in sorted(edge_list, key=lambda e: e["weight"], reverse=True)[:5]
    ]

    return {"nodes": node_list, "edges": edge_list, "topConnections": top_connections}


def _summarize_window(window: dict) -> str:
    if not window.get("nodes"):
        return ""
    top_node = max(window["nodes"], key=lambda n: n.get("messages", 0))
    start = window.get("start", "")
    end = window.get("end", "")
    return (
        f"关键词『{window['keyword']}』在 {window['sessionName']} 中触发 {window['messages']} 条消息，"
        f"共有 {window['uniqueUsers']} 人参与，{top_node['label']} 贡献 {top_node['messages']} 条。"
        f"窗口：{start} ~ {end}."
    )


def _iter_basic_chunks(text: str) -> Iterable[str]:
    for chunk in TOKEN_PATTERN.findall(text):
        cleaned = chunk.strip()
        if cleaned:
            yield cleaned


def _extract_noun_tokens(text: str, slang_lookup: dict[str, str]) -> list[str]:
    tokens: list[str] = []
    if _JIEBA_AVAILABLE and pseg is not None:
        for word, flag in pseg.cut(text):  # type: ignore[attr-defined]
            cleaned = (word or "").strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in STOPWORDS or lowered.isdigit() or lowered in slang_lookup:
                continue
            if flag.startswith("n") or flag in {"eng", "nz"}:
                tokens.append(cleaned)
    else:
        for chunk in _iter_basic_chunks(text):
            lowered = chunk.lower()
            if lowered in STOPWORDS or lowered.isdigit() or lowered in slang_lookup:
                continue
            if _looks_like_candidate(chunk):
                tokens.append(chunk)
    return tokens


def _looks_like_candidate(token: str) -> bool:
    if re.fullmatch(r"[\u4e00-\u9fff]+", token):
        return len(token) >= 2
    if re.fullmatch(r"[a-z]+", token, re.IGNORECASE):
        return len(token) >= 3
    return False


def load_slang_tokens(path: Path = SLANG_XLSX) -> list[str]:
    global _SLANG_CACHE
    if _SLANG_CACHE is not None:
        return _SLANG_CACHE

    tokens: list[str] = []
    if not path.exists():
        _SLANG_CACHE = tokens
        return tokens

    try:
        with ZipFile(path) as zf:
            shared_strings = _read_shared_strings(zf)
            sheet_name = "xl/worksheets/sheet1.xml"
            if sheet_name not in zf.namelist():
                sheet_name = next(
                    (name for name in zf.namelist() if name.startswith("xl/worksheets/sheet")),
                    None,
                )
            if not sheet_name:
                _SLANG_CACHE = tokens
                return tokens

            sheet_root = ET.fromstring(zf.read(sheet_name))
            seen: set[str] = set()
            for row in sheet_root.findall(f".//{XLSX_NS}row"):
                token = ""
                for cell in row.findall(f"{XLSX_NS}c"):
                    ref = cell.get("r", "")
                    col = "".join(ch for ch in ref if ch.isalpha()) or "A"
                    if col != "A":
                        continue
                    value = _cell_value_from_xml(cell, shared_strings).strip()
                    if value:
                        token = value
                    break
                if not token:
                    continue
                if token.strip().lower() == "token":
                    continue
                if token in seen:
                    continue
                cleaned = token.strip()
                if not _looks_like_candidate(cleaned):
                    continue
                if cleaned.isdigit():
                    continue
                if not re.search(r"[\u4e00-\u9fffA-Za-z]", cleaned):
                    continue
                if len(cleaned) > 12:
                    continue
                seen.add(cleaned)
                tokens.append(cleaned)
    except Exception:
        tokens = []

    _SLANG_CACHE = tokens
    return tokens


def _read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall(f"{XLSX_NS}si"):
        text_parts = [node.text or "" for node in si.findall(f".//{XLSX_NS}t")]
        strings.append("".join(text_parts))
    return strings


def _cell_value_from_xml(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find(f"{XLSX_NS}v")
    if value_node is None or value_node.text is None:
        return ""
    if cell.get("t") == "s":
        idx = int(value_node.text)
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return ""
    return value_node.text


def _is_special_cq(content: str) -> bool:
    lowered = content.lower()
    return "cq" in lowered


def _log_slang_hits(counter: Counter[str], top: int = 10) -> None:
    hits = [(token, count) for token, count in counter.most_common(top) if count > 0]
    if not hits:
        return
    summary = ", ".join(f"{token}:{count}" for token, count in hits)
    print(f"[topic_graph] slang hits => {summary}")


def _bucket_timestamp(ts: datetime) -> datetime:
    minute_bucket = ts.minute - (ts.minute % CO_OCCURRENCE_BUCKET_MINUTES)
    return ts.replace(minute=minute_bucket, second=0, microsecond=0)


if __name__ == "__main__":
    import json

    graphs = build_topic_graphs()
    print(json.dumps(graphs, ensure_ascii=False, indent=2))
