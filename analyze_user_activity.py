from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

INPUT_CSV = Path("message.csv")
OUTPUT_CSV = Path("user_activity_summary.csv")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def analyze(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    per_user: dict[str, dict] = {}
    interval_buckets: defaultdict[str, list[float]] = defaultdict(list)

    for row in rows:
        user = row["user_qq"]
        session = row["session_qq"]
        ts_str = row.get("message_time_utc")
        if not ts_str:
            continue
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        message = row.get("message_content", "") or ""
        message_len = len(message.strip())
        session_type = row.get("session_type", "")

        stats = per_user.setdefault(
            user,
            {
                "total_messages": 0,
                "group_messages": 0,
                "private_messages": 0,
                "sessions": set(),
                "first_ts": ts,
                "last_ts": ts,
                "total_length": 0,
                "active_days": set(),
                "last_seen_ts": None,
            },
        )

        stats["total_messages"] += 1
        if session_type == "group":
            stats["group_messages"] += 1
        else:
            stats["private_messages"] += 1

        stats["sessions"].add(session)
        stats["active_days"].add(ts.date())
        stats["total_length"] += message_len
        if ts < stats["first_ts"]:
            stats["first_ts"] = ts
        if ts > stats["last_ts"]:
            stats["last_ts"] = ts

        last_seen = stats["last_seen_ts"]
        if last_seen is not None:
            gap_minutes = (ts - last_seen).total_seconds() / 60.0
            if gap_minutes >= 0:
                interval_buckets[user].append(gap_minutes)
        stats["last_seen_ts"] = ts

    summarized: list[dict[str, str]] = []
    for user, stats in per_user.items():
        total = stats["total_messages"]
        avg_len = stats["total_length"] / total if total else 0
        span_days = (stats["last_ts"] - stats["first_ts"]).days or 1
        engagement_density = total / span_days
        gap_list = interval_buckets[user]
        avg_gap = sum(gap_list) / len(gap_list) if gap_list else 0
        session_count = len(stats["sessions"])
        bridge_score = round(math.log(total + 1, 2) * (1 + math.log(session_count + 1, 2)), 3)

        summarized.append(
            {
                "user_qq": user,
                "total_messages": str(total),
                "group_messages": str(stats["group_messages"]),
                "private_messages": str(stats["private_messages"]),
                "unique_sessions": str(session_count),
                "active_days": str(len(stats["active_days"])),
                "engagement_span_days": str(span_days),
                "messages_per_span_day": f"{engagement_density:.2f}",
                "avg_message_length": f"{avg_len:.1f}",
                "avg_gap_minutes": f"{avg_gap:.1f}",
                "bridge_score": f"{bridge_score:.3f}",
                "first_message_utc": stats["first_ts"].isoformat(sep=" "),
                "last_message_utc": stats["last_ts"].isoformat(sep=" "),
            }
        )

    summarized.sort(key=lambda row: int(row["total_messages"]), reverse=True)
    return summarized


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_highlights(rows: list[dict[str, str]], top_n: int = 10) -> None:
    print(f"Top {top_n} users by total messages:")
    for row in rows[:top_n]:
        print(
            f"{row['user_qq']}: {row['total_messages']} msgs, "
            f"{row['unique_sessions']} sessions, bridge {row['bridge_score']}"
        )


def main() -> None:
    rows = load_rows(INPUT_CSV)
    summary = analyze(rows)
    write_summary(summary, OUTPUT_CSV)
    print_highlights(summary)


if __name__ == "__main__":
    main()
