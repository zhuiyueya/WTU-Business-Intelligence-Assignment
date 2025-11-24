from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

INPUT_CSV = Path("message.csv")
OUTPUT_CSV = Path("session_health_summary.csv")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def gini(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    cumulative = 0.0
    for count in sorted(counter.values()):
        cumulative += count
    numerator = 0.0
    running = 0.0
    for count in sorted(counter.values()):
        running += count
        numerator += running - count / 2
    return 1 - 2 * numerator / (total * len(counter))


def analyze(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    sessions: dict[str, dict] = {}

    for row in rows:
        session = row["session_qq"]
        ts_str = row.get("message_time_utc")
        if not ts_str:
            continue
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        user = row["user_qq"]
        stats = sessions.setdefault(
            session,
            {
                "name": row.get("session_name", ""),
                "type": row.get("session_type", ""),
                "messages": 0,
                "users": Counter(),
                "days": set(),
                "first_ts": ts,
                "last_ts": ts,
                "hour_buckets": defaultdict(int),
            },
        )

        stats["messages"] += 1
        stats["users"][user] += 1
        stats["days"].add(ts.date())
        stats["hour_buckets"][ts.hour] += 1
        if ts < stats["first_ts"]:
            stats["first_ts"] = ts
        if ts > stats["last_ts"]:
            stats["last_ts"] = ts

    summarized: list[dict[str, str]] = []
    for session_id, stats in sessions.items():
        duration = (stats["last_ts"] - stats["first_ts"]).total_seconds() / 3600 or 1
        active_days = len(stats["days"])
        span_days = max((stats["last_ts"].date() - stats["first_ts"].date()).days, 0) + 1
        quiet_days = span_days - active_days
        messages = stats["messages"]
        users = len(stats["users"])
        top_user_share = max(stats["users"].values()) / messages if messages else 0
        inequality = gini(stats["users"])
        peak_hour = max(stats["hour_buckets"], key=stats["hour_buckets"].get, default="NA")

        summarized.append(
            {
                "session_qq": session_id,
                "session_name": stats["name"],
                "session_type": stats["type"],
                "messages": str(messages),
                "unique_users": str(users),
                "active_days": str(active_days),
                "span_days": str(span_days),
                "quiet_days": str(quiet_days),
                "messages_per_active_day": f"{messages / active_days:.2f}" if active_days else "0",
                "messages_per_hour_span": f"{messages / duration:.2f}",
                "top_user_share": f"{top_user_share:.2f}",
                "gini_participation": f"{inequality:.2f}",
                "peak_hour_utc": str(peak_hour),
                "first_message_utc": stats["first_ts"].isoformat(sep=" "),
                "last_message_utc": stats["last_ts"].isoformat(sep=" "),
            }
        )

    summarized.sort(key=lambda row: int(row["messages"]), reverse=True)
    return summarized


def main() -> None:
    rows = load_rows(INPUT_CSV)
    summary = analyze(rows)
    if not summary:
        print("No sessions found.")
        return
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} session records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
