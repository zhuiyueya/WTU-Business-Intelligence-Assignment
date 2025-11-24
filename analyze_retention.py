from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

INPUT_CSV = Path("message.csv")
DAILY_OUTPUT = Path("daily_active_users.csv")
WEEKLY_OUTPUT = Path("weekly_engagement.csv")
SUMMARY_JSON = Path("retention_metrics.json")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def start_of_week(day: datetime.date) -> datetime.date:
    return day - timedelta(days=day.weekday())


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    daily_users: defaultdict[datetime.date, set[str]] = defaultdict(set)
    weekly_users: defaultdict[datetime.date, set[str]] = defaultdict(set)
    session_last_ts: dict[str, datetime] = {}
    response_gaps: list[float] = []
    user_stats: dict[str, dict] = {}

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_str = row.get("message_time_utc")
            if not ts_str:
                continue
            ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
            day = ts.date()
            week = start_of_week(day)
            user = row["user_qq"]
            session = row["session_qq"]

            daily_users[day].add(user)
            weekly_users[week].add(user)

            stats = user_stats.setdefault(
                user,
                {
                    "first": day,
                    "last": day,
                    "active_days": set(),
                    "first_week": week,
                },
            )
            if day < stats["first"]:
                stats["first"] = day
                stats["first_week"] = start_of_week(day)
            if day > stats["last"]:
                stats["last"] = day
            stats["active_days"].add(day)

            last_ts = session_last_ts.get(session)
            if last_ts is not None:
                gap_minutes = (ts - last_ts).total_seconds() / 60
                if 0 <= gap_minutes <= 24 * 60:
                    response_gaps.append(gap_minutes)
            session_last_ts[session] = ts

    daily_records = sorted(
        (
            {"day": key.isoformat(), "active_users": len(users)}
            for key, users in daily_users.items()
        ),
        key=lambda item: item["day"],
    )
    with DAILY_OUTPUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["day", "active_users"])
        writer.writeheader()
        writer.writerows(daily_records)

    weekly_records = []
    for week, users in sorted(weekly_users.items()):
        new_users = sum(1 for stats in user_stats.values() if stats["first_week"] == week)
        weekly_records.append(
            {
                "week_start": week.isoformat(),
                "new_users": new_users,
                "returning_users": max(len(users) - new_users, 0),
            }
        )
    with WEEKLY_OUTPUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["week_start", "new_users", "returning_users"]
        )
        writer.writeheader()
        writer.writerows(weekly_records)

    total_users = len(user_stats)
    if total_users == 0:
        summary = {}
    else:
        spans = [(stats["last"] - stats["first"]).days for stats in user_stats.values()]
        retained_7 = sum(1 for span in spans if span >= 7)
        retained_30 = sum(1 for span in spans if span >= 30)
        active_day_counts = [len(stats["active_days"]) for stats in user_stats.values()]

        summary = {
            "total_users": total_users,
            "retained_7d": retained_7,
            "retained_30d": retained_30,
            "avg_active_span_days": round(mean(spans), 2),
            "median_active_days": median(active_day_counts),
            "median_response_minutes": round(median(response_gaps), 1)
            if response_gaps
            else 0.0,
        }

    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Retention metrics saved: {DAILY_OUTPUT.name}, {WEEKLY_OUTPUT.name}, {SUMMARY_JSON.name}"
    )


if __name__ == "__main__":
    main()
