from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

INPUT_CSV = Path("message.csv")
SUMMARY_JSON = Path("risk_summary.json")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
FLAGGED_KEYWORDS = [
    "兼职",
    "招聘",
    "急招",
    "代写",
    "出售",
    "转让",
    "广告",
    "黄牛",
    "考试答案",
    "退款",
    "投诉",
    "举报",
    "诈骗",
    "违规",
    "外挂",
]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    total_by_day: defaultdict[str, int] = defaultdict(int)
    late_by_day: defaultdict[str, int] = defaultdict(int)
    keyword_counts: defaultdict[str, int] = defaultdict(int)
    session_stats: dict[str, dict] = {}

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            content = (row.get("message_content") or "").lower()
            ts_str = row.get("message_time_utc")
            if not ts_str:
                continue
            ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
            day_key = ts.strftime("%Y-%m-%d")
            total_by_day[day_key] += 1
            session_id = row["session_qq"]
            session_name = row.get("session_name", "")

            stats = session_stats.setdefault(
                session_id,
                {
                    "name": session_name,
                    "flagged": 0,
                    "late_night": 0,
                    "total": 0,
                },
            )
            stats["total"] += 1

            if 0 <= ts.hour < 5:
                late_by_day[day_key] += 1
                stats["late_night"] += 1

            flagged = False
            for kw in FLAGGED_KEYWORDS:
                if kw.lower() in content:
                    keyword_counts[kw] += 1
                    flagged = True
            if flagged:
                stats["flagged"] += 1

    late_night = [
        {
            "day": day,
            "late_messages": late,
            "share": round(late / total_by_day[day], 3) if total_by_day[day] else 0,
        }
        for day, late in late_by_day.items()
        if late
    ]
    late_night.sort(key=lambda item: item["late_messages"], reverse=True)

    keyword_alerts = [
        {"keyword": kw, "count": count}
        for kw, count in sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)
        if count
    ]

    session_risks = [
        {
            "id": session_id,
            "name": stats["name"],
            "flagged_messages": stats["flagged"],
            "late_night_messages": stats["late_night"],
        }
        for session_id, stats in session_stats.items()
        if stats["flagged"] or stats["late_night"]
    ]
    session_risks.sort(
        key=lambda item: (item["flagged_messages"], item["late_night_messages"]), reverse=True
    )

    summary = {
        "lateNightDays": late_night[:10],
        "keywordAlerts": keyword_alerts[:10],
        "sessionRisks": session_risks[:10],
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Risk summary saved to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
