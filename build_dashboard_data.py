from __future__ import annotations

import csv
import json
from pathlib import Path

from topic_graph_builder import build_topic_graphs
BASE_DIR = Path(__file__).resolve().parent

USER_ACTIVITY = BASE_DIR / "user_activity_summary.csv"
SESSION_HEALTH = BASE_DIR / "session_health_summary.csv"
ACTIVITY_DAY = BASE_DIR / "activity_by_day.csv"
ACTIVITY_HOUR = BASE_DIR / "activity_by_hour.csv"
ACTIVITY_WEEKDAY = BASE_DIR / "activity_by_weekday.csv"
KEYWORDS_OVERALL = BASE_DIR / "keywords_overall.csv"
DAILY_ACTIVE_USERS = BASE_DIR / "daily_active_users.csv"
WEEKLY_ENGAGEMENT = BASE_DIR / "weekly_engagement.csv"
RETENTION_METRICS = BASE_DIR / "retention_metrics.json"
RISK_SUMMARY = BASE_DIR / "risk_summary.json"
KMEANS_INSIGHTS = BASE_DIR / "kmeans_insights.json"
CONTENT_TOPIC = BASE_DIR / "content_topic_analysis.json"
OUTPUT_JSON = BASE_DIR / "dashboard_data.json"

MANUAL_TOPIC_TYPES = [
    {"name": "饮食类", "count": 9043},
    {"name": "运动类", "count": 5212},
    {"name": "学习类", "count": 2742},
    {"name": "天气类", "count": 1246},
    {"name": "活动类", "count": 943},
    {"name": "生活类", "count": 851},
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_dataset() -> dict:
    users = read_csv(USER_ACTIVITY)[:10]
    sessions = read_csv(SESSION_HEALTH)[:10]
    activity_day = read_csv(ACTIVITY_DAY)
    activity_hour = read_csv(ACTIVITY_HOUR)
    activity_weekday = read_csv(ACTIVITY_WEEKDAY)
    keywords = read_csv(KEYWORDS_OVERALL)[:20]
    daily_active = read_csv(DAILY_ACTIVE_USERS)
    weekly_engagement = read_csv(WEEKLY_ENGAGEMENT)

    retention_metrics = {}
    if RETENTION_METRICS.exists():
        retention_metrics = json.loads(RETENTION_METRICS.read_text(encoding="utf-8"))

    risk_summary = {}
    if RISK_SUMMARY.exists():
        risk_summary = json.loads(RISK_SUMMARY.read_text(encoding="utf-8"))

    content_topic = {}
    if CONTENT_TOPIC.exists():
        content_topic = json.loads(CONTENT_TOPIC.read_text(encoding="utf-8"))

    kmeans_insights = {}
    if KMEANS_INSIGHTS.exists():
        kmeans_insights = json.loads(KMEANS_INSIGHTS.read_text(encoding="utf-8"))

    data = {
        "topUsers": [
            {
                "id": row["user_qq"],
                "totalMessages": parse_int(row["total_messages"]),
                "uniqueSessions": parse_int(row["unique_sessions"]),
                "bridgeScore": float(row["bridge_score"]),
                "activeDays": parse_int(row["active_days"]),
            }
            for row in users
        ],
        "topSessions": [
            {
                "id": row["session_qq"],
                "name": row["session_name"],
                "messages": parse_int(row["messages"]),
                "uniqueUsers": parse_int(row["unique_users"]),
                "gini": float(row["gini_participation"]),
                "topUserShare": float(row["top_user_share"]),
                "peakHour": row["peak_hour_utc"],
            }
            for row in sessions
        ],
        "activityByDay": [
            {"day": row.get("day") or row[next(iter(row))], "messages": parse_int(row["messages"])}
            for row in activity_day
        ],
        "activityByHour": [
            {"hour": row.get("hour") or row[next(iter(row))], "messages": parse_int(row["messages"])}
            for row in activity_hour
        ],
        "activityByWeekday": [
            {
                "weekday": row.get("weekday") or row[next(iter(row))],
                "messages": parse_int(row["messages"]),
            }
            for row in activity_weekday
        ],
        "keywords": [
            {"token": row["token"], "count": parse_int(row["count"])}
            for row in keywords
            if row.get("token")
        ],
        "retention": {
            "metrics": retention_metrics,
            "dailyActiveUsers": [
                {"day": row["day"], "users": parse_int(row["active_users"])}
                for row in daily_active
            ],
            "weeklyEngagement": [
                {
                    "week": row["week_start"],
                    "newUsers": parse_int(row["new_users"]),
                    "returningUsers": parse_int(row["returning_users"]),
                }
                for row in weekly_engagement
            ],
        },
        "risk": {
            "lateNight": risk_summary.get("lateNightDays", []),
            "keywordAlerts": risk_summary.get("keywordAlerts", []),
            "sessionRisks": risk_summary.get("sessionRisks", []),
        },
        "contentTopic": content_topic,
        "topicTypeCounts": MANUAL_TOPIC_TYPES,
        "kmeansInsights": kmeans_insights,
        "topicGraphs": build_topic_graphs(top_n=10),
    }
    return data


def main() -> None:
    data = build_dataset()
    OUTPUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
