from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from text_filters import is_noise_message

INPUT_CSV = Path("message.csv")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

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
    "this",
    "that",
    "the",
    "for",
    "with",
    "from",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def tokenize(message: str) -> list[str]:
    message = message.lower()
    tokens: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", message):
        if not chunk.strip():
            continue
        if chunk in STOPWORDS:
            continue
        if chunk.isdigit() and len(chunk) <= 1:
            continue
        tokens.append(chunk)
    return tokens


def write_counter(counter: Counter[str], path: Path, top_n: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["token", "count"])
        for token, count in counter.most_common(top_n):
            writer.writerow([token, count])


def write_distribution(counter: Counter[str], path: Path, label: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([label, "messages"])
        for key, value in sorted(counter.items()):
            writer.writerow([key, value])


def main() -> None:
    rows = load_rows(INPUT_CSV)
    if not rows:
        print("No data available.")
        return

    by_day = Counter()
    by_hour = Counter()
    by_weekday = Counter()
    keyword_overall = Counter()
    keyword_by_type: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        ts_str = row.get("message_time_utc")
        if not ts_str:
            continue
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        day_key = ts.strftime("%Y-%m-%d")
        by_day[day_key] += 1
        by_hour[f"{ts.hour:02d}:00"] += 1
        by_weekday[ts.strftime("%a")] += 1

        message = row.get("message_content", "") or ""
        if is_noise_message(message):
            continue
        tokens = tokenize(message)
        keyword_overall.update(tokens)
        keyword_by_type[row.get("session_type", "unknown")].update(tokens)

    write_distribution(by_day, Path("activity_by_day.csv"), "day")
    write_distribution(by_hour, Path("activity_by_hour.csv"), "hour")
    write_distribution(by_weekday, Path("activity_by_weekday.csv"), "weekday")

    write_counter(keyword_overall, Path("keywords_overall.csv"))
    for session_type, counter in keyword_by_type.items():
        write_counter(counter, Path(f"keywords_{session_type}.csv"))

    print("Temporal activity and keyword files generated.")


if __name__ == "__main__":
    main()
