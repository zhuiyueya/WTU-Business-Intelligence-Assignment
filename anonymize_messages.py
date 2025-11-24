from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Hashable


def _next_label(mapping: Dict[Hashable, int], key: Hashable) -> int:
    label = mapping.get(key)
    if label is None:
        label = len(mapping) + 1
        mapping[key] = label
    return label


def anonymize_messages(input_path: Path, output_path: Path) -> None:
    user_map: dict[str, int] = {}
    session_map: dict[str, int] = {}
    direction_map: dict[str, int] = {}
    account_map: dict[str, int] = {}
    remark_map: dict[tuple[int, int], str] = {}
    rows: list[dict[str, str]] = []

    with input_path.open(newline="", encoding="utf-8-sig") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row")
        fieldnames = reader.fieldnames

        for original_row in reader:
            row = original_row.copy()

            user_id = _next_label(user_map, original_row.get("user_qq", ""))
            session_id = _next_label(session_map, original_row.get("session_qq", ""))
            direction_id = _next_label(direction_map, original_row.get("direction", ""))
            account_value = original_row.get("account_id", "")
            account_label = ""
            if account_value:
                account_label = f"A{_next_label(account_map, account_value)}"

            user_label = f"U{user_id}"
            session_label = f"S{session_id}"

            row["user_qq"] = user_label
            user_name = f"用户{user_id}"
            row["user_name"] = user_name
            row["raw_sender_nickname"] = user_name
            row["raw_target_name"] = user_name

            row["session_qq"] = session_label
            row["session_name"] = f"会话{session_id}"
            row["raw_target_id"] = user_label

            remark_key = (user_id, session_id)
            remark_value = original_row.get("user_remark_in_session", "").strip()
            if remark_value:
                remark_label = remark_map.setdefault(
                    remark_key, f"备注{user_id}-{session_id}"
                )
            else:
                remark_label = ""
            row["user_remark_in_session"] = remark_label

            row["direction"] = f"D{direction_id}"
            row["account_id"] = account_label

            rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    source = project_root / "messages_export_1(3).csv"
    target = project_root / "message.csv"
    anonymize_messages(source, target)
