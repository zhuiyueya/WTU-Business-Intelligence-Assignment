from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile, ZIP_DEFLATED

BASE_DIR = Path(__file__).parent
CSV_FILES = [
    BASE_DIR / "internet_slangs.csv",
    BASE_DIR / "网络热梗.csv",
]
OUTPUT_XLSX = BASE_DIR / "网络热梗.xlsx"


def read_tokens(paths: Iterable[Path]) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            col = header[0] if header else "token"
            for row in reader:
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                key = raw.lower()
                if key in seen:
                    continue
                seen.add(key)
                tokens.append(raw)
    return tokens


def write_minimal_xlsx(tokens: list[str], path: Path) -> None:
    if not tokens:
        raise ValueError("No tokens to write")

    shared_strings = "".join(
        f"  <si><t>{html.escape(t)}</t></si>\n" for t in tokens
    )
    shared_strings_xml = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"{len(tokens)}\" uniqueCount=\"{len(tokens)}\">
{shared_strings}</sst>"""

    rows_xml = []
    # header
    rows_xml.append("<row r=\"1\"><c r=\"A1\" t=\"str\"><is><t>token</t></is></c></row>")
    for idx, token in enumerate(tokens, start=2):
        rows_xml.append(
            f"<row r=\"{idx}\"><c r=\"A{idx}\" t=\"s\"><v>{idx-2}</v></c></row>"
        )

    sheet_xml = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <sheetData>
    {' '.join(rows_xml)}
  </sheetData>
</worksheet>"""

    workbook_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <sheets>
    <sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/>
  </sheets>
</workbook>"""

    content_types_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
  <Default Extension=\"xml\" ContentType=\"application/xml\"/>
  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>
  <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>
  <Override PartName=\"/xl/sharedStrings.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml\"/>
</Types>"""

    rels_root_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>
</Relationships>"""

    workbook_rels_xml = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>
  <Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" Target=\"sharedStrings.xml\"/>
</Relationships>"""

    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_root_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/sharedStrings.xml", shared_strings_xml)


def main() -> None:
    tokens = read_tokens(CSV_FILES)
    if not tokens:
        print("No tokens found in CSV files.")
        return
    write_minimal_xlsx(tokens, OUTPUT_XLSX)
    print(f"Merged {len(tokens)} unique tokens into {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
