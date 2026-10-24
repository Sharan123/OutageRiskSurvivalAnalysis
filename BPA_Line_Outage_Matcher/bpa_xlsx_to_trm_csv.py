"""Convert the 'Transmission Line Interruptions' table of a BPA OutagesCY<year>.xlsx file into the
layout of Datasets/BPA_trm/OutagesCY<year>-TransmissionLineInterruptions.csv.

Source: https://transmission.bpa.gov/Business/Operations/Outages/OutagesCY<year>.xlsx
Each workbook has one sheet with several stacked tables; this takes the rows between the
"Transmission Line Interruptions" title and the next table title. BPA stores every cell as text,
so cells are copied unchanged (empty cells -> ""). " (PPT)" is removed from the 2022+ headers so
every year has the same column names. Output: title row + header + rows, CRLF, ASCII.

usage: python BPA_Line_Outage_Matcher/bpa_xlsx_to_trm_csv.py <OutagesCY<year>.xlsx> <out.csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import openpyxl

TITLE = "Transmission Line Interruptions"


def _text(v) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ValueError(f"non-text cell {v!r} ({type(v).__name__}); BPA workbooks store text only")
    return v


def read_line_table(xlsx: str | Path) -> tuple[list[str], list[list[str]]]:
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        rows = list(wb.worksheets[0].iter_rows(values_only=True))
    finally:
        wb.close()
    starts = [i for i, r in enumerate(rows) if isinstance(r[0], str) and r[0].strip() == TITLE]
    if len(starts) != 1:
        raise ValueError(f"{xlsx}: expected one '{TITLE}' title, found {len(starts)}")
    hdr_row = rows[starts[0] + 1]
    header = [_text(h).replace(" (PPT)", "") for h in hdr_row if h is not None]
    n = len(header)
    out = []
    for r in rows[starts[0] + 2:]:
        cells = [_text(v) for v in r[:n]]
        rest = [v for v in r[n:] if v not in (None, "")]
        if r[0] is not None and all(v is None for v in r[1:]):  # next table title (or blank title row)
            break
        if all(c == "" for c in cells):
            if out:  # blank separator after the table
                break
            continue
        if rest:
            raise ValueError(f"{xlsx}: values beyond the {n} header columns: {r}")
        out.append(cells)
    return header, out


def write_csv(header: list[str], rows: list[list[str]], dst: str | Path) -> None:
    with open(dst, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow([TITLE] + [""] * (len(header) - 1))
        w.writerow(header)
        w.writerows(rows)


def main(src: str, dst: str) -> None:
    header, rows = read_line_table(src)
    write_csv(header, rows, dst)
    print(f"{dst}: {len(rows)} rows, {len(header)} columns")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
