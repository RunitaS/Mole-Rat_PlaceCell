"""
List every session folder under ROOT_DIR and save them to an Excel sheet in ROOT_DIR.

Folder layout expected under ROOT_DIR:
    Animal ID > Arena type > Recording day > Session type > data

Each row of column A holds one session path relative to ROOT_DIR, e.g.
    Fa168037BD\Circle\Day1\Cntrl1
"""

import os

from openpyxl import Workbook

ROOT_DIR = r"X:\NMR_group_data\Runita\Analysis\Thesis\Neuropixel"
OUTPUT_NAME = "SessionFolders.xlsx"


def subdirs(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def main():
    rows = []
    for animal in subdirs(ROOT_DIR):
        animal_path = os.path.join(ROOT_DIR, animal)
        for arena in subdirs(animal_path):
            arena_path = os.path.join(animal_path, arena)
            for day in subdirs(arena_path):
                day_path = os.path.join(arena_path, day)
                for session in subdirs(day_path):
                    rows.append(os.path.join(animal, arena, day, session))

    wb = Workbook()
    ws = wb.active
    ws.title = "Sessions"
    for i, rel_path in enumerate(rows, start=1):
        ws.cell(row=i, column=1, value=rel_path)
    ws.column_dimensions["A"].width = max((len(r) for r in rows), default=10) + 2

    out_path = os.path.join(ROOT_DIR, OUTPUT_NAME)
    wb.save(out_path)
    print(f"Wrote {len(rows)} session folder(s) to {out_path}")


if __name__ == "__main__":
    main()
