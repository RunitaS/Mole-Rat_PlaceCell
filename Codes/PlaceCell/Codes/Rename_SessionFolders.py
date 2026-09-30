"""
Rename Neuropixel session folders to keep only the text after the last underscore.

Folder layout expected under ROOT_DIR:
    Animal ID > Arena type > Recording day > Session type > data

Example:
    ...\Fa168037BD\Circle\Day1\1680378B_Day1_CntrlCntrlCntrlZero_Shank1BankA_Cntrl1
    becomes
    ...\Fa168037BD\Circle\Day1\Cntrl1

Run with DRY_RUN = True first to preview the changes, then set it to False.
"""

import os

ROOT_DIR = r"X:\NMR_group_data\Runita\Analysis\Thesis\Neuropixel"
DRY_RUN = False  # True = only print what would be renamed; False = actually rename


def subdirs(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def main():
    renamed, skipped = 0, 0

    for animal in subdirs(ROOT_DIR):
        animal_path = os.path.join(ROOT_DIR, animal)
        for arena in subdirs(animal_path):
            arena_path = os.path.join(animal_path, arena)
            for day in subdirs(arena_path):
                day_path = os.path.join(arena_path, day)
                sessions = subdirs(day_path)

                # Detect sessions within the same day that would end up with the same name
                new_names = [s.rsplit("_", 1)[-1] for s in sessions]
                duplicates = {n for n in new_names if new_names.count(n) > 1}

                for old_name, new_name in zip(sessions, new_names):
                    if "_" not in old_name:
                        continue  # already relabelled
                    old_path = os.path.join(day_path, old_name)
                    new_path = os.path.join(day_path, new_name)

                    if not new_name:
                        print(f"[SKIP] Nothing after last underscore: {old_path}")
                        skipped += 1
                    elif new_name in duplicates:
                        print(f"[SKIP] Multiple sessions would become '{new_name}': {old_path}")
                        skipped += 1
                    elif os.path.exists(new_path):
                        print(f"[SKIP] Target already exists: {new_path}")
                        skipped += 1
                    else:
                        print(f"{'[DRY RUN] ' if DRY_RUN else ''}{old_path}  ->  {new_name}")
                        if not DRY_RUN:
                            os.rename(old_path, new_path)
                        renamed += 1

    action = "Would rename" if DRY_RUN else "Renamed"
    print(f"\n{action} {renamed} folder(s); skipped {skipped}.")


if __name__ == "__main__":
    main()
