r"""
Sort session folders by session type (Cntrl / Zero / Rotate).

Expected input layout:
    ROOT_DIR\<AnimalID>\<Arena>\Day<N>\<k>_<SessionType>
e.g. X:\...\All_TT_PlaceTrue\Fa1059\Open\Day10\2_Zero

Each session folder is copied to:
    OUTPUT_DIR\<SessionType>\<AnimalID>_Day<N>_<k><SessionType>
e.g. OUTPUT_DIR\Zero\Fa1059_Day10_2Zero
"""

import os
import re
import shutil

# ----------------------------------------------------------------------------
# HARDCODED PATHS / SETTINGS
# ----------------------------------------------------------------------------
ROOT_DIR = r"X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted\All_TT_PlaceTrue"
OUTPUT_DIR = r"X:\NMR_group_data\Runita\Analysis\Thesis\Data_v2_Accepted\SessionType_Sorted\Open"

ARENA_FOLDERS = ["Open"]            # arena subfolder(s) inside each animal folder; None = all
ANIMAL_IDS = None                   # e.g. ["Fa1059", "Fa1060"]; None = all animals in ROOT_DIR
SESSION_TYPES = ["Cntrl", "Zero", "Rotate"]

DRY_RUN = False                     # True = only print what would be copied
OVERWRITE = True                    # True = overwrite destinations that already exist
# ----------------------------------------------------------------------------

DAY_PATTERN = re.compile(r"^Day\d+$", re.IGNORECASE)
SESSION_PATTERN = re.compile(r"^(\d+)_?(" + "|".join(SESSION_TYPES) + r")$", re.IGNORECASE)


def list_dirs(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def list_files_rel(root):
    """All file paths under root (recursive), relative to root."""
    out = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            out.add(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def copy_missing(src_file, dst_file):
    """copytree copy_function: copy only if dst is missing or differs in size."""
    if os.path.exists(dst_file) and os.path.getsize(dst_file) == os.path.getsize(src_file):
        return dst_file
    return shutil.copy2(src_file, dst_file)


def missing_files(src, dst):
    """Relative paths present in src but absent (or wrong size) in dst."""
    bad = []
    for rel in sorted(list_files_rel(src)):
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        if not os.path.isfile(d) or os.path.getsize(d) != os.path.getsize(s):
            bad.append(rel)
    return bad


def parse_session_folder(name):
    """Return (order_number, canonical_type) or None if not a session folder."""
    m = SESSION_PATTERN.match(name)
    if m is None:
        return None
    canonical = next(t for t in SESSION_TYPES if t.lower() == m.group(2).lower())
    return m.group(1), canonical


def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"Input root not found: {ROOT_DIR}")

    animals = ANIMAL_IDS if ANIMAL_IDS is not None else list_dirs(ROOT_DIR)
    counts = {t: 0 for t in SESSION_TYPES}
    skipped, unrecognised, incomplete = [], [], []

    for animal in animals:
        animal_path = os.path.join(ROOT_DIR, animal)
        if not os.path.isdir(animal_path):
            print(f"[WARN] Animal folder missing: {animal_path}")
            continue

        arenas = ARENA_FOLDERS if ARENA_FOLDERS is not None else list_dirs(animal_path)
        for arena in arenas:
            arena_path = os.path.join(animal_path, arena)
            if not os.path.isdir(arena_path):
                continue

            for day in list_dirs(arena_path):
                if not DAY_PATTERN.match(day):
                    continue
                day_path = os.path.join(arena_path, day)

                for sess in list_dirs(day_path):
                    src = os.path.join(day_path, sess)
                    parsed = parse_session_folder(sess)
                    if parsed is None:
                        unrecognised.append(src)
                        continue
                    order, sess_type = parsed

                    dst = os.path.join(OUTPUT_DIR, sess_type, f"{animal}_{day}_{order}{sess_type}")
                    if os.path.exists(dst) and not OVERWRITE:
                        gaps = missing_files(src, dst)
                        if not gaps:
                            skipped.append(dst)
                            print(f"[SKIP] exists and complete: {dst}")
                            continue
                        print(f"[RESUME] {len(gaps)} missing file(s) in {dst}")

                    print(f"[{sess_type}] {src}  ->  {dst}")
                    if not DRY_RUN:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        try:
                            # Copies every file and subfolder; existing same-size files are kept
                            shutil.copytree(src, dst, dirs_exist_ok=True,
                                            copy_function=shutil.copy2 if OVERWRITE else copy_missing)
                        except shutil.Error as e:
                            print(f"[ERROR] some files failed in {src}: {e}")
                        gaps = missing_files(src, dst)
                        if gaps:
                            incomplete.append((src, gaps))
                            print(f"[INCOMPLETE] {len(gaps)} file(s) missing in {dst}")
                    counts[sess_type] += 1

    print("\n==== Summary ====")
    for t in SESSION_TYPES:
        print(f"{t:8s}: {counts[t]} folder(s) {'would be ' if DRY_RUN else ''}copied")
    print(f"Skipped (already exist and complete): {len(skipped)}")
    if incomplete:
        print(f"Folders with files that failed to copy ({len(incomplete)}):")
        for src, gaps in incomplete:
            print(f"   {src}: {len(gaps)} missing, e.g. {gaps[0]}")
    if unrecognised:
        print(f"Unrecognised folders inside Day folders ({len(unrecognised)}):")
        for u in unrecognised:
            print(f"   {u}")


if __name__ == "__main__":
    main()
