import os
import sys

# ---- Settings ----
ROOT_DIR = r"X:\NMR_group_data\Runita\Analysis\Mean_KDE_Open_PascalOldenburg\SessionType_Sorted\Open"   # <-- change this
DRY_RUN = False                      # True = only list files; False = actually delete


def clear_files_keep_folders(root_dir, dry_run=True):
    if not os.path.isdir(root_dir):
        sys.exit(f"Not a valid directory: {root_dir}")

    n_deleted, n_failed = 0, 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if dry_run:
                print(f"[DRY RUN] would delete: {fpath}")
                n_deleted += 1
                continue
            try:
                os.chmod(fpath, 0o777)   # clear read-only flag (common on Windows)
                os.remove(fpath)
                n_deleted += 1
            except Exception as e:
                print(f"FAILED: {fpath} -> {e}")
                n_failed += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"\n{action} {n_deleted} files. Failed: {n_failed}. Folders preserved.")


if __name__ == "__main__":
    if not DRY_RUN:
        ans = input(f"Permanently delete ALL files under:\n  {ROOT_DIR}\nType YES to confirm: ")
        if ans != "YES":
            sys.exit("Aborted.")
    clear_files_keep_folders(ROOT_DIR, dry_run=DRY_RUN)
