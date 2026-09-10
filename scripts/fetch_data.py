"""Fetch and stage ExtraSensory data (docs/TASKS.md 1A.1).

See docs/CITATIONS.md#extrasensory-dataset for the dataset citation and
docs/CITATIONS.md#extrasensory-raw-file-layout for how the URLs/format below
were confirmed. Nothing downloaded here is committed (PRD Sec 3.3); it all
lands under `data/raw/`, which .gitignore excludes.

Scope: only the two archives that give BOTH accelerometer and gyroscope from
the same device (the phone; the watch in this dataset has no gyroscope), plus
the small original-activity-labels archive:
    raw_acc.zip        6.1 GB  -- phone accelerometer, all 60 users
    proc_gyro.zip       8.7 GB  -- phone gyroscope (calibrated), all 60 users
    original_labels.zip ~1 MB  -- self-reported main-activity labels, all 60 users
Every other ExtraSensory modality (audio, magnetometer, watch sensors,
location, decomposed gravity) is out of scope per PRD Sec 2.2 and is never
fetched here.

Each big archive bundles all 60 users into one file, served with
`Accept-Ranges: bytes`. Rather than extract eagerly to ~60x per-subject loose
files (which would roughly quadruple the on-disk footprint once decompressed,
i.e. tens of extra GB we don't need), this script downloads the archives
whole, once, and leaves per-subject extraction to `ats/ingest.py`, which reads
directly out of the local zip on demand -- ordinary `zipfile` random access
against a local file is fast, so no separate extraction step is needed.

Raw per-example file layout inside the two archives (there is no README for
this; confirmed by probing the live archive's central directory over HTTP
range reads before committing to a full download):
    raw_acc/<UUID>/<example_unix_ts>.m_raw_acc.dat
    proc_gyro/<UUID>/<example_unix_ts>.m_proc_gyro.dat
Each file holds one ~20-second recording burst as whitespace-separated rows
`<device_local_clock_seconds> <x> <y> <z>` (raw_acc in units of g; proc_gyro
in rad/s), at an irregular rate nominally ~40 Hz. A burst the phone could not
record is a dummy file containing just the text 'nan'. `<example_unix_ts>` is
the same primary key used by the per-user original-labels file, which is what
lets sensor bursts line up with their ground-truth activity.

Usage:
    python scripts/fetch_data.py                   # download everything (~15GB)
    python scripts/fetch_data.py --list-subjects    # just print the 60 UUIDs
    python scripts/fetch_data.py --only original_labels,raw_acc
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "http://extrasensory.ucsd.edu/data"

# name -> (url, approx size in bytes, for progress reporting only)
ARCHIVES: dict[str, tuple[str, int]] = {
    "original_labels": (f"{BASE_URL}/additional_data_files/ExtraSensory.per_uuid_original_labels.zip", 992_570),
    "raw_acc": (f"{BASE_URL}/raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip", 6_475_399_720),
    "proc_gyro": (f"{BASE_URL}/raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip", 9_333_898_324),
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "raw"
META_DIR_NAME = "_meta"

_REPORT_EVERY = 200 * 1024 * 1024  # print progress every ~200MB


def _download_resumable(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, resuming a partial `.part` file if one exists
    (the archives are large enough that a mid-download interruption is
    likely). Relies on the server honoring `Accept-Ranges: bytes`, confirmed
    for these archives via a HEAD request before this script was written."""
    if dest.exists():
        print(f"  {dest.name}: already present, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        remaining = resp.headers.get("Content-Length")
        total = resume_from + int(remaining) if remaining is not None else None
        written = resume_from
        last_report = written
        mode = "ab" if resume_from else "wb"
        with tmp.open(mode) as out:
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                written += len(block)
                if written - last_report >= _REPORT_EVERY:
                    if total:
                        print(f"  {dest.name}: {written / 1e6:.0f}MB / {total / 1e6:.0f}MB ({100 * written / total:.0f}%)")
                    else:
                        print(f"  {dest.name}: {written / 1e6:.0f}MB")
                    last_report = written
    tmp.replace(dest)
    print(f"  {dest.name}: done ({written / 1e6:.0f}MB)")


def fetch_archives(meta_dir: Path, only: list[str] | None = None) -> dict[str, Path]:
    names = only or list(ARCHIVES)
    paths: dict[str, Path] = {}
    for name in names:
        if name not in ARCHIVES:
            raise SystemExit(f"unknown archive {name!r}; choose from {list(ARCHIVES)}")
        url, _size = ARCHIVES[name]
        dest = meta_dir / f"{name}.zip"
        print(f"[{name}]")
        _download_resumable(url, dest)
        paths[name] = dest
    return paths


def list_subjects(meta_dir: Path) -> list[str]:
    """All 60 UUIDs, read from the small original-labels archive."""
    paths = fetch_archives(meta_dir, only=["original_labels"])
    with zipfile.ZipFile(paths["original_labels"]) as zf:
        return sorted(name.split(".")[0] for name in zf.namelist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the ExtraSensory archives this project needs (accelerometer, "
        "gyroscope, main-activity labels) whole, once. Nothing here is committed "
        "(data/ is gitignored); ats/ingest.py reads subjects out of the cached "
        "zips directly, so no per-subject extraction step is needed."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Directory to cache archives under.")
    parser.add_argument(
        "--only",
        help=f"Comma-separated subset of archives to fetch: {', '.join(ARCHIVES)}. Default: all.",
    )
    parser.add_argument("--list-subjects", action="store_true", help="Print all 60 subject UUIDs and exit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    meta_dir = Path(args.out) / META_DIR_NAME

    if args.list_subjects:
        for uuid in list_subjects(meta_dir):
            print(uuid)
        return

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    fetch_archives(meta_dir, only=only)


if __name__ == "__main__":
    main()
