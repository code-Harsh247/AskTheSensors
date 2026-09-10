"""ExtraSensory ingestion: read one subject's raw accelerometer/gyroscope
bursts and self-reported main-activity labels out of the archives
`scripts/fetch_data.py` caches, and map them onto the frozen canonical class
order (docs/TASKS.md task 1A.2).

See docs/CITATIONS.md#extrasensory-dataset for the dataset citation and
docs/CITATIONS.md#extrasensory-raw-file-layout for the file format this
module depends on.

Label mapping: the dataset's "original" (pre-cleaning) labels include a
`main activity` category that the app enforced as mutually exclusive at
report time, with exactly the seven values PRD Sec 3.1 asks for. That is a
much cleaner match to our canonical class order than the "cleaned" labels
used in the primary features file (which fold walking/running through a
`FIX_` correction step and drop standing-in-place into a synthesized
`OR_standing`), so this module reads from `original_labels.zip` rather than
the primary features/labels file. The tradeoff, stated once here: original
labels are the dataset's own less-reliable, pre-cleaning version -- worth
restating in the report.
"""

from __future__ import annotations

import csv
import gzip
import zipfile
from dataclasses import dataclass
from pathlib import Path

# ExtraSensory's raw_acc values are in units of g; our internal convention
# (matching the synthetic Phase 1 fixtures in scripts/make_dev_fixture.py) is
# physical acceleration in m/s^2, so every accelerometer sample is scaled by
# standard gravity on the way in.
G_TO_MS2 = 9.80665

# original_labels.zip column name -> canonical class, in the frozen order
# (docs/TASKS.md Sec 0). The "main activity" category in the ExtraSensory app
# was mutually exclusive by design; column names taken verbatim from the
# archive's header row.
LABEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("original_label:LYING_DOWN", "LYING"),
    ("original_label:SITTING", "SITTING"),
    ("original_label:STANDING_IN_PLACE", "STANDING_STILL"),
    ("original_label:STANDING_AND_MOVING", "STANDING_MOVING"),
    ("original_label:WALKING", "WALKING"),
    ("original_label:RUNNING", "RUNNING"),
    ("original_label:BICYCLING", "BICYCLING"),
)


@dataclass(frozen=True)
class Burst:
    """One example's raw recording session (nominally ~20 seconds)."""

    example_ts: float  # unix seconds; the example's primary key
    acc: tuple[tuple[float, float, float, float], ...]  # (t_local, x, y, z), m/s^2
    gyro: tuple[tuple[float, float, float, float], ...]  # (t_local, x, y, z), rad/s
    activity: str | None  # canonical class, or None if unlabeled/ambiguous


@dataclass(frozen=True)
class Subject:
    subject_id: str
    bursts: tuple[Burst, ...]  # sorted by example_ts, ambiguous-only-if-both-empty excluded


def load_original_labels(zip_path: str | Path, subject_id: str) -> dict[int, str | None]:
    """timestamp -> canonical main-activity label, or None.

    None covers two real cases PRD Sec 3.2 asks us to handle rather than
    hide: no main-activity column fired (activity not reported that minute),
    or more than one fired (a self-reporting inconsistency, since the app's
    main-activity category is supposed to be mutually exclusive). Collapse
    rule (docs/TASKS.md 1A.7): in both cases we emit None rather than
    guessing, so the example becomes a gap in the timeline rather than a
    fabricated label.
    """
    with zipfile.ZipFile(zip_path) as zf:
        entry = f"{subject_id}.original_labels.csv.gz"
        with zf.open(entry) as raw, gzip.open(raw, mode="rt", newline="") as f:
            reader = csv.DictReader(f)
            out: dict[int, str | None] = {}
            for row in reader:
                ts = int(row["timestamp"])
                positive = [
                    canonical
                    for column, canonical in LABEL_COLUMNS
                    if row.get(column) == "1"
                ]
                out[ts] = positive[0] if len(positive) == 1 else None
    return out


def _read_dat_bytes(data: bytes) -> tuple[tuple[float, float, float, float], ...]:
    """Parse one `.m_raw_acc.dat` / `.m_proc_gyro.dat` file's bytes: rows of
    `local_clock x y z`, or a dummy file containing just 'nan' when the phone
    could not record that burst (returns empty in that case)."""
    text = data.decode("ascii", errors="strict").strip()
    if not text or text == "nan":
        return ()
    rows: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        t, x, y, z = (float(p) for p in parts)
        rows.append((t, x, y, z))
    return tuple(rows)


def load_subject(data_dir: str | Path, subject_id: str) -> Subject:
    """Load one subject's bursts directly out of the cached local archives.

    `data_dir` is the same directory passed to `scripts/fetch_data.py --out`
    (default `data/raw`); the archives themselves live under its `_meta/`
    subdirectory, matching where that script writes them. Nothing is
    extracted to loose files; `zipfile` random access against a local file
    is fast enough to read on demand.
    """
    meta_dir = Path(data_dir) / "_meta"
    labels = load_original_labels(meta_dir / "original_labels.zip", subject_id)

    acc_by_ts: dict[int, tuple[tuple[float, float, float, float], ...]] = {}
    with zipfile.ZipFile(meta_dir / "raw_acc.zip") as zf:
        prefix = f"raw_acc/{subject_id}/"
        for name in zf.namelist():
            if not name.startswith(prefix) or not name.endswith(".m_raw_acc.dat"):
                continue
            ts = int(name.rsplit("/", 1)[-1].split(".", 1)[0])
            raw = _read_dat_bytes(zf.read(name))
            acc_by_ts[ts] = tuple((t, x * G_TO_MS2, y * G_TO_MS2, z * G_TO_MS2) for t, x, y, z in raw)

    gyro_by_ts: dict[int, tuple[tuple[float, float, float, float], ...]] = {}
    with zipfile.ZipFile(meta_dir / "proc_gyro.zip") as zf:
        prefix = f"proc_gyro/{subject_id}/"
        for name in zf.namelist():
            if not name.startswith(prefix) or not name.endswith(".m_proc_gyro.dat"):
                continue
            ts = int(name.rsplit("/", 1)[-1].split(".", 1)[0])
            gyro_by_ts[ts] = _read_dat_bytes(zf.read(name))

    all_ts = sorted(set(acc_by_ts) | set(gyro_by_ts))
    bursts = []
    for ts in all_ts:
        acc = acc_by_ts.get(ts, ())
        gyro = gyro_by_ts.get(ts, ())
        if not acc and not gyro:
            continue  # both channels dummy/missing: contributes nothing, becomes a gap
        bursts.append(Burst(example_ts=float(ts), acc=acc, gyro=gyro, activity=labels.get(ts)))

    return Subject(subject_id=subject_id, bursts=tuple(bursts))
