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


def _resolve_source(meta_dir: Path, name: str) -> tuple[str, Path]:
    """Returns ('zip', path) if `<name>.zip` exists, or ('dir', path) if a
    directory named `<name>` exists instead.

    Kaggle auto-extracts uploaded zip archives while processing a dataset,
    so the exact same data can arrive locally as a zip (from
    scripts/fetch_data.py) or, on Kaggle, as an already-extracted directory
    of loose files -- this project's data can legitimately show up either
    way, so every reader in this module accepts both.
    """
    zip_path = meta_dir / f"{name}.zip"
    if zip_path.is_file():
        return "zip", zip_path
    dir_path = meta_dir / name
    if dir_path.is_dir():
        return "dir", dir_path
    raise FileNotFoundError(f"neither {zip_path} nor {dir_path} exists")


def _parse_labels_csv(f) -> dict[int, str | None]:
    reader = csv.DictReader(f)
    out: dict[int, str | None] = {}
    for row in reader:
        ts = int(row["timestamp"])
        positive = [canonical for column, canonical in LABEL_COLUMNS if row.get(column) == "1"]
        out[ts] = positive[0] if len(positive) == 1 else None
    return out


def load_original_labels(meta_dir: str | Path, subject_id: str) -> dict[int, str | None]:
    """timestamp -> canonical main-activity label, or None.

    None covers two real cases PRD Sec 3.2 asks us to handle rather than
    hide: no main-activity column fired (activity not reported that minute),
    or more than one fired (a self-reporting inconsistency, since the app's
    main-activity category is supposed to be mutually exclusive). Collapse
    rule (docs/TASKS.md 1A.7): in both cases we emit None rather than
    guessing, so the example becomes a gap in the timeline rather than a
    fabricated label.
    """
    kind, path = _resolve_source(Path(meta_dir), "original_labels")
    entry_name = f"{subject_id}.original_labels.csv.gz"
    if kind == "zip":
        with zipfile.ZipFile(path) as zf, zf.open(entry_name) as raw, gzip.open(raw, mode="rt", newline="") as f:
            return _parse_labels_csv(f)
    else:
        # Kaggle auto-decompresses .gz files while processing an uploaded
        # dataset, so the extracted directory holds plain .csv, not .csv.gz
        # -- accept either rather than assuming which one shows up.
        gz_path = path / entry_name
        csv_path = path / f"{subject_id}.original_labels.csv"
        if gz_path.is_file():
            with gzip.open(gz_path, mode="rt", newline="") as f:
                return _parse_labels_csv(f)
        with csv_path.open(newline="", encoding="ascii") as f:
            return _parse_labels_csv(f)


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


def _iter_bursts(meta_dir: Path, name: str, subject_id: str, suffix: str):
    """Yields (timestamp, raw_bytes) for one channel ('raw_acc' or
    'proc_gyro'), whether the archive arrived as a zip or an
    already-extracted directory (see `_resolve_source`)."""
    kind, path = _resolve_source(meta_dir, name)
    if kind == "zip":
        with zipfile.ZipFile(path) as zf:
            prefix = f"{name}/{subject_id}/"
            for entry in zf.namelist():
                if not entry.startswith(prefix) or not entry.endswith(suffix):
                    continue
                ts = int(entry.rsplit("/", 1)[-1].split(".", 1)[0])
                yield ts, zf.read(entry)
    else:
        # The archive's internal entries are already prefixed with `<name>/`
        # (e.g. `raw_acc/<uuid>/...` -- see docs/CITATIONS.md#extrasensory-raw-file-layout),
        # so extracting `<name>.zip` into a directory *also* named `<name>`
        # double-nests one level (`<name>/<name>/<uuid>/...`) -- exactly what
        # Kaggle's dataset processing does. Accept either layout.
        for subject_dir in (path / subject_id, path / name / subject_id):
            if subject_dir.is_dir():
                for file in subject_dir.iterdir():
                    if file.name.endswith(suffix):
                        ts = int(file.name.split(".", 1)[0])
                        yield ts, file.read_bytes()
                return


def load_subject(data_dir: str | Path, subject_id: str) -> Subject:
    """Load one subject's bursts directly out of the cached archives.

    `data_dir` is the same directory passed to `scripts/fetch_data.py --out`
    (default `data/raw`); the archives themselves live under its `_meta/`
    subdirectory, matching where that script writes them, as either zip
    archives (local) or already-extracted directories (e.g. Kaggle, which
    auto-extracts an uploaded dataset's zip files) -- see `_resolve_source`.
    """
    meta_dir = Path(data_dir) / "_meta"
    labels = load_original_labels(meta_dir, subject_id)

    acc_by_ts: dict[int, tuple[tuple[float, float, float, float], ...]] = {}
    for ts, data in _iter_bursts(meta_dir, "raw_acc", subject_id, ".m_raw_acc.dat"):
        raw = _read_dat_bytes(data)
        acc_by_ts[ts] = tuple((t, x * G_TO_MS2, y * G_TO_MS2, z * G_TO_MS2) for t, x, y, z in raw)

    gyro_by_ts: dict[int, tuple[tuple[float, float, float, float], ...]] = {}
    for ts, data in _iter_bursts(meta_dir, "proc_gyro", subject_id, ".m_proc_gyro.dat"):
        gyro_by_ts[ts] = _read_dat_bytes(data)

    all_ts = sorted(set(acc_by_ts) | set(gyro_by_ts))
    bursts = []
    for ts in all_ts:
        acc = acc_by_ts.get(ts, ())
        gyro = gyro_by_ts.get(ts, ())
        if not acc and not gyro:
            continue  # both channels dummy/missing: contributes nothing, becomes a gap
        bursts.append(Burst(example_ts=float(ts), acc=acc, gyro=gyro, activity=labels.get(ts)))

    return Subject(subject_id=subject_id, bursts=tuple(bursts))
