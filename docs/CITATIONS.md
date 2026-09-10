# Citations

Running log of every external resource this project builds on — dataset, pretrained model, software library, or borrowed code fragment — per PRD §9.3 ("cited precisely at the point of use"). Append an entry **in the same commit** as the code that uses the resource, and leave a short comment at the use site pointing back here (e.g. `# see docs/CITATIONS.md#extrasensory-dataset`).

Format per entry: what it is, where it's from, which file(s) use it, and what it's used for.

---

### ExtraSensory dataset
- **What:** wearable accelerometer + gyroscope recordings from 60 users, self-reported activity labels.
- **Source:** Vaizman, Y., Ellis, K., and Lanckriet, G. "Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches." http://extrasensory.ucsd.edu/
- **Used in:** the entire project — all training, validation, and dev/test data (PRD §3.1).

#### ExtraSensory raw file layout
- **What:** the archives and internal per-example file format `scripts/fetch_data.py` and `ats/ingest.py` depend on. Not documented in any README on the site; confirmed 2026-09-10 by inspecting the live site and, for the multi-gigabyte archives, probing the remote zip's central directory over HTTP range reads before committing to a full download.
- **Archives used** (all linked from http://extrasensory.ucsd.edu/, under `data/`): `additional_data_files/ExtraSensory.per_uuid_original_labels.zip` (~1MB, all 60 users' self-reported "original" — pre-cleaning — labels, one `<uuid>.original_labels.csv.gz` per user); `raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip` (6.1GB, phone accelerometer); `raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip` (8.7GB, phone gyroscope, calibrated/drift-corrected). Every other modality (audio, magnetometer, watch sensors, location, decomposed gravity) is out of scope per PRD §2.2 and is never fetched.
- **original_labels.csv.gz columns:** `timestamp` (unix seconds, the example's primary key) plus one `original_label:<NAME>` column per possible label, value `1`/`0`. The seven mutually-exclusive main-activity columns (`LYING_DOWN`, `SITTING`, `STANDING_IN_PLACE`, `STANDING_AND_MOVING`, `WALKING`, `RUNNING`, `BICYCLING`) map directly onto the frozen canonical class order (docs/TASKS.md §0) — see `ats/ingest.py:LABEL_COLUMNS`.
- **Raw sensor archive layout:** `raw_acc/<UUID>/<example_unix_ts>.m_raw_acc.dat` and `proc_gyro/<UUID>/<example_unix_ts>.m_proc_gyro.dat`, one file per ~20-second recording burst. Each file is whitespace-separated rows `<device_local_clock_seconds> <x> <y> <z>` (`raw_acc` in units of g; `proc_gyro` in rad/s) at an irregular rate nominally ~40Hz. A burst the phone could not record is a dummy file containing just `nan`. `<example_unix_ts>` is the same key as `original_labels.csv.gz`'s `timestamp` column, which is what lines a sensor burst up with its ground-truth label.
- **Used in:** `scripts/fetch_data.py` (download), `ats/ingest.py` (parsing + label mapping, including the g→m/s² unit conversion this discovery made necessary).

### jsonschema (Python library)
- **What:** JSON Schema Draft 2020-12 validator.
- **Source:** https://github.com/python-jsonschema/jsonschema
- **Used in:** `ats/contracts.py` — validates every artifact crossing the A↔B boundary (`window_track`, `answer`, `question_set`, `cost_report`) against the schemas in `schemas/`.

### pytest (Python library)
- **What:** test framework.
- **Source:** https://docs.pytest.org/
- **Used in:** `tests/` — all automated tests, starting with `tests/test_contracts.py`.

### numpy (Python library)
- **What:** numerical array library.
- **Source:** https://numpy.org/
- **Used in:** declared as a Phase-0 dependency (`pyproject.toml`) for the signal-processing and metric work coming in Phase 1.

---

## Still to review

PRD §8.2 lists four related readings, not yet reviewed against our design:

1. https://dl.acm.org/doi/10.1145/3699765
2. https://dl.acm.org/doi/abs/10.1145/3810210
3. https://dl.acm.org/doi/pdf/10.1145/3749496
4. https://dl.acm.org/doi/abs/10.1145/3699747

TASKS.md task 0.8 asks both members to skim these before the Phase 2 architecture is locked in. Log anything borrowed from them here when that happens.
