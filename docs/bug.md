# Bug: accelerometer scaled ~9.8× too high for some subjects

| | |
|---|---|
| **Status** | Fixed in `ats/ingest.py` (per-subject unit detection); CNN retrain still pending on Kaggle |
| **Severity** | High: affects classifier training data, recognition results, and explanations, and blocks the Phase 4 stillness check |
| **Owner** | Member A (`ats/ingest.py`) |
| **Reported by** | Member B, 2026-09-11 |
| **Found by** | the held-out check in `scripts/calibrate_stillness.py` (docs/TASKS.md task 4B.3) |

## Summary

For `subj_real_b` (`74B86067-5D4B-43CF-82CF-341B76BEA0F4`), every accelerometer value in both committed tracks is about **9.7 times too large**: the recording's median acceleration magnitude is **94.75 m/s²**, where physics requires about **9.81 m/s²** (gravity). `subj_real_a` reads a correct **9.78 m/s²**. The likely cause is that `ats/ingest.py` converts every subject's accelerometer from g to m/s², while this subject's raw file was probably already in m/s², so it is converted twice.

## Why the number must be near 9.81

A phone's accelerometer always measures gravity plus the wearer's motion. Averaged over a whole recording, the magnitude sits close to 9.81 m/s² whatever the wearer does: lying, sitting or walking. A median of 94.75 m/s² is physically impossible for a phone carried by a person.

## Evidence

From the committed tracks. No raw data is needed to see it.

| Measure | `subj_real_a` | `subj_real_b` | Expected |
|---|---:|---:|---|
| Median acceleration magnitude (m/s²) | **9.78** | **94.75** | ≈ 9.81 |
| Implied raw value (÷ 9.80665) | 1.00 | **9.66** | ≈ 1.00 if the file is in g |
| Median accelerometer-magnitude std, lying down (m/s²) | 0.0099 | 0.1310 | small, similar across phones |
| Gyroscope-energy floor, 5th percentile of all windows | 0.0002 | 0.0193 | small, similar across phones |

The second row is the telling one. Divided back by the conversion factor, `subj_real_b`'s raw values come out at about 9.66, which is already gravity in m/s². A file in g would come out at about 1.

## How to reproduce

Using only files already in the repo:

```bash
python scripts/calibrate_stillness.py
```

Its first lines print:

```
subj_real_a: median acceleration magnitude 9.78 m/s^2 (near gravity)
subj_real_b: median acceleration magnitude 94.75 m/s^2 (NOT near gravity: units problem)
```

Or directly, for any track:

```bash
python -c "from ats.aggregate import load_track; from ats.signal import recording_gravity; print(recording_gravity(load_track('tests/fixtures/track_subj_real_b.jsonl')))"
```

## Where in the code

- `ats/ingest.py:30-34`: the comment and constant `G_TO_MS2 = 9.80665` state that ExtraSensory's `raw_acc` values are in g.
- `ats/ingest.py:232`: `acc_by_ts[ts] = tuple((t, x * G_TO_MS2, y * G_TO_MS2, z * G_TO_MS2) ...)` applies that conversion to **every** subject, unconditionally.

## Suspected cause (a hypothesis to verify, not confirmed)

ExtraSensory users carried different phones, and the raw accelerometer files are probably in each device's native units: g for some platforms, m/s² for others. I have not confirmed this from the dataset documentation. Please check the ExtraSensory docs or any per-user phone metadata. Either way, the data itself shows the single fixed conversion is wrong for at least this subject.

## Impact

1. **Classifier training data.** If several of the 60 subjects are affected, the CNN was trained on inputs whose scale differs by about 10× between subjects. That may account for part of the low recognition performance (macro-F1 0.2749). This is unmeasured until the fix is in and the model is retrained.
2. **Real-model tracks for affected subjects.** The model sees out-of-scale inputs at inference too, so their predictions are suspect.
3. **Explanations.** Every explanation for an affected subject quotes `feature_summary` values, so it currently reports impossible numbers, e.g. "accelerometer magnitude averaging 94.75 m/s^2". That hurts the explanation-faithfulness rubric (PRD §7.3.5).
4. **Phase 4 stillness check (4B.3).** It can't be validated on `subj_real_b`: 0% of its 41,000 windows can look still. Member B has added a guard so stillness is only judged on recordings whose median magnitude is 8–12 m/s², so no wrong answers go out meanwhile. But the check stays open until this is fixed.
5. **Phase 3 numbers.** The delta table and Figures 1 and 3 include `subj_real_b`, and all are marked draft in docs/TASKS.md partly for this reason.

The gyroscope may be affected too: `subj_real_b`'s gyroscope-energy floor is about 100× `subj_real_a`'s. That could be a different sensor, or another units difference such as degrees versus radians per second. It's less clear-cut than the accelerometer, but worth checking while you are in there.

## Suggested fix (your call)

- Decide the units **per subject** rather than assuming g for all. For example, take the median raw magnitude across the subject's bursts: close to 1 means g, so convert; close to 9.8 means it is already m/s², so don't. Log the decision for each subject. If ExtraSensory has per-user phone metadata, that is an even better source.
- Add a check after ingestion: every subject's median acceleration magnitude must lie within 8–12 m/s², raising or warning otherwise. That catches any future units mistake in either direction.
- Check the gyroscope floor across subjects in the same pass.

## How to verify the fix

- [x] The median acceleration magnitude of every one of the 60 subjects lies within 8–12 m/s². Commit a small per-subject summary (e.g. `results/acc_units_by_subject.csv`) showing each subject's value and the units decision. **59/60 land in range** (`scripts/check_acc_units.py`); one subject (`BEF6C611-50DA-4971-A040-87FB979F3FC1`, raw median 3.16) is genuinely ambiguous and is flagged rather than guessed at -- neither g nor m/s² fits, so its accelerometer is left unconverted and out of range on purpose. Worth a mention in the report as a known data-quality limit, not a bug in the detector.
- [x] A test covers both cases: a synthetic subject recorded in g and one recorded in m/s² both ingest to about 9.81 m/s² (`tests/test_oracle.py::test_accelerometer_already_in_g_is_converted_to_ms2`, `::test_accelerometer_already_in_ms2_is_not_double_converted`).
- [x] `tests/fixtures/track_subj_real_{a,b}.jsonl` and `tests/fixtures/real_model_tracks/` are regenerated, and `python scripts/calibrate_stillness.py` reports **both** subjects near gravity (subj_real_a 9.78, subj_real_b 9.66 m/s², both "near gravity"). Note: subj_real_a's fixture files came out byte-identical to what was already committed -- it was already correctly detected as g-scale under the old unconditional conversion, so only subj_real_b's files actually changed.
- [ ] The feature dataset is rebuilt and the CNN retrained, with before/after Phase 2 numbers in `docs/results_recognition.md`. **Pending** -- this needs a Kaggle run, not done locally.

## After the fix (Member B will re-run)

The gold answers in `data/questions_dev_v2/` come from labels, not sensor values, so they should not change. Once the regenerated tracks are in, Member B re-runs:

```bash
python scripts/calibrate_stillness.py
python scripts/make_real_dev_questions.py
python scripts/oracle_delta.py --oracle-dir tests/fixtures --real-dir tests/fixtures/real_model_tracks --questions-dir data/questions_dev_v2
python scripts/make_fig1.py
python scripts/make_fig3.py
```

The Phase 3 caveats and the 4B.3 status in docs/TASKS.md are then updated with the new numbers.

Member B has not edited `ats/ingest.py`; it is Member A's file.

---

# Issue 2: a recording without labels cannot be loaded, which blocks the system graders run

| | |
|---|---|
| **Status** | Fixed in `ats/ingest.py` |
| **Severity** | High: without it the "runnable system" deliverable (PRD §9.4) fails on any recording that arrives without labels |
| **Owner** | Member A (`ats/ingest.py`) |
| **Reported by** | Member B, 2026-09-11 |

## Summary

At evaluation time the graders hand over a recording and questions, not labels. The entry point is now wired for that: `python -m ats.answer --recording <data_dir> --subject <id> --questions <path> --out <path>` calls `ats.recognize.build_track`, which calls `ats.ingest.load_subject`. But `load_subject` reads the labels first (`ats/ingest.py:227`), and `load_original_labels` raises `FileNotFoundError` when `_meta/original_labels.zip` or its directory is missing (`ats/ingest.py:77-83`, `125-140`). So an unlabelled recording fails before any recognition runs.

## How to reproduce

```bash
python -m pytest tests/test_answer_cli.py::test_an_unlabelled_recording_can_be_loaded -rxX
```

The test builds a minimal ExtraSensory-layout recording (`_meta/raw_acc/<id>/1000.m_raw_acc.dat` and `_meta/proc_gyro/<id>/1000.m_proc_gyro.dat`) with no labels archive. It currently raises `FileNotFoundError`, and is marked `xfail(strict=True)` so the suite stays green until the fix.

## Suggested fix (your call)

When the labels archive, or the subject's entry in it, is missing, load every burst with `activity=None`, the value already used for unlabelled minutes. If training should still insist on labels, put this behind an explicit flag and let `ats.recognize` pass it.

## How to verify the fix

- [x] `tests/test_answer_cli.py::test_an_unlabelled_recording_can_be_loaded` passes. The `xfail` marker is removed in the same change.
- [x] The issue 1 units fix also applies on this path -- `load_subject` runs `_detect_acc_scale` regardless of whether labels are present, since the two code paths were merged rather than kept separate.

## Also needed for the system graders run: trained weights

`--model` defaults to `models/full/activity_cnn.pt`, but `models/` is gitignored and the weights are not on Member B's machine, so the full path can't be run there yet. The model is about 10k parameters (tens of kilobytes), so committing the final trained weights may be the simplest way to ship a runnable system. Decide together, and check with the instructor what format the evaluation recordings will arrive in.
