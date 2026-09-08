# Citations

Running log of every external resource this project builds on — dataset, pretrained model, software library, or borrowed code fragment — per PRD §9.3 ("cited precisely at the point of use"). Append an entry **in the same commit** as the code that uses the resource, and leave a short comment at the use site pointing back here (e.g. `# see docs/CITATIONS.md#extrasensory-dataset`).

Format per entry: what it is, where it's from, which file(s) use it, and what it's used for.

---

### ExtraSensory dataset
- **What:** wearable accelerometer + gyroscope recordings from 60 users, self-reported activity labels.
- **Source:** Vaizman, Y., Ellis, K., and Lanckriet, G. "Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches." http://extrasensory.ucsd.edu/
- **Used in:** the entire project — all training, validation, and dev/test data (PRD §3.1).

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
