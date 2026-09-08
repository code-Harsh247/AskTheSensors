# Product Requirements Document
## Ask the Sensors — Grounded, Explainable Activity Question Answering from Wearable Signals

**Course:** CS60055 Ubiquitous Computing, Department of Computer Science and Engineering, IIT Kharagpur
**Assignment:** Hackathon Challenge 1
**Document version:** 1.0
**Source of truth:** [CS60055_Challenge1.pdf](CS60055_Challenge1.pdf) — this PRD translates that brief into engineering requirements. Wherever the two disagree, the brief governs.

---

## 1. Background and Problem Statement

### 1.1 Scenario
Aparna, 72, lives alone; her son works remotely in another city. He does not want continuous video/audio surveillance of her — he wants to **ask plain-language questions** and trust the answers. Her smartwatch already records raw accelerometer and gyroscope streams continuously. A physiotherapist tracking her recovery has related, more quantitative questions (e.g., is her walking time trending up week over week) and wants any claim **checkable against the underlying signal**, not taken on trust.

### 1.2 Core problem
Raw motion signals are meaningless to a lay reader, and an ordinary activity classifier only partially helps: it emits a per-instant label but never says **for how long, how often, at what time, or on what basis**. In a healthcare context, an unsupported verdict is not actionable — a clinician (or a worried son) will not act on "the machine said so." Every answer must arrive **with the evidence that supports it**: the exact stretch of signal and the reason it was read as one activity versus another.

### 1.3 What must NOT be built
- A system that answers using language-model reasoning alone, with no reference to the recorded signal. This explicitly does **not** meet the requirement, regardless of how plausible the answer sounds.
- A system that only handles a single fixed question type — the same system must handle all four task tiers (see §4).

---

## 2. Goals and Non-Goals

### 2.1 Goals
1. Build a **sensor question-answering service**: given (a) a multimodal sensor recording and (b) a natural-language query about the activity in that recording, return one structured answer that is accurate and grounded in specific cited signal evidence.
2. Support all four question tiers (identification, temporal/quantitative reasoning, evidence grounding, open-world reasoning) from a **single system**, since the evaluation will mix questions from every tier.
3. Report the **resource cost** of the system (size, memory, latency, and ideally energy/utilization) on a clearly stated target device/emulation — not just accuracy.
4. (Extra credit) Demonstrate an accuracy-vs-cost tradeoff for a compressed/edge-deployable variant of the system.
5. Produce a reproducible artifact: a GitHub repository the teaching team can clone, set up, and rerun to obtain the reported results.

### 2.2 Non-goals
- Building a surveillance/video system — the input is restricted to accelerometer + gyroscope motion data only.
- Handling modalities beyond accelerometer and gyroscope (ExtraSensory has more, but this challenge restricts scope to these two).
- Optimizing for a leaderboard on a known/disclosed test set — the evaluation set and exact questions are undisclosed until evaluation time; the system must generalize, not overfit to seen data.
- Chasing raw code volume or cleverness — this is explicitly a **design challenge**, graded on soundness of design, correctness, accuracy, and grounding, not lines of code.

---

## 3. Data Requirements

### 3.1 Dataset
- **Source:** ExtraSensory dataset (http://extrasensory.ucsd.edu/) — wearable recordings gathered in the wild from 60 users during ordinary daily life.
- **Modalities in scope:** triaxial accelerometer (X, Y, Z) and triaxial gyroscope (X, Y, Z) only. All other ExtraSensory modalities (audio, location, etc.) are out of scope.
- **Activity/posture label set (exactly 7 classes):**
  1. Lying down
  2. Sitting
  3. Standing in place
  4. Standing and moving
  5. Walking
  6. Running
  7. Bicycling

### 3.2 Preprocessing requirements
- **Resample every sensor stream to 25 Hz** before any downstream processing, so that all groups/systems share a common time base. This is a hard requirement, not a suggestion.
- The data must be treated as **realistic and messy by design**, and this messiness must be handled, not sidestepped:
  - **Uneven sampling** (irregular timestamps).
  - **Missing stretches** of data (gaps).
  - **Self-reported, possibly overlapping labels** (label noise/ambiguity).
  - **Heavy class imbalance** toward sedentary behavior (lying/sitting/standing dominate over walking/running/bicycling).
- A solution that stays reliable under noise, gaps, and imbalance is explicitly valued **over** one that only performs well on a clean subset. Robustness to degraded input is itself a graded dimension (§7.4, Robustness curve).

### 3.3 Data handling / repo hygiene
- **Do not commit the raw dataset** to the repository.
- The repository **must** contain the code/scripts/instructions needed to fetch or prepare the data from source.

---

## 4. Functional Requirements — The Four Tasks

A single system must correctly handle all four tiers below; the evaluation mixes questions from every tier without telling you which tier a given question belongs to.

### 4.1 Task 1 — Activity Identification
- **Purpose:** verify the system reads the sensor stream correctly and names the activity.
- **Question forms:**
  - Open identification (e.g., "What activity is the user performing?")
  - Binary verification (e.g., "Is the user running?")
- **Scoring surface:** only the `Answer` field is scored at this tier. Evidence fields **may** be `N/A`, though filling them in is welcomed and encouraged wherever possible.
- **Example (from brief):**
  ```
  Query: "What activity is the user performing?"
  Answer: Walking
  Activity/Event: Walking
  Evidence: Timestamp(s): N/A | Sensor Modality: N/A | Sensor Channel(s): N/A
  Explanation: N/A
  ```

### 4.2 Task 2 — Temporal and Quantitative Reasoning
- **Purpose:** reason across the **whole recording**, not a single window.
- **Question forms:** how long an activity lasted; how many times it occurred; when a particular event took place; how two activities compare in total time.
- **Requirement:** answers must be **computed from detected activity intervals** and must reflect the user's behavior over the full recording (not a single snapshot).
- **Scoring surface:** evidence **is expected** here — report the interval(s) the answer rests on.
- **Examples (from brief):**
  ```
  Query: "How long was the user walking?"
  Answer: 700 seconds
  Activity/Event: Walking
  Evidence: Timestamp(s): 905 to 1420, 2110 to 2295 (seconds from start)
            Sensor Modality: Accelerometer, Gyroscope
            Sensor Channel(s): All
  Explanation: Walking was detected in two separate intervals, of 515 and 185
               seconds, which sum to 700 seconds.
  ```
  ```
  Query: "Did the user spend more time walking or running?"
  Answer: Walking
  Activity/Event: Walking, Running
  Evidence: Timestamp(s): Walking = 3480 seconds total, Running = 1440 seconds total
            Sensor Modality: Accelerometer, Gyroscope
            Sensor Channel(s): All
  Explanation: Total walking time exceeded total running time over the recording.
  ```

### 4.3 Task 3 — Evidence Grounding
- **Purpose:** make the evidence itself the object of assessment. A correct answer is **necessary but not sufficient**.
- **Requirement:** every response at this tier must supply:
  1. The answer.
  2. The temporal location of the supporting evidence.
  3. The sensor modality it is drawn from.
  4. The specific channel(s).
  5. The reasoning that ties the cited pattern to the conclusion.
- **Example (from brief):**
  ```
  Query: "Did the user begin running at any point, and if so, when?"
  Answer: Yes, running began at 1512 seconds
  Activity/Event: Onset of running
  Evidence: Timestamp(s): 1512 to 1980 (seconds from start)
            Sensor Modality: Accelerometer, Gyroscope
            Sensor Channel(s): All
  Explanation: A sustained rise in accelerometer magnitude at a higher step
               frequency, together with larger gyroscope oscillations, marks the
               transition from lower-intensity gait to running at the cited time.
  ```

### 4.4 Task 4 — Open-World Activity Reasoning
- **Purpose:** move past the fixed 7-class label set. Questions probe the **meaning** of the motion rather than its class name.
- **Question forms:** inferring a broad behavior; judging whether observed evidence is consistent with a described scenario; explaining why a pattern looks the way it does; reading body dynamics/temporal structure for behaviors the system was **never trained to name**.
- **Requirement:** both evidence and explanation are **mandatory** at this tier; the answer must be **argued from the signal**, even for untrained/unlabeled behaviors.
- **Examples (from brief):**
  ```
  Query: "Did the user lie down for a prolonged period?"
  Answer: Likely yes
  Activity/Event: Prolonged lying down
  Evidence: Timestamp(s): 6300 to 9420 (seconds from start)
            Sensor Modality: Accelerometer, Gyroscope
            Sensor Channel(s): All
  Explanation: A long, continuous stretch of near-zero acceleration variance and
               minimal gyroscope activity, well beyond any brief stationary pause,
               is consistent with sustained rest rather than a transient stop.
  ```
  ```
  Query: "Was the user using a wheeled or pedal-based mode of movement?"
  Answer: Yes
  Activity/Event: Unknown outdoor physical activity, consistent with cycling
  Evidence: Timestamp(s): 2400 to 3120 (seconds from start)
            Sensor Modality: Accelerometer, Gyroscope
            Sensor Channel(s): All
  Explanation: The segment shows smooth, continuous, cyclic acceleration at a
               steady cadence, without the discrete heel-strike spikes of walking
               or running, accompanied by sustained periodic gyroscope
               oscillation consistent with pedaling and balance, which points to
               a low-impact wheeled mode.
  ```

---

## 5. Output Format Specification (must be followed exactly)

Every response, for every task tier, must use the following fields **in this order**:

```
Answer:              <direct answer to the query, or N/A>
Activity/Event:      <activity or event, or N/A>
Evidence:
    Timestamp(s):        <time range or ranges, or N/A>
    Sensor Modality:     <accelerometer, gyroscope, or both, or N/A>
    Sensor Channel(s):   <for example Acc X/Y/Z, Gyro X/Y/Z, or All, or N/A>
Explanation:         <reasoning grounded in the observed signal, or N/A>
```

### 5.1 Format rules
1. **Timestamp convention must be consistent and stated.** Choose exactly one time base for all answers: either absolute clock time (`HH:MM:SS`) or seconds elapsed from the start of the recording. State which convention was chosen. (The brief's own examples use seconds-from-start.)
2. **Evidence-field requirement varies by tier:**
   - Task 1: evidence fields may be `N/A` (only `Answer` is scored).
   - Task 2: evidence is expected.
   - Task 3: evidence is required and directly assessed.
   - Task 4: evidence and explanation are mandatory.
3. **Default to providing evidence wherever possible**, even when not strictly required for that tier — "wherever you can support an answer with evidence, do so."

---

## 6. Non-Functional Requirements — Efficiency and Cost

The brief is explicit that **accuracy alone is only half the goal**; ubicomp systems run on small, battery-powered, often-disconnected devices, and living only in the cloud sacrifices privacy and availability.

### 6.1 Minimum required reporting (mandatory, not extra credit)
Alongside accuracy figures, report:
1. **Model size** — both in parameters and on-disk size.
2. **Peak memory used** during inference.
3. **Time taken to answer a single query** (latency).
4. Where feasible, also report **processor utilization** and an **estimate of energy spent per query**.

### 6.2 Target device disclosure
- Must **clearly state the target** on which these numbers were measured/emulated: e.g., laptop CPU, single-board computer (e.g., Raspberry Pi), embedded accelerator (e.g., Jetson-class board), mid-range Android phone, or an explicitly constrained/emulated environment.
- **Clear, consistent reporting matters more than which specific hardware is chosen.**

### 6.3 Extra credit — edge deployability
- Available for a solution that runs **accurately on a small, resource-constrained device**.
- Must **demonstrate**, not merely assert, the accuracy/overhead tradeoff:
  - Present **at least two operating points**: (a) a full model, and (b) a compressed model obtained via quantization, pruning, or knowledge distillation.
  - Plot **accuracy against one or more cost axes** so the shape of the tradeoff is visible (this feeds the "Accuracy versus overhead" figure, §7.4.4).
- Reward criterion: trading a **small** amount of accuracy for a **large** reduction in cost, with the curve to prove it.

---

## 7. Evaluation, Metrics, and Required Figures

### 7.1 Evaluation protocol
- **Both the sensor recordings and the questions used for grading are undisclosed until evaluation time.** Build for the general problem, not a leaderboard you can study in advance.
- The evaluation set will include **difficult and edge-case questions** chosen specifically to probe robustness.
- This is explicitly framed as a **design challenge, not a coding test**. Marks reward design quality, implementation correctness, and accuracy/performance — not code volume or cleverness.

### 7.2 Grading dimensions and weights (indicative; instructor may confirm/adjust)

| Dimension | What is assessed | Weight |
|---|---|---|
| Design and approach | Soundness of the overall design and reasoning, given the problem's constraints | 25% |
| Correctness and reproducibility | A correct implementation the teaching team can rerun from the repository to reproduce reported results | 20% |
| Accuracy across the four tasks | Answer quality on identification, temporal/quantitative reasoning, and open-world reasoning | 35% |
| Evidence grounding | Whether cited timestamps, modalities, and channels are correct and clearly support the stated answer | 20% |
| Efficiency and tradeoff | Accuracy held against measured resource cost, shown as a clear accuracy-vs-overhead curve | up to +10% (extra credit) |

### 7.3 Correctness rules by answer type
Because answers come in different kinds (categorical labels, yes/no verdicts, numeric values, temporal intervals, free-form judgments), **no single metric applies to all of them** — the correctness rule must be stated per type, and the overall headline number must be a breakdown by question type plus one macro-averaged overall score (macro-averaged across question types, specifically so that abundant easy sedentary examples do not dominate the score).

#### 7.3.1 Categorical answers (identification, comparison, yes/no plausibility)
- **Primary metric:** accuracy (exact match count / total).
- **Also report macro-F1**, since the data is heavily imbalanced:
  - Per class: precision = TP/(TP+FP), recall = TP/(TP+FN).
  - F1 = 2 × precision × recall / (precision + recall).
  - Macro-F1 = average of per-class F1.
  - Balanced accuracy = mean of per-class recall (serves the same imbalance-correcting purpose) — report as well.
- **Binary verification specifically:** report precision, recall, and F1 **on the positive class**, plus **specificity**, since plain accuracy hides a bias toward always answering "no."

#### 7.3.2 Numeric answers (durations, counts)
- Exact match is too harsh — use **accuracy within tolerance**: an answer counts as correct when its absolute error is within a chosen threshold.
  - Threshold may be **absolute** (e.g., within a few seconds) or **relative** (e.g., within ±10% for a duration, or ±1 for a count).
- Also report the **size of the error**:
  - **Mean Absolute Error (MAE)** — required.
  - **Mean Absolute Percentage Error (MAPE)** for durations — optional but recommended, since it is threshold-independent.

#### 7.3.3 Temporal answers and cited evidence intervals (event times, onsets, evidence spans)
- Use **Intersection over Union (IoU)** between a predicted interval and the ground-truth interval: overlap length / union length.
- When multiple intervals are involved, either:
  - Match predicted to true intervals and average the IoUs, **or**
  - Compute temporal precision (overlap / predicted length) and temporal recall (overlap / true length), then take their F1.
- **Grounding accuracy** = fraction of intervals whose IoU meets a chosen threshold — this is exactly what the "accuracy versus strictness" figure (§7.4.3) traces.

#### 7.3.4 Evidence grounding as a combined rule
- An answer counts as **grounded and correct** only when **all three** hold simultaneously:
  1. The answer itself is correct.
  2. The cited interval reaches the IoU threshold.
  3. The cited modality and channel(s) match the reference.
- Report this **combined grounded-accuracy figure alongside plain answer accuracy** — the gap between the two is "the price of demanding evidence" and is worth reporting explicitly.
- A lighter, interval-label-free proxy is also acceptable/useful: **grounding precision** — the fraction of answers whose cited interval, per the ground-truth activity labels, actually contains the named activity.

#### 7.3.5 Open-world reasoning answers
- Score categorically wherever a reference/intended broad behavior exists (map the free-form answer to the intended behavior label).
- The explanation must be judged for **faithfulness and plausibility** via a short rubric on a **1–5 scale**, applied by either human graders or an LLM-judge with a fixed rubric, rating whether:
  1. The reasoning cites real signal features.
  2. Those features actually support the stated conclusion.
  3. The conclusion is plausible.
- Report the **mean rubric score**, and **inter-grader agreement** where humans do the grading.
- **Embedding similarity to a reference explanation** may be used as a cheap automatic proxy but must stay **secondary**, since it rewards surface-level overlap rather than genuine faithfulness.

### 7.4 Required figures (minimum five)
Individual queries may be plotted along the x-axis for qualitative error analysis, but the **reportable/graded figures must aggregate by question type**.

1. **Accuracy by question type** — grouped bar chart; x-axis = question type (identification, verification, duration, count, comparison, grounding, open-world reasoning); y-axis = accuracy metric; include a final bar for the overall score. **Caption must state the correctness rule used per group**, since the rule differs across groups.
2. **Activity confusion matrix** — heatmap over the 7 classes for the recognition backbone (true activity on rows, predicted on columns), reported **alongside per-class precision, recall, and F1**. Purpose: expose systematic confusions accuracy alone hides (e.g., sitting vs. standing-in-place, walking vs. running).
3. **Accuracy versus strictness** — line plot:
   - For numeric answers: x-axis = error tolerance.
   - For temporal answers/cited intervals: x-axis = IoU threshold, swept from 0.1 to 0.9.
   - y-axis (both cases) = fraction of answers accepted.
   - Purpose: reveals whether misses are near-misses or wild misses; the value at a fixed threshold gives one comparable number.
4. **Accuracy versus overhead** — scatter plot; x-axis = a resource-cost axis (model size in MB, single-query latency, peak memory, or energy per query); y-axis = overall QA accuracy; each point = one configuration (full model, 8-bit quantized, pruned, distilled); **join the non-dominated points into a Pareto frontier**. This is the figure that supports the edge extra credit.
5. **Robustness curve** — line plot; x-axis = a controlled degradation of the input (added sensor noise, percentage of dropped samples, or sampling rate reduced below 25 Hz); y-axis = accuracy on a fixed question set. A curve that stays flat as conditions worsen is stronger evidence of a usable system than any single clean number.

---

## 8. Suggested System Architecture (non-binding starting point)

The brief explicitly frames this as *a* reasonable approach, not the only one — teams are encouraged to bring their own design, but any design must still satisfy the grounding constraint below.

**Suggested layered pipeline:**
1. **Preprocessing layer** — cleans and resamples the raw accel/gyro streams to 25 Hz; divides into analysis windows.
2. **Recognition layer** — assigns activity evidence to each window, using either engineered features + a light classifier, or a compact neural model. Keeping this layer small is what enables later edge targeting.
3. **Aggregation layer** — turns per-window evidence into intervals, durations, counts, and transitions — the structured timeline that temporal questions (Task 2) query against.
4. **Interface layer** — maps a natural-language question to the right operation over the structured evidence/timeline (candidate approach: a small language model — see §8.1) and writes the result in the required output format (§5).

**Hard constraint on the interface layer:** whatever design is chosen, the language it produces must be **tied to evidence the earlier layers actually found**, never generated freely. This is the architectural expression of the "no answer from language reasoning alone" rule in §1.3.

**Recommended build order:** begin with a small, correct pipeline that runs end-to-end over all seven activities; confirm well-formed output for a few questions of each of the four types; only then push on accuracy, evidence quality, and efficiency.

### 8.1 Suggested language models (for the interface layer)
Open-source small language models (SLMs) are suggested, e.g.:
- Ministral-3
- Llama 3.2 (1B and 3B)
- Qwen2.5 / Qwen3 (0.5B, 1.5B, 3B)
- or a comparable SLM

Teams are free to choose any design and model, but **must justify the choice in the report**.

### 8.2 Related readings (provided in brief, for background)
1. https://dl.acm.org/doi/10.1145/3699765
2. https://dl.acm.org/doi/abs/10.1145/3810210
3. https://dl.acm.org/doi/pdf/10.1145/3749496
4. https://dl.acm.org/doi/abs/10.1145/3699747

---

## 9. Rules, Logistics, and Deliverables

### 9.1 Team structure
- Groups of **2–3 members**.
- Every member is expected to contribute.
- The report must include a **short statement of what each member did**.

### 9.2 Repository requirements
- Each group keeps **one GitHub repository** for the challenge, created **at the start** and used **throughout**.
- **Commit history must reflect real, incremental work** over the challenge period — a single end-of-project upload is explicitly against the rules and is part of how effort is judged.
- Repository must contain:
  1. Source code.
  2. A **README** explaining environment setup and how to reproduce results.
  3. Scripts/configuration to run the system on a **fresh recording** and produce output in the required format.
  4. Code and instructions to **fetch or prepare** the dataset — **not** the raw dataset itself (do not commit it).

### 9.3 Academic integrity — AI use and plagiarism
- CS60055's announced AI-use and plagiarism policies apply **in full, without exception**.
- Design and reasoning must be the group's own.
- Every external resource used (dataset, pretrained model, software library, code fragment) must be **cited precisely at the point of use**.
- Copying another group's work, in whole or in part, is **not permitted**.
- Fabricated results, misattributed data, and undisclosed reuse are treated as **policy violations**.
- If unsure whether a use is allowed, **ask before relying on it**.

### 9.4 Deliverables checklist
1. **GitHub repository** — code, README, reproducible setup (per §9.2).
2. **Technical report, ~10–12 pages**, covering:
   - The problem as the team frames it.
   - Design and the reasoning behind it.
   - Implementation details.
   - Results across all four tasks.
   - Accuracy-vs-overhead analysis (if attempting extra credit).
   - Per-member contribution statement.
3. **A runnable system** that, given a data recording and a set of questions at evaluation time, produces answers in the required output format.
4. **A short demonstration or presentation**, per the schedule announced in class.

### 9.5 Logistics not yet fixed
The brief notes: *"A few logistical details, shown in brackets, will be confirmed separately."* At the time of writing, the only bracketed placeholders resolved in the brief are the three teaching-team GitHub usernames above. Any remaining logistics (exact deadline, demo schedule, submission portal) should be tracked once announced — **flag this PRD for an update once the deadline and demo schedule are confirmed in class.**

---

## 10. Traceability Matrix (requirement → PRD section)

| Brief requirement | PRD section |
|---|---|
| Scenario / motivation | §1.1 |
| No language-only answers | §1.3, §8 (hard constraint) |
| Dataset, modalities, 7 classes | §3.1 |
| 25 Hz resampling | §3.2 |
| Handle noise/gaps/imbalance | §3.2 |
| Don't commit raw dataset | §3.3, §9.2 |
| Output format fields/order | §5 |
| Timestamp convention | §5.1 |
| Task 1 spec + example | §4.1 |
| Task 2 spec + examples | §4.2 |
| Task 3 spec + example | §4.3 |
| Task 4 spec + examples | §4.4 |
| Resource cost minimum reporting | §6.1 |
| Target device disclosure | §6.2 |
| Edge extra credit + Pareto proof | §6.3, §7.4.4 |
| Evaluation set undisclosed / generalize | §7.1 |
| Grading weights table | §7.2 |
| Per-answer-type correctness rules | §7.3 |
| Macro-averaged overall QA accuracy | §7.3 |
| 5 required figures | §7.4 |
| Suggested architecture diagram | §8 |
| Suggested SLMs | §8.1 |
| Related readings | §8.2 |
| Groups of 2-3, contribution statement | §9.1 |
| Repo access grants, commit history | §9.2 |
| AI-use and plagiarism policy | §9.3 |
| Deliverables (repo, report, system, demo) | §9.4 |

---

## 11. Open Questions / Risks (not resolved by the brief)

- Exact submission deadline and demo date — not present in the source document; confirm in class and update §9.5.
- Exact IoU/tolerance thresholds to use for grading are left to each team to choose and justify — no single fixed number is prescribed; our chosen thresholds must be stated explicitly in the report (per §7.3.2, §7.3.3).
- Whether the LLM-judge rubric (Task 4 explanations) will be graded by humans or an automated grader is left to team discretion; whichever we pick must be justified and inter-rater/consistency behavior reported (§7.3.5).
- Target hardware for the efficiency numbers (§6.2) is our choice — needs a decision early since it affects instrumentation code.
