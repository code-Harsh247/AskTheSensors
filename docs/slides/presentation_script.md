# Ask the Sensors — Presentation Script

**CS60055 Ubiquitous Computing — Hackathon Challenge 1**
**Hard limit: 10 minutes.** Measured at a normal speaking pace (140 words/minute), the script is **9:02 of speech** — about **9:30 real** once pointing, pauses and the two handoffs are added. That leaves roughly 30 seconds of buffer, which is why the "if running long" cuts below exist.

Deck: `docs/slides/slides.tex` (14 slides).

---

## Who presents what

| Speaker | Slides | Covers | Time |
|---|---|---|---|
| **ABHINAV** | 1–5 | Opening, the problem, requirements, architecture, recognition model | 3:05 |
| **HARSH** | 6–10 | Answering and checking, real output, what we did differently, results | 4:00 |
| **ABHINAV** | 11–12 | Efficiency and compression, limitations and AI use | 1:30 |
| **BOTH** | 13–14 | Contributions (each reads their own), close | 0:40 |

**Total speaking time (measured):** Abhinav ≈ 4:50 · Harsh ≈ 4:10

The split follows ownership: each of you presents the half you built. That matters for grading, because the contribution statement is checked against who can actually explain what.

**Three handoffs only:** Abhinav → Harsh at slide 6, Harsh → Abhinav at slide 11, then both at slide 13.

---

## Cue sheet — keep this in view while presenting

| # | Slide | Speaker | Length | Clock at end |
|---|---|---|---:|---:|
| 1 | Title | **ABHINAV** | 0:10 | 0:10 |
| 2 | The problem | **ABHINAV** | 0:40 | 0:50 |
| 3 | Functional requirements | **ABHINAV** | 0:35 | 1:25 |
| 4 | Architecture | **ABHINAV** | 0:45 | 2:10 |
| 5 | Recognition | **ABHINAV** | 0:55 | **3:05** ← handoff |
| 6 | Answering and checking | **HARSH** | 0:50 | 3:55 |
| 7 | Real output | **HARSH** | 0:45 | 4:40 |
| 8 | What we did differently (1) | **HARSH** | 0:45 | 5:25 |
| 9 | What we did differently (2) | **HARSH** | 0:45 | 6:10 |
| 10 | Results | **HARSH** | 0:55 | **7:05** ← handoff |
| 11 | Efficiency | **ABHINAV** | 0:45 | 7:50 |
| 12 | Limitations and AI tools | **ABHINAV** | 0:45 | **8:35** |
| 13 | Contributions | **BOTH** | 0:30 | 9:05 |
| 14 | Thank you | **BOTH** | 0:10 | **9:15** |

### Timing checkpoints

| Checkpoint | Target | If you're behind |
|---|---|---|
| Start of slide 6 (handoff to Harsh) | **3:05** | Over 3:30 → Harsh uses the short versions on slides 7 and 9 |
| Start of slide 11 (handoff to Abhinav) | **7:05** | Over 7:30 → Abhinav uses the short versions on slides 11 and 12 |
| Start of slide 13 | **8:35** | Over 9:00 → skip straight to "Thank you" after one line each |

---

# ABHINAV — Slides 1 to 5

---

## Slide 1 — Title

**[ABHINAV]** · 0:10 · clock 0:00 → 0:10

**On screen:** Title, both names.

**Say:**

> "Good morning. I'm Abhinav, and this is Harsh. Our project is *Ask the Sensors* — answering questions about a person's day from phone motion sensors, with evidence behind every answer."

---

## Slide 2 — The problem

**[ABHINAV]** · 0:40 · clock 0:10 → 0:50

**On screen:** The scenario, and the example output from the brief.

**Point at:** The `Timestamp(s)` line in the example, then the orange phrases at the bottom.

**Say:**

> "The scenario: an older woman lives alone. Her son and her physiotherapist want to know how she's doing — using only the motion sensors in her phone. No cameras.
>
> They'll ask things like *'how long was she walking?'*. But notice what the brief actually demands. Not just *'seven hundred seconds'* — *[point]* but **which stretches of signal**, from **which sensors**, and **why** they read as walking.
>
> So this isn't just a classification problem. Every answer has to come with proof."

**If running long:** Don't describe the output block — just point at it and go straight to *"Not just seven hundred seconds…"*

---

## Slide 3 — Functional requirements

**[ABHINAV]** · 0:35 · clock 0:50 → 1:25

**On screen:** The four-tier table, the seven activities, the fixed output format.

**Point at:** Down the "Tier" column, then Tier 4 specifically.

**Say:**

> "The evaluation mixes four tiers of question, without saying which is which.
>
> Tier one is identification — *what is she doing?* Tier two needs numbers across the *whole* recording — *how long, how many times.* Tier three needs the evidence itself: timestamps, sensors, reasoning.
>
> And tier four is the hard one — *[point]* questions about behaviour *outside* the seven activities we trained on, like *'was she fidgeting?'*. Those have to be argued from the signal.
>
> One system, one fixed output format, for all four."

**If running long:** Skip the two boxes at the bottom entirely.

---

## Slide 4 — Architecture

**[ABHINAV]** · 0:45 · clock 1:25 → 2:10

**On screen:** The two-row pipeline diagram.

**Point at:** The blue boxes (top-left), the dashed box in the middle, then the orange boxes, then the `N/A` arrow.

**Say:**

> "Here's the whole system, and it splits cleanly into two halves.
>
> **The blue half is mine.** *[point]* It takes the raw recording, cleans it — fixing units, resampling to exactly twenty-five hertz, cutting four-second windows — and a small neural network labels each window.
>
> **The orange half is Harsh's.** *[point]* It builds a timeline from those labels, matches the question, computes the answer, and a checker verifies the evidence before anything is shown. *[point at N/A]* If the check fails, you get *N/A* with a reason — never a guess.
>
> *[point at dashed box]* The only thing crossing between us is this one file. We froze its format on day one — which let us build both halves in parallel."

---

## Slide 5 — Recognition

**[ABHINAV]** · 0:55 · clock 2:10 → 3:05

**On screen:** The CNN diagram, four bullets on the left, the confusion matrix on the right.

**Point at:** The diagram left to right, then the `9,975 parameters` highlight, then the confusion matrix — specifically the **sitting row, lying column**.

**Say:**

> "The recognition model is deliberately small. A four-second window of six sensor channels goes through three convolution layers, and out come probabilities for seven activities.
>
> **Why so small?** Under ten thousand parameters, forty-eight kilobytes — small enough for a phone, and it compresses cleanly, as I'll show later.
>
> We trained on thirty-six people and validated on twelve *others* — nobody appears in both. Rare activities like running are weighted up.
>
> The result: macro-F1 of **0.289**, beating both baselines — 0.079 for always guessing the commonest activity, and 0.225 for logistic regression.
>
> *[point at matrix]* The biggest error: sitting mistaken for lying — two still postures that differ mainly in the phone's angle."

**Handoff line — say this as you finish:**

> "So that's how we get a label for every window. **Harsh will now show you how the system turns those labels into a grounded answer.**"

---

# ═══════════ HANDOFF → HARSH ═══════════

**Checkpoint: the clock should read about 3:05.**

---

# HARSH — Slides 6 to 10

---

## Slide 6 — Answering and checking

**[HARSH]** · 0:50 · clock 3:05 → 3:55

**On screen:** The blue "rule" box at the top, the eight operations, the "Check" list.

**Point at:** The word **never** in the top box, then down the operations table, then the four check bullets.

**Say:**

> "Thanks, Abhinav. Everything on my side follows one rule. *[point]* A language model **never** produces a number, an interval, or a verdict. Every quantity is computed in ordinary code, so it can be checked.
>
> Answering has three steps. **First, match** — keyword rules map the question to one of eight operations: identify, verify, duration, count, and so on.
>
> **Second, compute** — the operation reads the timeline and calculates the answer *and* the intervals behind it.
>
> **Third, check** — before anything is shown. Every cited interval must exist. The printed times must match. Durations and counts must add up again from the evidence. If any check fails, the answer is withheld and replaced by *N/A* with the reason. There is no bypass."

---

## Slide 7 — Real output

**[HARSH]** · 0:45 · clock 3:55 → 4:40

**On screen:** Four real system outputs, one per tier.

**Point at:** Tier 2's `Timestamp(s)` and explanation, then Tier 3's explanation, then Tier 4 in orange.

**Say:**

> "This is verbatim output from our system — one answer per tier.
>
> *[point at Tier 2]* For duration, it doesn't just say *seven hundred twenty seconds*. It cites the two intervals and shows they sum to that — and quotes the step rhythm it measured.
>
> *[point at Tier 3]* For onset, the explanation quotes the actual signal changing: acceleration variability jumping from 0.08 to 1.55 right at the transition.
>
> *[point at Tier 4]* And for *'was she driving?'* — we have no way to measure that. So instead of guessing the nearest activity, it says *N/A* and explains why. That refusal is a feature, not a failure."

**If running long:** Skip Tier 3. Do Tier 2, then Tier 4.

---

## Slide 8 — What we did differently (1)

**[HARSH]** · 0:45 · clock 4:40 → 5:25

**On screen:** "The usual way" vs "Ours" on the left; the bursts-to-timeline diagram on the right.

**Point at:** *"nothing checks it"* in orange, then the checker box, then the timeline diagram.

**Say:**

> "Two things we did differently.
>
> **First, grounding by construction.** *[point]* The usual way, a question goes into a language model and an answer comes out — nothing checks it. In ours, every number and interval is recomputed before release. An ungrounded answer isn't just unlikely — it's **structurally impossible**.
>
> **Second, a timeline that fits the data.** *[point at diagram]* The phone only records about twenty seconds of every minute. If you simply added up windows, a thirty-minute walk would come out as ten. So each burst stands for its whole minute, and gaps stay gaps. Handling the phone's clock drift turned three thousand broken fragments into three hundred and twenty-five real bouts."

---

## Slide 9 — What we did differently (2)

**[HARSH]** · 0:45 · clock 5:25 → 6:10

**On screen:** The two-track diagram on the left; the decision diamond on the right; a sentence at the bottom.

**Point at:** The two tracks, then the diamond, then the bottom sentence.

**Say:**

> "**Third — we can prove where every mistake comes from.** *[point]* Each question is answered twice, by identical code: once from true labels, once from the model's. Right with true labels but wrong with the model's means the fault *must* be recognition. That's how we know **all nineteen** misses are recognition errors — by construction, not by guessing.
>
> **Fourth, untrained behaviours.** *[point at diamond]* For *'was she resting?'* we use stillness, calibrated from real data. Where there's no measurable signature — like driving — the system declines rather than guesses.
>
> *[point at bottom]* And we built a small language model for question matching, measured it losing to keyword rules, and reported that honestly."

**If running long:** Drop the final sentence about the language model.

---

## Slide 10 — Results

**[HARSH]** · 0:55 · clock 6:10 → 7:05

**On screen:** Figure 1 — accuracy by question type, blue vs orange bars — and four bullets.

**Point at:** The tall blue bars, then the orange bars on duration and count, then the "overall" pair on the far right.

**Say:**

> "Here are the results. Blue is the system answering from true labels, orange is from our trained model.
>
> *[point at blue]* With true labels, the answering layer gets **eighty-seven out of eighty-seven** questions right, across all our test sets. It makes no errors of its own.
>
> *[point at orange]* With the model's labels, it's **fifteen out of thirty-four** on real users — thirty-nine percent. And every one of those nineteen misses is a recognition error.
>
> *[point at duration and count]* Durations and counts collapse completely, because they add up the model's per-minute mistakes over tens of hours.
>
> Even so, when we deliberately mislabelled ten percent of minutes, the score still held at eighty-four percent.
>
> So the answering layer is not the bottleneck — recognition is."

**Handoff line — say this as you finish:**

> "**Abhinav will now cover efficiency and our limitations.**"

---

# ═══════════ HANDOFF → ABHINAV ═══════════

**Checkpoint: the clock should read about 7:05.**

---

# ABHINAV — Slides 11 and 12

---

## Slide 11 — Efficiency

**[ABHINAV]** · 0:45 · clock 7:05 → 7:50

**On screen:** Figure 4 — accuracy against model size, with the frontier line — and four bullets.

**Point at:** The point labelled `full` on the right, then `quant8` at the top, then the robustness bullet.

**Say:**

> "Thanks, Harsh. On our target laptop, the model takes **0.4 milliseconds** per window, and answering a question takes **sixty-seven milliseconds**.
>
> We made three smaller versions — with no retraining. One stores weights as 8-bit integers; two remove thirty and sixty percent of the network's filters.
>
> *[point at full, then quant8]* The 8-bit version is **twenty-seven percent smaller with no loss in accuracy**. It actually scored slightly higher — but on thirty-four questions, that's noise. And honestly, all versions run at about the same speed: at this size, compression buys space, not speed.
>
> For robustness, we dropped half of all the raw sensor samples — and accuracy barely moved, from 0.393 to 0.375."

**If running long:** Drop the sentence "It actually scored slightly higher…"

---

## Slide 12 — Limitations and AI tools

**[ABHINAV]** · 0:45 · clock 7:50 → 8:35

**On screen:** Limitations on the left; use of AI tools on the right.

**Point at:** The first limitation, then the right-hand column.

**Say:**

> "Our limitations, stated plainly.
>
> *[point]* Recognition caps everything — even with an answering layer that made no errors, the model holds us to fifteen of thirty-four. Our real test users were also in the training group, so new users would likely do worse.
>
> *[point at right]* On AI tools: inside the system, a small Qwen model is an optional question parser, and Claude Haiku scored explanations during evaluation — neither ever produces an answer. During development, Claude Code was our AI coding assistant — helping write and debug code, and understand concepts. We made the design decisions, and verified every number by running our own code."

**If running long:** Say only the first limitation, then go straight to AI tools.

---

# ═══════════ BOTH ═══════════

**Checkpoint: the clock should read about 8:35.**

---

## Slide 13 — Contributions

**[BOTH]** · 0:30 · clock 8:35 → 9:05

**On screen:** Two columns, one per person.

**Each person points at their own column while speaking.**

**[ABHINAV]** — about 15 seconds:

> "I built the signal half: downloading and parsing the data, detecting each phone's units, resampling, windowing, and the subject splits — then the recognition model, cost measurement, compression and robustness testing."

**[HARSH]** — about 15 seconds:

> "I built the answering half: the timeline, question matching and the eight operations, the open-ended answers, the checker — and the whole evaluation, including tracing every error to its source."

*(The slide's bottom line — "both halves meet at one frozen file" — was already said on slide 4. Don't repeat it.)*

---

## Slide 14 — Thank you

**[BOTH]** · 0:10 · clock 9:05 → 9:15

**[HARSH]:**

> "Thank you."

**[ABHINAV]:**

> "We're happy to take questions."

---

# Q&A — who answers what

Not part of the 10 minutes, but decide this in advance so you don't talk over each other. **Default rule: whoever built it answers it.**

| If asked about… | Answers | Key facts |
|---|---|---|
| Data, units, 25 Hz, windows, splits | **ABHINAV** | 59 of 60 subjects fixed by per-subject unit detection; gaps over 1 s marked missing; split by person |
| The CNN, accuracy, macro-F1 | **ABHINAV** | 9,975 params; macro-F1 0.289; macro recall 0.330; sitting→lying 43.8% |
| Why a CNN | **ABHINAV** | Chosen for quantization, pruning and distillation |
| Compression, cost, energy, robustness | **ABHINAV** | 8-bit 27% smaller; energy is an estimate from rated power |
| The timeline, operations, checker | **HARSH** | Minute-attributed bouts; 8 operations; no bypass |
| How correctness is scored | **HARSH** | IoU ≥ 0.5; duration within max(4 s, 10%); count ±1 — all pre-registered |
| Error tracing, the 19 misses | **HARSH** | Identical code on true vs model labels |
| Open-world, stillness | **HARSH** | Calibrated from lying-down windows; declines when no signature |
| Language model vs rules | **HARSH** | Rules 69% vs SLM 46%; 67 ms vs 11.4 s |
| "39% isn't very good, is it?" | **HARSH** starts, **ABHINAV** adds | Answering is 87/87 on true labels; recognition is the bottleneck, proven by construction |
| AI tool usage | **either** | Three uses: in the product, in evaluation, in development |

---

# Rehearsal notes for a strict 10 minutes

1. **Rehearse with a stopwatch, out loud, at least twice.** Reading silently is about 30% faster than speaking — a script that "feels" like 8 minutes in your head runs 10+ out loud.
2. **The script is 1,265 spoken words — 9:02 at 140 words per minute.** Nervous speakers often speed up, but careful ones slow down. If your first timed run comes in over 9:40, use the "if running long" cuts from the start, not just in emergencies.
3. **The handoff is the most common place to lose time.** Practise the two handoff lines word for word, and have the next speaker start talking the moment the line ends — no "um, okay, so…".
4. **Point, don't read.** Slides 2, 6 and 7 are dense. The audience can read; your job is to tell them *where to look* and *why it matters*.
5. **Don't read numbers robotically.** Say "under ten thousand parameters," not "nine thousand nine hundred and seventy-five." Keep exact numbers for the ones that matter: 0.289, 87 of 87, 15 of 34, 19 misses, 27%.
6. **Whoever isn't speaking stands still and watches the slide** — it directs the audience's attention to it.
7. **The non-speaker keeps time.** Agree on a silent signal (e.g. a small hand gesture) for "we're 30 seconds behind — cut."

---

# Things to fix in the deck before presenting

1. **Wrong cross-reference on slide 5.** It says the model "compresses cleanly **(slide 9)**", but efficiency is **slide 11**. If you present from the images as-is, don't say "slide 9" aloud — the script says "which I'll show you shortly" instead. Worth correcting in `slides.tex`.
2. **Three slides overflow their frame** (6, 7 and 12). On slide 7, the last line of the Tier 2 block collides with the footnote at the bottom. Legible, but untidy on a projector.

---

**One wording choice to confirm:** on slide 12, the spoken line says Claude Code was used for *"helping write and debug code, and to understand concepts and documentation."* That's a little fuller than the slide text, which omits *write*. It's deliberate — a coding assistant writing code is the accurate description, and it's safer for the spoken disclosure to be complete than understated. Agree the exact phrasing with Harsh before you present, so the two of you say the same thing.
