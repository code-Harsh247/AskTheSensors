**Figure 1: Accuracy by question type.** 34 questions over subj_real_a, subj_real_b (`data/questions_dev_v2`), answered from oracle labels (blue) and from the trained model's predictions (orange); the same reasoning layer answers both. Overall is the macro-average of the per-type accuracies, so plentiful easy types cannot dominate it (PRD 7.3). Correctness rule by group:

- **identification**: exact match of the named activity.
- **verification**: exact match of the yes/no verdict.
- **duration**: within max(4 s, 10%) of the true total (pre-registered).
- **count**: within 1 bout of the true count.
- **comparison**: exact match of the activity that took longer, or 'Equal'.
- **grounding**: cited intervals reach matched IoU >= 0.5 with the true intervals (pre-registered).
- **open-world**: exact match of the verdict to the reference behaviour.

Produced by `scripts/make_fig1.py`; numbers in `results/fig1_accuracy_by_question_type.csv`.
