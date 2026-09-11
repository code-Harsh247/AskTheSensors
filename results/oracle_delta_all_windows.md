# Oracle vs real delta

- Real track: tests\fixtures\real_model_tracks
- Oracle track: tests\fixtures
- Questions: data\questions_dev_v2
- Subjects: subj_real_a, subj_real_b
- Compared over: every recorded window, including minutes nobody labelled

"Right" means correct, and grounded wherever the gold cites evidence (PRD 7.3.4).

| Question type | n | Right (oracle) | Right (real) | Lost: recognition (A) | Lost: routing (B) | Lost: reasoning (B) | Masked bugs |
|---|---|---|---|---|---|---|---|
| identification | 4 | 4 | 2 | 2 | 0 | 0 | 0 |
| verification | 4 | 4 | 4 | 0 | 0 | 0 | 0 |
| duration | 5 | 5 | 0 | 5 | 0 | 0 | 0 |
| count | 2 | 2 | 0 | 2 | 0 | 0 | 0 |
| comparison | 3 | 3 | 0 | 3 | 0 | 0 | 0 |
| grounding | 8 | 8 | 3 | 5 | 0 | 0 | 0 |
| open_world | 8 | 8 | 6 | 2 | 0 | 0 | 0 |
| **overall** | 34 | 34 | 15 | 19 | 0 | 0 | 0 |

Answers withheld by the grounding validator: 0 on the oracle track, 0 on the real track.

## Fix list - Member A (recognition)

- `subj_real_a_t1_id0` (identification, operator `identify`, recognition): gold "Sitting"; oracle "Sitting"; real "Lying down"
- `subj_real_a_t1_id1` (identification, operator `identify`, recognition): gold "Walking"; oracle "Walking"; real "Sitting"
- `subj_real_a_t2_dur0` (duration, operator `duration`, recognition): gold "66386 seconds"; oracle "66386 seconds"; real "29696 seconds"
- `subj_real_a_t2_dur1` (duration, operator `duration`, recognition): gold "109 seconds"; oracle "109 seconds"; real "42267 seconds"
- `subj_real_a_t2_dur_absent` (duration, operator `duration`, recognition): gold "0 seconds"; oracle "0 seconds"; real "4859 seconds"
- `subj_real_a_t2_count0` (count, operator `count`, recognition): gold "15"; oracle "15 separate bouts"; real "57 separate bouts"
- `subj_real_a_t2_cmp0` (comparison, operator `compare`, recognition): gold "Sitting"; oracle "Sitting"; real "Lying down"
- `subj_real_a_t3_ground0` (grounding, operator `ground`, recognition): gold "Standing in place"; oracle "Standing in place"; real "Standing in place"
- `subj_real_a_t3_ground2` (grounding, operator `ground`, recognition): gold "Standing and moving"; oracle "Standing and moving"; real "Standing and moving"
- `subj_real_a_t3_onset` (grounding, operator `onset`, recognition): gold "Yes, walking began at 9356 seconds"; oracle "Yes, walking began at 9356 seconds"; real "Yes, walking began at 1411 seconds"
- `subj_real_a_t4_rest` (open_world, operator `open_world`, recognition): gold "Likely yes"; oracle "Likely yes"; real "Likely yes"
- `subj_real_a_t4_wheeled` (open_world, operator `open_world`, recognition): gold "No"; oracle "No"; real "Yes"
- `subj_real_b_t2_dur0` (duration, operator `duration`, recognition): gold "167675 seconds"; oracle "167675 seconds"; real "22147 seconds"
- `subj_real_b_t2_dur1` (duration, operator `duration`, recognition): gold "1084 seconds"; oracle "1084 seconds"; real "14568 seconds"
- `subj_real_b_t2_count0` (count, operator `count`, recognition): gold "5"; oracle "5 separate bouts"; real "48 separate bouts"
- `subj_real_b_t2_cmp0` (comparison, operator `compare`, recognition): gold "Running"; oracle "Running"; real "Walking"
- `subj_real_b_t2_cmp1` (comparison, operator `compare`, recognition): gold "Sitting"; oracle "Sitting"; real "Lying down"
- `subj_real_b_t3_ground1` (grounding, operator `ground`, recognition): gold "Walking"; oracle "Walking"; real "Walking"
- `subj_real_b_t3_ground2` (grounding, operator `ground`, recognition): gold "Standing in place"; oracle "Standing in place"; real "Standing in place"

## Fix list - Member B (routing and reasoning)

None.

