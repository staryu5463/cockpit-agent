# Uncertainty for `success` (300 tasks per split, same tasks for every method)

## Rates with 95% Wilson intervals

| Method | test_id | test_expr | test_entity | test_combo |
|---|---:|---:|---:|---:|
| rule | 92.3 [88.8, 94.8] | 24.7 [20.1, 29.8] | 88.7 [84.6, 91.8] | 84.7 [80.2, 88.3] |
| sft | 94.0 [90.7, 96.2] | 80.7 [75.8, 84.7] | 89.0 [85.0, 92.1] | 80.3 [75.5, 84.4] |
| dpo | 94.7 [91.5, 96.7] | 80.0 [75.1, 84.1] | 90.7 [86.8, 93.5] | 81.0 [76.2, 85.0] |
| grpo | 95.0 [91.9, 96.9] | 83.3 [78.7, 87.1] | 93.0 [89.5, 95.4] | 86.3 [82.0, 89.8] |

## Paired differences (A − B), percentage points

| A vs B | split | diff | 95% bootstrap CI | only A | only B | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| grpo vs sft | test_id | +1.0 | [-1.0, +3.3] | 7 | 4 | 0.549 |
| grpo vs sft | test_expr | +2.7 | [+0.3, +5.3] | 12 | 4 | 0.0768 |
| grpo vs sft | test_entity | +4.0 | [+1.0, +7.3] | 18 | 6 | 0.0227 |
| grpo vs sft | test_combo | +6.0 | [+2.0, +10.0] | 27 | 9 | 0.00393 |
| grpo vs dpo | test_id | +0.3 | [-1.7, +2.3] | 5 | 4 | 1 |
| grpo vs dpo | test_expr | +3.3 | [+0.7, +6.0] | 14 | 4 | 0.0309 |
| grpo vs dpo | test_entity | +2.3 | [-0.3, +5.3] | 13 | 6 | 0.167 |
| grpo vs dpo | test_combo | +5.3 | [+1.7, +9.0] | 25 | 9 | 0.00904 |
| grpo vs rule | test_id | +2.7 | [-1.0, +6.7] | 22 | 14 | 0.243 |
| grpo vs rule | test_expr | +58.7 | [+53.0, +64.3] | 177 | 1 | 9.34e-52 |
| grpo vs rule | test_entity | +4.3 | [+0.3, +8.7] | 28 | 15 | 0.066 |
| grpo vs rule | test_combo | +1.7 | [-4.3, +7.7] | 45 | 40 | 0.665 |
| dpo vs sft | test_id | +0.7 | [-1.0, +2.3] | 4 | 2 | 0.688 |
| dpo vs sft | test_expr | -0.7 | [-2.0, +0.7] | 1 | 3 | 0.625 |
| dpo vs sft | test_entity | +1.7 | [+0.0, +3.3] | 6 | 1 | 0.125 |
| dpo vs sft | test_combo | +0.7 | [-1.7, +3.0] | 7 | 5 | 0.774 |
| sft vs rule | test_id | +1.7 | [-2.0, +5.7] | 20 | 15 | 0.5 |
| sft vs rule | test_expr | +56.0 | [+50.3, +61.7] | 169 | 1 | 2.29e-49 |
| sft vs rule | test_entity | +0.3 | [-4.7, +5.3] | 29 | 28 | 1 |
| sft vs rule | test_combo | -4.3 | [-10.3, +2.0] | 38 | 51 | 0.203 |
