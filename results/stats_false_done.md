# Uncertainty for `false_done` (300 tasks per split, same tasks for every method)

## Rates with 95% Wilson intervals

| Method | test_id | test_expr | test_entity | test_combo |
|---|---:|---:|---:|---:|
| rule | 5.7 [3.6, 8.9] | 45.3 [39.8, 51.0] | 8.7 [6.0, 12.4] | 12.3 [9.1, 16.5] |
| sft | 3.7 [2.1, 6.4] | 17.0 [13.2, 21.7] | 10.7 [7.7, 14.7] | 16.3 [12.6, 20.9] |
| dpo | 3.0 [1.6, 5.6] | 18.3 [14.4, 23.1] | 8.7 [6.0, 12.4] | 16.7 [12.9, 21.3] |
| grpo | 3.0 [1.6, 5.6] | 15.7 [12.0, 20.2] | 6.0 [3.8, 9.3] | 13.0 [9.7, 17.3] |

## Paired differences (A − B), percentage points

| A vs B | split | diff | 95% bootstrap CI | only A | only B | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| grpo vs sft | test_id | -0.7 | [-2.7, +1.3] | 4 | 6 | 0.754 |
| grpo vs sft | test_expr | -1.3 | [-3.7, +1.0] | 4 | 8 | 0.388 |
| grpo vs sft | test_entity | -4.7 | [-8.0, -1.7] | 5 | 19 | 0.00661 |
| grpo vs sft | test_combo | -3.3 | [-7.0, +0.0] | 10 | 20 | 0.0987 |
| grpo vs dpo | test_id | +0.0 | [-1.7, +2.0] | 4 | 4 | 1 |
| grpo vs dpo | test_expr | -2.7 | [-5.0, -0.3] | 3 | 11 | 0.0574 |
| grpo vs dpo | test_entity | -2.7 | [-5.7, +0.0] | 5 | 13 | 0.0963 |
| grpo vs dpo | test_combo | -3.7 | [-7.3, -0.3] | 9 | 20 | 0.0614 |
| grpo vs rule | test_id | -2.7 | [-6.0, +0.7] | 8 | 16 | 0.152 |
| grpo vs rule | test_expr | -29.7 | [-35.3, -24.3] | 4 | 93 | 4.57e-23 |
| grpo vs rule | test_entity | -2.7 | [-6.7, +1.0] | 14 | 22 | 0.243 |
| grpo vs rule | test_combo | +0.7 | [-5.0, +6.3] | 38 | 36 | 0.908 |
| dpo vs sft | test_id | -0.7 | [-2.3, +1.0] | 2 | 4 | 0.688 |
| dpo vs sft | test_expr | +1.3 | [+0.3, +2.7] | 4 | 0 | 0.125 |
| dpo vs sft | test_entity | -2.0 | [-3.7, -0.7] | 0 | 6 | 0.0312 |
| dpo vs sft | test_combo | +0.3 | [-1.7, +2.7] | 6 | 5 | 1 |
| sft vs rule | test_id | -2.0 | [-5.0, +1.0] | 8 | 14 | 0.286 |
| sft vs rule | test_expr | -28.3 | [-34.0, -23.0] | 5 | 90 | 3.09e-21 |
| sft vs rule | test_entity | +2.0 | [-2.7, +6.7] | 29 | 23 | 0.488 |
| sft vs rule | test_combo | +4.0 | [-1.7, +9.7] | 43 | 31 | 0.201 |
