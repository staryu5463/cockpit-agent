# Efficiency experiments (300 tasks per split, same tasks for every run, one seed)

## test_id

| Run | Setup | Success | Turns | Tool calls | Generated tokens | False completion |
|---|---|---:|---:|---:|---:|---:|
| grpo_v2 | JSON, one call per turn, SFT -> GRPO (main result) | 95.0% | 3.28 | 3.28 | — | 3.0% |
| sft_fn_par | compact fn style, 30% parallel SFT | 95.3% | 3.28 | 3.28 | 49.7 | 3.7% |
| grpo_fn_base |   -> GRPO | 95.0% | 3.29 | 3.29 | 49.8 | 5.0% |
| grpo_fn_eff |   -> GRPO + turn penalty (applied to every episode) | 94.7% | 2.56 | 3.31 | 49.4 | 5.3% |
| sft_js_par | JSON, 30% parallel SFT | 91.7% | 3.23 | 3.24 | 84.4 | 4.7% |
| grpo_js_base |   -> GRPO | 96.0% | 3.33 | 3.33 | 86.8 | 3.3% |
| grpo_js_eff |   -> GRPO + turn penalty (successful episodes only) | 94.7% | 2.63 | 3.36 | 86.6 | 4.0% |

| A vs B | Success diff | 95% bootstrap CI | only A | only B | McNemar p | Turns A / B |
|---|---:|---:|---:|---:|---:|---:|
| grpo_js_eff vs grpo_js_base | -1.3 | [-4.0, +1.3] | 6 | 10 | 0.454 | 2.63 / 3.33 (-21%) |
| grpo_js_base vs grpo_v2 | +1.0 | [-1.3, +3.3] | 9 | 6 | 0.607 | 3.33 / 3.28 (+2%) |
| grpo_js_base vs sft_js_par | +4.3 | [+1.0, +7.7] | 20 | 7 | 0.0192 | 3.33 / 3.23 (+3%) |
| grpo_fn_base vs grpo_v2 | +0.0 | [-3.0, +3.0] | 10 | 10 | 1 | 3.29 / 3.28 (+0%) |

## test_expr

| Run | Setup | Success | Turns | Tool calls | Generated tokens | False completion |
|---|---|---:|---:|---:|---:|---:|
| grpo_v2 | JSON, one call per turn, SFT -> GRPO (main result) | 83.3% | 3.41 | 3.41 | — | 15.7% |
| sft_fn_par | compact fn style, 30% parallel SFT | 71.0% | 3.45 | 3.45 | 53.0 | 24.3% |
| grpo_fn_base |   -> GRPO | 69.7% | 3.41 | 3.41 | 52.0 | 29.3% |
| grpo_fn_eff |   -> GRPO + turn penalty (applied to every episode) | 70.7% | 2.73 | 3.52 | 53.0 | 28.0% |
| sft_js_par | JSON, 30% parallel SFT | 84.0% | 3.19 | 3.19 | 82.7 | 14.0% |
| grpo_js_base |   -> GRPO | 88.3% | 3.26 | 3.26 | 84.2 | 11.3% |
| grpo_js_eff |   -> GRPO + turn penalty (successful episodes only) | 86.3% | 2.67 | 3.33 | 84.7 | 13.7% |

| A vs B | Success diff | 95% bootstrap CI | only A | only B | McNemar p | Turns A / B |
|---|---:|---:|---:|---:|---:|---:|
| grpo_js_eff vs grpo_js_base | -2.0 | [-4.7, +0.7] | 6 | 12 | 0.238 | 2.67 / 3.26 (-18%) |
| grpo_js_base vs grpo_v2 | +5.0 | [+1.3, +8.7] | 24 | 9 | 0.0135 | 3.26 / 3.41 (-4%) |
| grpo_js_base vs sft_js_par | +4.3 | [+1.3, +7.3] | 17 | 4 | 0.0072 | 3.26 / 3.19 (+2%) |
| grpo_fn_base vs grpo_v2 | -13.7 | [-18.3, -9.0] | 9 | 50 | 5.26e-08 | 3.41 / 3.41 (+0%) |

## test_entity

| Run | Setup | Success | Turns | Tool calls | Generated tokens | False completion |
|---|---|---:|---:|---:|---:|---:|
| grpo_v2 | JSON, one call per turn, SFT -> GRPO (main result) | 93.0% | 4.40 | 4.40 | — | 6.0% |
| sft_fn_par | compact fn style, 30% parallel SFT | 89.0% | 4.37 | 4.38 | 66.4 | 11.0% |
| grpo_fn_base |   -> GRPO | 95.3% | 4.40 | 4.40 | 66.7 | 4.7% |
| grpo_fn_eff |   -> GRPO + turn penalty (applied to every episode) | 89.3% | 3.14 | 4.40 | 65.5 | 10.7% |
| sft_js_par | JSON, 30% parallel SFT | 89.3% | 4.36 | 4.36 | 112.2 | 9.7% |
| grpo_js_base |   -> GRPO | 91.3% | 4.39 | 4.39 | 113.6 | 7.7% |
| grpo_js_eff |   -> GRPO + turn penalty (successful episodes only) | 90.0% | 3.19 | 4.43 | 112.6 | 9.3% |

| A vs B | Success diff | 95% bootstrap CI | only A | only B | McNemar p | Turns A / B |
|---|---:|---:|---:|---:|---:|---:|
| grpo_js_eff vs grpo_js_base | -1.3 | [-4.7, +2.0] | 10 | 14 | 0.541 | 3.19 / 4.39 (-27%) |
| grpo_js_base vs grpo_v2 | -1.7 | [-4.7, +1.0] | 7 | 12 | 0.359 | 4.39 / 4.40 (-0%) |
| grpo_js_base vs sft_js_par | +2.0 | [-1.0, +5.0] | 13 | 7 | 0.263 | 4.39 / 4.36 (+1%) |
| grpo_fn_base vs grpo_v2 | +2.3 | [-0.3, +5.0] | 11 | 4 | 0.118 | 4.40 / 4.40 (+0%) |

## test_combo

| Run | Setup | Success | Turns | Tool calls | Generated tokens | False completion |
|---|---|---:|---:|---:|---:|---:|
| grpo_v2 | JSON, one call per turn, SFT -> GRPO (main result) | 86.3% | 6.17 | 6.17 | — | 13.0% |
| sft_fn_par | compact fn style, 30% parallel SFT | 79.7% | 6.13 | 6.13 | 92.1 | 20.3% |
| grpo_fn_base |   -> GRPO | 85.3% | 6.14 | 6.14 | 92.4 | 14.7% |
| grpo_fn_eff |   -> GRPO + turn penalty (applied to every episode) | 71.7% | 3.19 | 6.12 | 89.5 | 28.3% |
| sft_js_par | JSON, 30% parallel SFT | 77.7% | 6.08 | 6.08 | 156.1 | 20.0% |
| grpo_js_base |   -> GRPO | 85.3% | 6.17 | 6.17 | 159.2 | 14.3% |
| grpo_js_eff |   -> GRPO + turn penalty (successful episodes only) | 77.7% | 3.26 | 6.18 | 155.9 | 21.7% |

| A vs B | Success diff | 95% bootstrap CI | only A | only B | McNemar p | Turns A / B |
|---|---:|---:|---:|---:|---:|---:|
| grpo_js_eff vs grpo_js_base | -7.7 | [-12.0, -3.3] | 11 | 34 | 0.000824 | 3.26 / 6.17 (-47%) |
| grpo_js_base vs grpo_v2 | -1.0 | [-4.3, +2.3] | 11 | 14 | 0.69 | 6.17 / 6.17 (+0%) |
| grpo_js_base vs sft_js_par | +7.7 | [+3.7, +11.7] | 32 | 9 | 0.000431 | 6.17 / 6.08 (+1%) |
| grpo_fn_base vs grpo_v2 | -1.0 | [-4.0, +2.0] | 10 | 13 | 0.678 | 6.14 / 6.17 (-1%) |
