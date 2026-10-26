# MoRCA results

Synthetic arithmetic; prospective routed rollouts with calibrated event-position gates. Incomplete proposal coverage; see README.
Intervals cluster by task, preserving paired conditions and routing seeds. Modes analyzed separately.

| Mode | Policy | Accuracy | Paired accuracy change vs none (95% CI) | Mean reference log score | Mean tokens |
|---|---|---:|---:|---:|---:|
| nonthinking | none | 0.910 | +0.000 [+0.000, +0.000] | -0.576 | 61.9 |
| nonthinking | morca | 0.938 | +0.027 [+0.008, +0.047] | -0.475 | 58.4 |
| nonthinking | random_budget | 0.926 | +0.016 [+0.000, +0.035] | -0.507 | 58.9 |

Primary comparison: morca vs none in mutual_correction. n=64, wins=7, losses=0, exact two-sided p=0.015625, complete=True.
