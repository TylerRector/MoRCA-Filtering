# MoRCA

Qwen3-0.6B message-routing pilot. Python 3.14.

```sh
pip install -r requirements.txt
python experiment.py prepare
python experiment.py run --approve-run
python -m unittest discover -s tests
```

New runs write to `runs/pilot`. The recorded laptop run took about five hours.

[Results](results-laptop-3h/analysis.md) · [Design and mathematics](results-laptop-3h/EXPERIMENT_DESIGN_AND_MATH.md)

Seeded-error accuracy: 75.0% → 85.9%, n=64, paired p=0.015625. Echo fidelity stayed zero; the gain came from the harmful-marginal gate.

Lean proofs are a separate project. Original run hashes are retained; packaged source has comments and docstrings removed.
