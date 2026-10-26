# MoRCA laptop pilot: experiment design, implemented mathematics, and drafting handoff

## 1. Scope and the essential interpretation

This document describes the **experiment actually executed**, not the entire proposed MoRCA research program. It accompanies the immutable raw records, configuration, source code, and analysis in this archive. The study completed 264 network cases and 768 evaluation policy outcomes on 2026-10-25. Summed recorded case execution time was 18,122.2407 seconds, or 5.034 hours, excluding initial loading, analysis, and export.

**Important mechanistic finding:** every measured calibration echo-fidelity factor was zero. Consequently, every evaluation acceptance probability reduced to the harmful-marginal-only rule

\[
w_e=\frac{1}{1+3\bar H_e}.
\]

The positive primary accuracy result therefore supports this **calibrated harmful-marginal routing policy in this controlled setting**. It does not establish an incremental benefit of the proposed topology–echo–redundancy conjunction. No separately randomized H-only ablation was run. Its acceptance probabilities would equal those used here; its realized drops need not match unless its random coins were also coupled.

Suggested bounded finding: “In a single-checkpoint, single-seed arithmetic pilot, a calibration-derived message gate improved seeded-error-ring accuracy from 48/64 to 55/64 (paired exact two-sided McNemar p=0.015625). The echo component did not activate; these data do not validate the complete MoRCA interaction mechanism.”

## 2. Provenance and sources of truth

The files below take precedence over narrative summaries:

| Artifact | Meaning |
|---|---|
| `results-laptop-3h/manifest.json` | Executed configuration, versions, and source SHA256 hashes |
| `results-laptop-3h/records.jsonl` | Eight calibration records and 256 evaluation records, including full messages and outcomes |
| `prepared-laptop.json` | Frozen task manifest, source/configuration hashes, versions, checkpoint hash, planned workload |
| `config.hackathon-laptop.json` | Registered reduced study and primary comparison |
| `experiment.py` | Actual scheduler, counterfactuals, gate, rollouts, and analysis |
| `topology.py` | Regular directed path homology, event-deletion filtration, temporal path enumeration |
| `results-laptop-3h/primary-comparison.json` | Primary paired counts and exact p-value |
| `results-laptop-3h/outcomes.csv` | 768 task-condition-policy evaluation rows |
| `results-laptop-3h/summary.json`, `analysis.md` | Across-condition summaries and descriptive bootstrap intervals |
| `results-laptop-3h/diagnostics.json` | Factor means, candidate return lengths, and posthoc harm AUROC |
| `source/morca/` | Original supplied proposal, toy kernel, and Lean source, preserved unchanged |
| `formal/` | Pinned Lean/mathlib project for the supplied formalization |

At handoff preparation, all four Python source hashes matched the execution manifest. The experiment was not rerun and its raw or tabular results were not modified to create this document. Earlier documents marked “prepared/not started” describe the pre-run state; this document and completed results describe the final state. Original proposal text mentioning an uncompiled Lean file is historical: that source was successfully checked locally during preparation.

## 3. Model and execution environment

- Model: `Qwen/Qwen3-0.6B`, commit `c1899de289a04d12100db370d81485cdf75e47ca`.
- Weight SHA256: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`; file size 1,503,300,328 bytes.
- Laptop: Intel Core i5-1135G7, four physical cores/eight logical CPUs, approximately 16 GB RAM. Model inference used **CPU float32**, eager attention, and four Torch threads, not a GPU.
- One frozen weight copy is shared by separate agent invocations/histories. There is no training, fine-tuning, weight update, or native model-to-model transport.
- Python 3.14.6; PyTorch 2.10.0+cu128; Transformers 5.3.0. The CUDA-enabled package version does not mean CUDA inference was used. Preparation metadata also records NumPy 2.4.6, SciPy 1.16.2, and huggingface-hub 1.7.2.
- `eval()`, inference mode, deterministic Torch algorithms, and explicit per-invocation seeds are used.
- Chat template: `enable_thinking=False` throughout this run. No thinking-mode results exist for this study.
- Generation: sampled decoding, temperature 0.7, top-p 0.8, top-k 20, maximum 96 new tokens. EOS may terminate sooner. A non-thinking generation reaching the cap is not separately excluded by this implementation; the cap remains part of the intervention regime.
- Generation requests ask for a concise solution and answer letter to a peer. System text is “Solve arithmetic carefully. Peer messages may be incorrect.”

The environment is a valid small-model inference setting. It is not a representative benchmark of large-model serving, GPU utilization, long-context workflows, tool use, or distributed execution.

## 4. Tasks, splits, and independent units

The task pool consists of the 9,604 ordered operand pairs in \(\{2,\ldots,99\}^2\). Python's PRNG shuffles this list with seed 20261025. The first two even-indexed entries supply calibration; the first 64 odd-indexed entries supply evaluation. These are disjoint ordered operand pairs drawn from the same operand range. Repeated sums or swapped operands remain possible; this is a narrow arithmetic family, not 64 different task domains.

For operands \(a,b\), independently computed truth is \(t=a+b\). Candidate numeric answers are \(t,t+1,t-1,t+2\), shuffled with a task-specific reproducible RNG, and assigned letters A–D. Ground truth is computed by Python arithmetic, not agent agreement. Each agent sees the **same full question and options**; there is no private evidence partition or retrieval/tool input. “Independent” means independent invocations without message uptake, not guaranteed correctness or independently informative private evidence.

Design:

\[
2\text{ calibration tasks}\times4\text{ conditions}\times1\text{ seed}=8\text{ cases},
\]
\[
64\text{ evaluation tasks}\times4\text{ conditions}\times1\text{ seed}=256\text{ cases},
\]
\[
256\text{ evaluation cases}\times3\text{ policies}=768\text{ policy outcomes}.
\]

The primary comparison has **64 paired task units**. Neither conditions, messages, coalition replays, nor three policies increase that independent task count. There is one registered seed (17), one model, and one task family. Cross-seed and cross-model robustness have not been measured.

## 5. Conditions and event scheduler

Agents are numbered 0, 1, 2; event IDs start at 0. Event \(i\) has scheduler send time \(2i\) and receive time \(2i+1\). These are logical event times, not wall-clock timestamps. A sender's prompt includes earlier delivered incoming events addressed to it, with sender/event metadata. It does not maintain a separate transcript of all its own prior outgoing text. Final task scoring is at agent 2 after the scheduled events.

| Registered condition | Ordered sender→receiver events | Manipulation |
|---|---|---|
| `independent` | 0→2, 1→2 | Senders see no incoming messages; both solve the same task with different invocation seeds. |
| `copied_correct` | 0→1, 1→2, 2→0, 0→2 | Each generated payload is replaced with the externally verified correct letter/value claim. All four events are marked injected. |
| `mutual_correction` | 0→1, 1→2, 2→0, 0→2 | Only event 0 is replaced with “My initial guess is [wrong letter]; please verify.” The wrong index is \((y+1)\bmod4\). Later payloads are actual model generations conditioned on delivered inputs. This is the seeded-error-ring primary condition. |
| `aggregate_only` | 2→0, 1→2, 0→1 | Aggregating yields a directed triangle; no ordering of these events completes a time-respecting return. Senders' incoming histories are explicitly emptied. |

The last condition differs from the seeded-error ring in multiple ways: event order, event count, and intentional history isolation. It is a negative-control mechanism condition, not a clean single-factor randomized estimate of temporal ordering alone.

Generation seeds are the integer represented by the first eight hexadecimal characters of SHA256 of the JSON-encoded `[task_id, 17, event_id]`. The same event seeds pair policies and counterfactual network replays. Equal seeds do not force equal tokens when prompts change.

The copied-correct manipulation generates a model response and then discards/replaces it. Those inference calls are retained in compute accounting. These fixed injected copies cannot demonstrate propagation of an intervened probe through their payloads.

## 6. Counterfactual value and causal uptake

For receiver \(j\), let \(A_j\) denote all scheduled incoming candidate messages in the baseline, fully delivered transcript. Coalition \(S\subseteq A_j\) enables exactly those payloads and replaces others with a neutral control. Role and sender/event metadata are held fixed. Controls repeat the token “neutral” to match payload token count; the full template token-count invariant is checked during calibration. Matching token count does not establish semantic neutrality.

The implemented replay set contains empty, singleton, full, leave-one-out, and pair coalitions. Here each receiver has at most two incoming messages, so the relevant set function is fully enumerated. An unmatched message-removal replay is also evaluated for diagnostics, even though its ablation policy is not in this reduced experiment.

Let \(\ell_a(S)\) be the model's **full-vocabulary** next-token log probability for letter \(a\in\{A,B,C,D\}\) after the chat prompt. A–D are single tokens (IDs 32–35), so one forward pass supplies all four coordinates. For reference letter \(y\),

\[
V_j(S)=\ell_y(S).
\]

This is one reference-letter log score for the fixed task/context, **not** a repeated-sampling mean and not likelihood of a generated free-text solution. Token-length normalization is trivial because each reference has length one. It does not score the numeric answer sequence. The reported reference log score is not renormalized to A–D.

Distribution-valued diagnostics use the finite-coordinate approximation

\[
q_a(S)=\frac{\exp\ell_a(S)}{\sum_{b\in\{A,B,C,D\}}\exp\ell_b(S)}.
\]

The predicted answer is \(\arg\max_a q_a\). These coordinates omit probability mass assigned to other tokens. Message uptake is

\[
u_e=\min\left\{1,\|q(A_j)-q(A_j\setminus\{e\})\|_2\right\}.
\]

These uptake values use final receiver coalitions and are context-dependent. They are not per-hop hypothesis tests or estimates with calibrated uncertainty intervals.

## 7. H and second-order Möbius redundancy R

With fixed \(B=B_R=1\) log-probability unit and \([x]_+=\max(0,x)\), the harmful marginal is

\[
H_e=\min\left\{1,\frac{[V_j(A_j\setminus\{e\})-V_j(A_j)]_+}{B}\right\}.
\]

Positive H means masking a payload increases the reference-letter log score. It does not identify malicious intent.

For co-received messages \(e,f\),

\[
\Delta_{ef}V_j=V_j(\{e,f\})-V_j(\{e\})-V_j(\{f\})+V_j(\varnothing).
\]

Negative increments indicate diminishing returns in this reference-log-score set function. Positive increments are not penalized. Each event's ancestry set includes its own ID and recursively includes logged parent events when the parent's measured uptake is at least 0.001; recursion through the current event stops if that event is a fixed injected payload. After all uptake estimates are available, \(a_{ef}=1\) iff the two ancestry sets intersect. This is an operational provenance/uptake heuristic, not a statistical proof of shared causal origin.

\[
R_e=\min\left\{1,
\frac{\sum_{f\ne e}a_{ef}[-\Delta_{ef}V_j]_+}{B_R\max(1,\sum_{f\ne e}a_{ef})}
\right\}.
\]

The broader implementation also stores unrestricted-pair and unmatched-control quantities; their policies were not evaluated in this reduced study.

## 8. Directed path homology and P

Each network case is one aggregation window. Arc weight is the **maximum** measured uptake of events sharing its directed agent pair:

\[
W_{ij}=\max_{e:i\to j}u_e.
\]

At threshold \(\theta\), keep arcs with \(W_{ij}\ge\theta\). The code computes first **regular directed path homology over real coefficients**, not undirected cycle rank or clique homology.

Let \(D_1\) be the vertex/allowed-arc incidence matrix. For every allowed 2-path \((i,j,k)\), regular boundary is

\[
\partial(i,j,k)=(j,k)-(i,k)+(i,j),
\]

with degenerate \((i,i)\) terms discarded. Split this boundary into rows for allowed arcs (matrix \(A\)) and forbidden regular shortcuts (matrix \(F\)). If columns of \(N\) span \(\ker F\), then allowed 2-chains with allowed boundaries have image \(AN\), and

\[
\beta_1=|E|-\operatorname{rank}(D_1)-\operatorname{rank}(AN).
\]

Null spaces use SciPy's numerical SVD routine; rank calculations use tolerance \(10^{-9}\). There is no finite-field implementation or topology uncertainty analysis.

For event e, delete that event and recompute max aggregation (do not automatically delete a shared arc). Then

\[
P_e=\int_0^1\min\{1,|\beta_1(G_\theta)-\beta_1(G^{(-e)}_\theta)|\}\,d\theta.
\]

The integral is evaluated exactly for the measured step filtration, up to numerical rank decisions, by summing interval widths between all distinct original/deleted arc weights. A large P is structural sensitivity, not harmful-echo evidence. The aggregate-only control had mean P=1 but E=R=H=0 and was fully retained.

## 9. Time-respecting returns and echo fidelity E

Candidate paths have at most three hops, no repeated agent except the return to the origin, strictly increasing event times, matching consecutive sender/receiver IDs, and measured event uptake at least 0.001. There is no multi-window persistence test, unlimited walk enumeration, or learned semantic representation. The time-unrolled event graph itself is acyclic; returns connect an agent to a later copy of itself.

For each candidate return, replace its first payload with a wrong-option probe or length-matched neutral control. Regenerate the network with paired seeds under both routed and first-event-blocked conditions. End-point answer coordinates are measured at the same logical end time. If \(q_{p,r},q_{c,r}\) denote probe/control with route and \(q_{p,0},q_{c,0}\) their blocked counterparts,

\[
d_C=(q_{p,r}-q_{c,r})-(q_{p,0}-q_{c,0}).
\]

The original-agent reference displacement \(d_0\) is measured from its pre-route received history plus the probe/control payload, using the same named answer coordinates. This is an explicit prompt construction, not a hidden-state identity. Separate probe/control propagated messages are also scored to check downstream hop uptake. Eligibility requires \(\|d_0\|_2\ge0.001\) and every tested downstream hop displacement at least 0.001. Paths containing a fixed injected intermediate payload are excluded.

For eligible paths,

\[
F_C=\left[1-\frac{\|d_C-d_0\|_2}{\|d_0\|_2+10^{-9}}\right]_+,
\qquad E_e=\operatorname{mean}_{C\ni e}F_C.
\]

Ineligible tested paths contribute zero fidelity; an empty path set gives E=0. This differs from averaging only successful/eligible paths: the implementation's denominator includes tested candidates assigned zero for failed eligibility.

**Observed:** three return candidates were tested across the two seeded-error calibration cases: one length-3 and two length-2 paths. All had zero downstream hop displacement, were marked ineligible, and had F=0. Other conditions had no tested eligible return candidates. E was zero everywhere. Do not describe these paths as confirmed causal echoes or claim that fidelity discrimination succeeded.

## 10. Frozen calibration, gate, and prospective routing

For each registered condition, mode, and event position, average H, P, E, R **separately** over the two calibration tasks to obtain \(\bar H_e,\bar P_e,\bar E_e,\bar R_e\). Evaluation truths are not used to choose evaluation gate factors. This is condition-supervised transfer: the scheduler knows the condition and event position; the router is not inferring an unknown attack class.

The product of mean factors is used, not the mean of factor products:

\[
S_e=\bar H_e+\kappa\bar P_e\bar E_e\bar R_e,
\qquad w_e=\frac{1}{1+\lambda S_e},\qquad\lambda=3,\ \kappa=1.
\]

For fixed nonnegative s, this gate uniquely minimizes the worst-case surrogate

\[
\min_{w\in[0,1]}\max_{a\in[0,\lambda s]}(1-w)^2+a w^2.
\]

The adversary chooses \(a=\lambda s\), and the excess objective is \((1+\lambda s)(w-w^*)^2\). This is a theorem about a specified quadratic surrogate, not a theorem about model accuracy. Likewise, independent delivery along k fixed edges with \(\lambda S_e\ge1\) gives route survival at most \(2^{-k}\); it does not bound alternative paths or network error probability.

Observed acceptance probabilities, in event-ID order:

| Condition | Frozen MoRCA probabilities |
|---|---|
| independent | 0.83676209, 1 |
| copied_correct | 1, 1, 1, 1 |
| seeded-error ring | 0.31408533, 0.90032967, 1, 1 |
| aggregate_only | 1, 1, 1 |

Each evaluation condition first generates a common unfiltered **shadow transcript**. For each policy, probabilities/decisions are established from the calibration factors and shadow messages, followed by a new prospective rollout: downstream agents see only accepted messages. Dropped messages are absent, not neutral-filled, during deployment-style rollouts. Final scoring uses accepted messages addressed to agent 2. All messages, including rejected ones, remain in audit logs.

Uniform coins derive from distinct SHA256-seeded Python PRNGs keyed by task, mode, condition, registered seed, policy, and event ID. A protected flag forces delivery, but this synthetic study contains no protected events; zero protected drops is therefore not empirical evidence of safety-critical preservation. Policy-specific seeds mean two policies with identical probabilities need not have identical coin realizations.

For shadow payload token lengths \(n_e\), matched-budget random thinning uses a constant probability

\[
\rho=\frac{\sum_e w_e n_e}{\sum_e n_e}.
\]

Thus expected **shadow** token traffic matches MoRCA. Realized counts differ, and prospective text lengths change after upstream intervention; this is not exact realized-budget matching. Shadow inference, calibration replays, and counterfactual probes are additional compute costs and must be included in economic claims.

## 11. Registered statistical analysis

Primary endpoint: binary reference-letter correctness at agent 2 under MoRCA versus no routing, within `mutual_correction`, across all 64 evaluation tasks. One seed is fixed. The primary test is two-sided exact McNemar at alpha 0.05. No early stopping for significance occurred; all registered cases completed.

Let b be MoRCA-only correct pairs and c baseline-only correct pairs. Conditional on m=b+c discordant pairs, the null uses \(\operatorname{Binomial}(m,1/2)\). The two-sided exact p-value is

\[
p=\min\left\{1,2\sum_{k=0}^{\min(b,c)}{m\choose k}2^{-m}\right\}.
\]

Observed b=7, c=0, both-correct=48, both-incorrect=9. Therefore

\[
\widehat\Delta=\frac{7}{64}=0.109375,
\qquad p=2(1/2)^7=0.015625.
\]

This meets the registered primary criterion. It is not a posterior probability that the technique works, and it does not test the full MoRCA interaction contribution.

Across-condition accuracy-difference intervals in `summary.json` are descriptive percentile bootstrap intervals from 2,000 resamples of **task clusters**, with all four conditions paired within each task; RNG seed is 2026. These intervals concern the across-condition mean, not the primary ring-only difference. They are not adjusted simultaneous intervals for every exploratory result. No primary-effect confidence interval was computed by the original analysis; do not relabel the pooled interval as primary.

Planning power assumed 25% MoRCA-only correct and 5% baseline-only correct (20-point improvement, 30% discordance); 61 tasks provided about 80% power under that scenario, motivating 64 tasks. Observed discordance and improvement differed. Do not report the planning 80% as attained power for the observed result or treat sample size as a guarantee of significance.

## 12. Verified numerical results

| Condition (64 tasks each) | No routing | MoRCA | Random thinning |
|---|---:|---:|---:|
| Independent | 61/64 | 61/64 | 60/64 |
| Copied correct | 64/64 | 64/64 | 64/64 |
| Seeded-error ring (primary) | 48/64 | 55/64 | 53/64 |
| Aggregate-only ring | 60/64 | 60/64 | 60/64 |
| All 256 condition-task cases, descriptive | 233/256 (91.02%) | 240/256 (93.75%) | 237/256 (92.58%) |

MoRCA's pooled gain is 2.734375 percentage points; its task-cluster bootstrap 95% interval is [0.78125, 4.6875] percentage points. This pooled result is exploratory. MoRCA versus random thinning in the primary condition has four wins and two losses, exact two-sided p=0.6875 (**posthoc exploratory calculation**). Therefore this pilot does not demonstrate superiority to matched-budget random thinning.

Total delivered payload tokens across evaluation cases were 15,840 (none), 14,939 (MoRCA), and 15,074 (random). MoRCA's reduction versus none was 901/15,840 = 5.69%. This is communication-payload traffic, not total inference tokens or a proven compute saving.

Posthoc baseline helpfulness labels use \(V(A)-V(A\setminus\{e\})>0.001\) for incoming messages to final receiver 2 in the unfiltered shadow transcript. Of 327 helpful event instances, MoRCA suppressed 5 and random suppressed 14 (1.53% versus 4.28%). These are correlated event-level descriptive rates, not an independently powered safety comparison or labels of usefulness after routing changes the transcript.

The reported harm AUROC is 0.83805949. Its labels use negative held-out shadow marginal below -0.001, and the score is 1-w. It is pooled across conditions, has no reported confidence interval, and may reflect known condition/event-position differences. It is **not** held-out redundancy AUROC or evidence that unknown harmful events can be discovered online.

Actual model-call totals: 3,426 generations and 1,602 answer-scoring forwards (no embedding calls). Recorded aggregate inference tokens: 530,861 input and 89,139 generated. These include calibration, probe replays, shadow networks, all policy rollouts, and audits. Estimated planning counts were slightly higher because fewer return candidates required replay. CPU inference had no paid model API invoice; electricity, cloud cost, and GPU timing were not measured.

## 13. Limitations and claims to avoid

1. **Full mechanism not established:** E=0 makes the implemented multiplicative term inactive. P and R were computed, but their interaction did not drive any acceptance decision.
2. Only two calibration tasks determine condition/event-position weights. No calibration uncertainty bounds, sensitivity analysis, or alternate split replication were run.
3. The primary gain is against no routing; superiority over random thinning was not established. No published AgentPrune/E2-Explainer/CAGE-CAL reproduction or H-only randomized ablation was executed.
4. This is one small checkpoint, one seed, one arithmetic family, three agents, and a short fixed schedule. No real incident-triage task, six-agent scaling, production workload, H100 run, or thinking-mode experiment was performed.
5. “Independent” agents share the same question and model. They do not provide distinct private evidence; “mutual correction” is a manipulation name, not evidence that all agents corrected themselves.
6. H/R depend on a chosen letter-log-score utility, neutral-token control, fixed B scales, and specific contexts. Scoring the full answer text, alternative controls, or a larger answer space could change results.
7. Known condition labels and fixed event positions enable transfer from calibration. Do not market the policy as an unconditioned detector for arbitrary message streams.
8. The combined gain and traffic reduction do not establish net compute savings. Counterfactual inference costs are material; some injected-condition generation is deliberately discarded.
9. Zero protected drops is vacuous here because no protected events were present. No safety-critical preservation claim follows.
10. The full proposal's bibliography and novelty claims were preserved, not independently validated. The experiment cannot establish research priority.

## 14. Formal proof scope and drafting guidance

The supplied Lean source successfully checked unchanged using Lean 4.30.0 and mathlib commit `c5ea00351c28e24afc9f0f84379aa41082b1188f`. It establishes score nonnegativity, gate bounds, a neutral case, a finite-list product bound, and a quadratic optimization inequality. It does not formalize the empirical estimator, implementation correctness of path homology, causal identification, distribution of real router coins, model accuracy, or all uniqueness/generalization claims made in prose. The source does not contain the proposal's analytic stability theorem.

For drafting, distinguish (i) conditional mathematical properties of a gate, (ii) implementation of causal/topological measurement machinery, and (iii) empirical evidence for the actually active H-based gate. An appropriate paper section is a preliminary controlled pilot with a positive preregistered-within-project primary comparison and a negative/unsuccessful echo-activation result. The local frozen plan is not a public preregistration registry. Any new ablation or broader benchmark should be labeled a new experiment with fresh held-out data.

The archive contains complete inputs and raw outcomes for independent reanalysis. A drafting agent should inspect `records.jsonl` and source code before deriving new statistics, label posthoc work explicitly, and retain the original primary result and study limitations even if a later analysis is less favorable.

## GitHub packaging note

The standalone Python package removes comments/docstrings and uses `config.json` for the recorded laptop design. Default new outputs go to `runs/pilot`, separate from the archived results. `prepare` now downloads the pinned checkpoint when absent. Numerical experiment and analysis functions are unchanged. Original execution hashes in `manifest.json` and the historical prepared plan are retained; they describe the original commented source, not byte hashes of the cleaned package. Lean is distributed as a separate standalone project. Historical file paths in this handoff refer to the original experiment bundle; this package includes the executed core, tests, and results.
