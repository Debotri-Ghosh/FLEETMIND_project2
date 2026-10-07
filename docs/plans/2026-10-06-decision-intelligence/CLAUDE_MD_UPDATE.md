## Decision layer (authoritative — supersedes the "rule-based agents are the authoritative decision path" wording above)

FleetMind's maintenance decisions are moving from a hand-tuned risk heuristic to a
**calibrated probabilistic decision layer**. Where the old path multiplied invented
constants together, the new one requires that every number be either *measured on
held-out data* or an *explicit domain input stated in real units*, and that the
resulting decision carry a stated statistical guarantee.

The four rule functions in `agents/agents.py` are **retained as a named baseline to
beat, not as the product's intelligence.** They are not deleted, because a negative
control is evidence and because three things still depend on them: `agents/run.py`'s
documented batch figures, the Evidence Package's recomputation of RMS/peak via
`diagnostic_agent()`, and the reproducibility of the 4,367 stored decisions. When
quoting any decision-quality figure, **say which path produced it** — `heuristic
baseline` or `decision layer`.

### Why the heuristic is being replaced — measured, not asserted

Every constant in the old scoring path was invented: `FAULT_SEVERITY`
(Normal 0.0 / Ball 0.4 / InnerRace 0.7 / OuterRace 0.85), `LOAD_RISK_FACTOR`
(0.80 / 1.00 / 1.15 / 1.30 per HP), the `0.7 + 0.3 × confidence` blend, and the
0.15 / 0.40 / 0.70 urgency bands. None is fitted, cited or derived.

**The concrete flaw, confirmed by running the real functions:** `diagnostic_agent()`
computes a `diagnosis_confidence` from signal RMS (`min(0.95, 0.5 + rms × 3.0)`) and
`risk_agent()` reads *that*, not the model's `confidence`. The classifier's predicted
probability therefore never reaches the risk score — only the class label does. With
`predicted_label="OuterRace"`, `load_hp=3` and a fixed signal:

| model confidence | risk_score | urgency |
|---|---|---|
| 0.26 | 1.0 | critical |
| 0.55 | 1.0 | critical |
| 0.9991 | 1.0 | critical |

A barely-above-chance prediction and a near-certain one produce byte-identical
decisions. What *does* move the score is signal amplitude, through that fabricated
formula: at rms 0.01 the risk is 0.8613, at rms 0.10 and 0.40 it saturates at 1.0.
**"What happens when the model is unsure?" currently has the answer "nothing
changes",** which is indefensible for a system whose purpose is to decide when a
human should act.

### The four replacements, and what each one buys

| Technique | Replaces | Guarantee it carries |
|---|---|---|
| Temperature scaling | the RMS-derived fake confidence | measured ECE / Brier improvement on held-out recordings |
| Split-conformal prediction sets | an overconfident point prediction | marginal coverage at a chosen level (e.g. 90%) |
| Expected-cost policy over an explicit cost matrix | the 0.15/0.40/0.70 thresholds | the chosen action minimises expected cost under the calibrated posterior; the matrix is in engineering units and sensitivity-analysed |
| Sequential probability ratio test over consecutive windows | one 0.17 s window flipping a machine into maintenance | controllable false-alarm rate (α) and miss rate (β) |

The cost matrix is still a human input. The difference from `0.85` is that it is
explicit, unit-bearing, auditable and reported with a sensitivity analysis — say that
plainly rather than claiming the judgement was removed.

### Planned module layout

```
backend/fleetmind/decision/
  holdout.py      the canonical held-out split, shared by every phase
  probs.py        per-window 4-class posteriors from a trained checkpoint
  calibration.py  ECE / Brier / reliability bins; then temperature scaling
  conformal.py    split-conformal prediction sets
  costs.py        the cost matrix, in stated units
  policy.py       expected-cost-minimising action choice
  sequential.py   SPRT / CUSUM over consecutive windows
  quality.py      false-alarm rate, miss rate, time-to-detection
  baseline.py     adapter so the heuristic is scored on identical footing
```

Full phasing, exit criteria and risks: `docs/plans/2026-10-06-decision-intelligence/PLAN.md`.

---

## The published accuracies are 3-class figures measured against a 50% floor

Verified by reproducing `ml/train.py`'s split exactly (`stratified_experiment_split`,
`frac=0.8`, `seed=42`) and re-deriving the window counts recorded in
`mlruns/depot_{1-4}/metrics.json` — 766/236, 885/236, 885/236, 887/236 all reproduce.

**Each site has exactly one Normal recording** (per site: 4 Ball, 4 InnerRace,
1 Normal, 7 OuterRace = 16 recordings; 64 recordings and 4,367 windows in total).
`stratified_experiment_split` sends any fault group holding a single recording
*entirely to train* (`if len(ids) == 1: train += ids; continue`). Therefore:

- **The validation set contains zero Normal windows at all four sites.** Its
  composition is identical everywhere: `{OuterRace 118, Ball 59, InnerRace 59}`.
- The published accuracies are **3-class** figures, even though the model emits four.
- **OuterRace is exactly 50.0% of the validation set** (118/236), so always predicting
  OuterRace scores 50.0%. The honest comparison is against that floor, not against
  25% uniform chance:

| site | published val accuracy | margin over always-OuterRace |
|---|---|---|
| 1 | 64.83% | +14.83 pts |
| 2 | 72.46% | +22.46 pts |
| 3 | 62.71% | +12.71 pts |
| 4 | 65.68% | +15.68 pts |
| mean | 66.42% | **+16.42 pts** |

- **False-alarm rate on a healthy machine has never been measured**, because no healthy
  recording is ever held out. For a predictive-maintenance product that is the
  commercially decisive number, and it is currently absent from every figure the
  project quotes.
- This also explains how a Normal recall of 0.00 could sit inside the federated result
  without disturbing any per-site accuracy figure.

None of the published numbers is wrong. They are narrower than they look, and they must
be quoted as "62.7–72.5% over three fault classes, against a 50% majority-class floor,
with healthy machines absent from validation" rather than as bare accuracy.

---

## Correction: the federated collapse is partly a BatchNorm-transport artifact

Supersedes the flat statement that FedProx "collapsed to a majority-class predictor".
That is true of the configuration that was run, but roughly half the gap was an
implementation artifact rather than a property of federated averaging.

`BearingCNN` has **11 learnable parameter tensors** and **9 BatchNorm buffers**, and
`get_parameters`/`set_parameters` move only `named_parameters()` — so the running
statistics never reached the global model, which then evaluated with `running_mean=0,
running_var=1` while carrying weights trained to expect normalised activations.

| variant | mean federated accuracy | gain vs. 0.6005 local baseline | InnerRace recall per site |
|---|---|---|---|
| (a) buffers never sent — **the published run** | 0.4286 | −0.1719 | 0.00 / 0.00 / 0.00 / 0.00 |
| (b) buffers sample-weight-averaged each round | **0.5260** | −0.0745 | 0.06 / 0.50 / 0.50 / 0.30 |
| (c) (b) + per-site BN recalibration before eval | 0.4552 | −0.1453 | 0.00 / 0.19 / 0.19 / 0.00 |

Variant (a) reproduced the published figures exactly, which is what makes (b) and (c)
trustworthy. Aggregating the buffers recovers **+9.7 accuracy points** and lifts
InnerRace recall off the floor everywhere — but Normal and Ball recall stay at 0.00 and
OuterRace at 1.00 in every variant, and even (b) is 7.5 points below the local
baseline. **So a real non-IID majority-class bias remains after the artifact is
removed.** Recalibrating per site (c) was *worse* than simply averaging (b):
re-estimating statistics from each site's own skewed partition reintroduces the
site-level bias that averaging smoothed out.

Defaults are unchanged — `aggregate_bn=False, recalibrate=False` — so the published
result stays reproducible; the new behaviour is opt-in via `--aggregate-bn` /
`--recalibrate-bn` / `--out-dir`. Study outputs:
`backend/mlruns/federated/bn_study/{a_baseline,b_aggregate_bn,c_aggregate_and_recalibrate}/`
(gitignored, like all of `mlruns/`).

**The honest one-sentence version:** federated averaging produced a near-majority-class
predictor; about half that gap was BatchNorm statistics never being transported, and
fixing it recovers roughly ten accuracy points and partially restores minority-class
recall, leaving a smaller but real non-IID penalty.

---

## Additions to the standing constraints for new work

6. **No decision number may be invented.** Anything that influences a maintenance
   action is either fitted on held-out data, or an explicit domain input carrying its
   units and a sensitivity analysis. A bare constant in a scoring path is a defect.
7. **The classifier's uncertainty must reach the decision.** Any new scoring code must
   consume the calibrated posterior, not a quantity recomputed from the raw signal. The
   regression test pinning the old invariance exists so this cannot silently return.
8. **Held-out means held out by recording, and its class composition must be stated**
   beside any figure derived from it. "Accuracy" on a split with no Normal windows is a
   3-class figure and must be labelled as one.
9. **A test that asserts "nothing is wrong" must also assert that it examined
   something**, with a positive count. Three tests in this repository were found
   incapable of failing; every reachability-style test now carries a companion guard
   pinning a floor on what it inspected.
