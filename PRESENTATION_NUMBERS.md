# FleetMind — numbers for the defence

**Every figure below was re-verified against the repository on 2026-10-07**, from the
file or command named beside it. Nothing here is quoted from a previous document.
Run all commands from `backend/`.

---

## (a) Safe to quote

### Classifier accuracy — the deployed checkpoints

`mlruns/depot_{1-4}/cnn_state_dict.pt`, the files that produced every stored
prediction and decision.

**Source:** `python -m fleetmind.decision.report` → `mlruns/decision/phase0/`

| Site | Load | Window-level accuracy | Correct / held out |
|---|---|---|---|
| Plant A | 0 HP | **53.81%** | 127 / 236 |
| Plant B | 1 HP | **69.49%** | 164 / 236 |
| Plant C | 2 HP | **61.02%** | 144 / 236 |
| Plant D | 3 HP | **57.20%** | 135 / 236 |
| **overall** | — | **60.38%** | **570 / 944** |

60.38% is both the mean of the four sites and the pooled figure — they coincide to
six decimals because every held-out set is 236 windows.

**Three qualifications are part of the claim, not footnotes.**

1. **Three-class, not four.** Each site has exactly one Normal recording and the
   stratified split assigns a single-recording class entirely to training, so **no
   healthy machine is ever held out**. The held-out set is identical at every site:
   `{OuterRace 118, Ball 59, InnerRace 59}`.
2. **The floor is 50.0%, not 25%.** OuterRace is exactly half the held-out set, so
   always predicting OuterRace scores 50.00%. Plant A clears that by **3.81 points**.
3. **The false-alarm rate on a healthy machine has never been measured** and cannot
   be on this split. It is absent, not zero.

### Recording-level accuracy — the decision the product actually makes

The product flags a machine, not a 0.17 s window. Logits averaged over each
recording's windows, argmax taken, scored against the documented label.

**Source:** `python -m fleetmind.decision.recording` → `mlruns/decision/recording_level/`

| | Window-level | Recording-level |
|---|---|---|
| overall | 570 / 944 = **60.38%** | **10 / 16 = 62.50%** |
| Plant A | 0.5381 | 2 / 4 |
| Plant B | 0.6949 | 3 / 4 |
| Plant C | 0.6102 | 3 / 4 |
| Plant D | 0.5720 | 2 / 4 |

**Say this before anyone asks:** n = 16 recordings, so **one recording is 6.25% of
the figure** — the whole +2.1-point difference is a third of a single flip, and is
not an improvement. All three aggregation rules (logit mean, probability mean,
majority vote) give identically 10/16.

**Volunteer the per-class breakdown — it is the strongest honest point available:**

| True class | Recordings correct | What the misses became |
|---|---|---|
| OuterRace | **8 / 8** | — |
| Ball | 2 / 4 | Ball → OuterRace ×2 |
| **InnerRace** | **0 / 4** | → Ball ×3, → OuterRace ×1 |

**Not one held-out InnerRace recording is correctly diagnosed, at any site.** The
FFT-magnitude fix made InnerRace learnable in training; it did not make it
identifiable on held-out recordings. Stating this yourself is worth more than
having it extracted.

### Federated learning — a negative result, stated as one

**Source:** `mlruns/federated/fedprox/federated.json` and `mlruns/federated/bn_study/`

| Configuration | Mean accuracy | vs. the 60.05% local baseline |
|---|---|---|
| Mean local baseline | **60.05%** | — |
| FedProx as run (BatchNorm buffers never transported) | **42.86%** | **−17.19 pts** |
| FedProx with BatchNorm buffers aggregated | **52.60%** | **−7.45 pts** |
| FedProx with buffers aggregated **and** recalibrated per site | 45.52% | −14.53 pts |

**Both federated configurations are below the local baseline.** About half the
original gap was an implementation artifact: `get_parameters()` moves only
`named_parameters()`, so `BearingCNN`'s 9 BatchNorm buffers never reached the
global model, which then evaluated with `running_mean=0, running_var=1`.
Aggregating them recovers 9.7 points and lifts InnerRace recall off the floor, but
Normal and Ball recall stay at 0.00 and OuterRace at 1.00 in **every** variant — so
a real non-IID majority-class bias remains after the artifact is removed.

Plain FedAvg oscillated round to round and is not the headline result; FedProx
(mu = 0.1, 30 rounds) is.

---

## (b) Do NOT quote these numbers

| Number | Why not |
|---|---|
| **91.44%** as accuracy | It is **in-sample model-vs-ground-truth agreement** — 3,993 of 4,367 stored predictions match the documented label, but each site's model predicted over all of its own windows, including ones it trained on. A useful integration check; a meaningless generalisation claim. If someone quotes it back at you, correct them. |
| **62.7 – 72.5%** (and 64.83 / 72.46 / 62.71 / 65.68%) | These are `metrics.json`'s `best_val_acc` — the best accuracy any *epoch* reached. `ml/train.py` calls `torch.save()` **after** the training loop with no best-epoch guard, so the saved weights are the *last* epoch's. Those figures describe models **never written to disk**. Confirmed twice: through `decision/probs.py`, and by replaying `train.py`'s own validation loop against the saved checkpoints. |
| Any **"32-fold"** calibration improvement | It came from a *signed* over-confidence mean (+0.2101 → +0.0065), where over- and under-confident bins cancel. The honest figure is **ECE 0.2146 → 0.0966, about 2.2× lower**, with **Brier 0.5830 → 0.5726 — barely changed**. Brier is a proper scoring rule, so that near-flat result is the one that matters. |

---

## (c) Do NOT claim these things

- **Cloud deployment.** The system has never been deployed. `docker compose config`
  resolves; `docker compose up` has **never run** — the Docker daemon is unreachable
  on this laptop (WSL2/Hyper-V blocked by policy). It runs locally only.
- **Live sensors.** There are none. CWRU is a static archive; the replay engine
  progressively releases recorded windows and the UI says so. GPS and site locations
  on `Equipment`/`WorkOrder` are **simulated** — CWRU has no location data.
- **Learned or autonomous agents.** The Monitor → Diagnostic → Risk → Maintenance
  chain is four deterministic rule functions. The LLM layer writes prose and may not
  originate a number. Nothing in the decision path is learned except the classifier.
- **Federated learning beating local training.** It does not, in any configuration
  measured. See (a).
- **Any ROI, cost-saving, downtime-avoided or payback figure.** No such number exists
  anywhere in this project, and none can be derived from CWRU.
- **That the bearing defect frequencies explain the model.** They are *independent*
  physical corroboration. The CNN never sees them — verified by grep: no defect
  frequency appears in `ml/model.py`, `ml/train.py`, `ml/predict.py` or
  `ml/dataset.py`.

---

## (d) Phase 2, in one line

**Temperature scaling and split-conformal prediction sets were both implemented,
tested and measured — and neither helped on this data, so neither was shipped.**
Temperature made test calibration *worse* on both model arms (mean ECE 0.1082 →
0.1172; mean Brier 0.5812 → 0.5876), with the fitted temperature disagreeing about
the very direction of miscalibration across sites (0.724 / 0.646 / 1.351 / 2.379).
Conformal coverage missed its 90% target at **2 of 4 sites** (0.8686, 0.8644), and
where it succeeded it was close to vacuous — Plant D reached 1.0000 coverage with
**3.58 of 4 classes** in the set. The binding constraint is the calibration sample:
**4 recordings per site**, whose windows are strongly autocorrelated, so the
exchangeability that split conformal rests on is violated by construction. Both
implementations were verified correct where they can be — coverage ≥ 0.90 over 40
i.i.d. repetitions on exchangeable synthetic data — which is what makes the negative
result attributable to the dataset rather than to the code.

**Source:** `python -m fleetmind.decision.phase2` → `mlruns/decision/phase2/`

---

## Verification state

**Source:** run from `backend/`

| Check | Result |
|---|---|
| `pytest` | **329 passed** |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 111 files already formatted |
| `mypy fleetmind` | no issues in 78 source files |

`mlruns/` is gitignored, so every artefact above is reproducible from the commands
named rather than committed.
