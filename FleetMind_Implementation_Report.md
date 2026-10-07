# FleetMind — Implementation Report

**Federated Intelligent Predictive Maintenance for Bearing Fault Diagnosis**
M.Tech Project 2 · Debotri Ghosh (25MML1008) · VIT Chennai + LTIMindTree
Report generated 2026-10-03 against commit `74fedc61`.

---

## How to read this report

Every figure below was re-verified against the live system on 2026-10-03 — the
`mlruns/` metrics files, the PostgreSQL database, and the test suite — rather
than copied from an earlier write-up. Where a number is weak, unflattering or
easy to misread, it is stated that way on purpose; this project's working rule
throughout has been that a documented limitation is worth more than a smoothed
one.

Two conventions:

- **Verified** means a command was run and its output read. The commands are
  named so they can be re-run.
- Where the repository's own history does not record something, this report
  says so rather than filling the gap. There is one such case, in §1.

---

## 1. The dataset

### What it is

The **Case Western Reserve University (CWRU) Bearing Data Center** dataset —
the standard public benchmark for rolling-element bearing fault diagnosis.
Cite: `engineering.case.edu/bearingdatacenter`.

As ingested into this system (verified by direct SQL):

| Property | Value |
|---|---|
| Experiments (recordings) | 64 |
| Signal windows | 4,367 |
| Window length | 2,048 samples |
| Sampling rate | 12,000 Hz |
| Sensor position | DE (drive end) |
| Fault classes | Normal, Ball, InnerRace, OuterRace |
| Motor load levels | 0, 1, 2, 3 HP |

Class balance across all 4,367 windows:

| Class | Windows | Share |
|---|---|---|
| OuterRace | 1,652 | 37.83% |
| InnerRace | 944 | 21.62% |
| Ball | 943 | 21.59% |
| Normal | 828 | 18.96% |

This imbalance is not incidental — it is the direct cause of the federated
failure mode described in §2.3, and it is why the per-site classifiers use
inverse-frequency class weighting.

### Why CWRU

The selection criteria this project actually relies on, each traceable to a
design decision in the codebase:

1. **Four real motor load levels.** The federated-learning simulation needs
   genuinely non-IID clients. CWRU's 0/1/2/3 HP loads give four populations
   that differ in shaft speed (1797 / 1772 / 1750 / 1730 rpm) and in class
   balance — real heterogeneity rather than a random partition of one pool.
   Each load level becomes one simulated manufacturing site.
2. **Documented ground truth per recording.** The fault type *and the machined
   defect diameter* are published per file. Without that, the
   ground-truth-vs-prediction evidence panel (§6.1) could not exist.
3. **Known bearing geometry.** CWRU specifies the bearings (SKF 6205-2RS JEM
   at the drive end, 6203-2RS JEM at the fan end), which makes the
   characteristic defect frequencies computable (§6.3). A dataset without
   published bearing specifications would make the physics layer impossible.
4. **No synthetic data.** A hard project constraint: the only generated data
   anywhere is the explicitly documented GAN augmentation step (§8).

> **A gap this report will not paper over.** The repository's history —
> `CLAUDE.md`'s Progress log and all 44 commits — contains **no record of an
> alternative dataset being evaluated and rejected**. The criteria above are
> reconstructed from what the implementation demonstrably depends on, which is
> defensible; a head-to-head comparison against a named rejected candidate is
> not something this repository can support, and inventing one would be the
> kind of retrofitted justification this project has avoided elsewhere.

---

## 2. The three real bugs found and fixed

These are the substantive engineering findings. All three **lowered** the
reported numbers, which is the point: each one had been inflating a result.

### 2.1 Data leakage — the 100% accuracy that was not real

**Symptom.** Per-site validation accuracy read as essentially 100%.

**Cause.** `ml/train.py` and `ml/evaluate.py` split *individual signal windows*
randomly 80/20. Because 4,367 windows come from only 64 recordings, windows
from the same continuous recording landed on both sides of the split. The
model was being validated on near-neighbours of its own training data —
samples milliseconds apart from the same rotating machine.

**Fix.** Switched to an **experiment-level split**, using the
`augmentation/exp_split.py:stratified_experiment_split` that
`augmentation/run.py` and `federated/run.py` were already using correctly. No
recording contributes to both sides.

**Before → after.** ~100% (fictitious) → the honest 53.8–69.5% range in §3.

**Standing rule.** Any new training code must split by experiment ID, never by
individual window.

### 2.2 InnerRace was unlearnable — an architecture bug, not a data problem

**Symptom.** The confusion matrix showed **InnerRace 100% misclassified as
OuterRace, at every one of the four sites.** The class was not merely hard; it
was never predicted at all.

**Cause.** `BearingCNN` ended in an `AdaptiveAvgPool1d` that collapsed the
entire raw time-domain window into a single global-average feature. What
distinguishes InnerRace from OuterRace is the *fault repetition rate* — how
often the rolling elements strike the defect — and averaging the whole window
destroys exactly that information.

**Fix.** Feed **`ml/model.py:to_fft_magnitude()`** — a 512-bin FFT magnitude,
log-scaled and z-normalised — as the model's input instead of raw time
samples, everywhere a classifier trains or predicts. Repetition rate is a
frequency-domain property, so expressing the input in the frequency domain
makes it survive the pooling.

Inverse-frequency `CrossEntropyLoss` weighting (`ml/model.py:class_weights()`)
was added alongside. It helped, but **it did not fix InnerRace on its own** —
the FFT change was the actual fix. This distinction matters: the problem
looked like class imbalance and was not.

**A failed attempt worth recording.** An architecture-only fix — widening the
convolution kernels while keeping raw time-domain input — was tried and made
things **worse: 0.306 vs 0.527 accuracy.** Do not retry it.

### 2.3 Federated instability — FedAvg oscillated; FedProx is the primary result

**Symptom.** Plain FedAvg oscillated wildly round to round under the real
per-site class imbalance.

**Fix.** FedProx with a proximal term (`--strategy fedprox --mu 0.1`, 30
rounds), which stabilised convergence.

**But the honest result is not "stabilised and good".** See §3.2 — FedProx
converges *stably* to a degenerate solution. Stability was achieved; accuracy
was not. Both halves of that sentence belong in any presentation of this
result.

---

## 3. Results — stated unambiguously

### 3.1 The headline number: 53.8–69.5% per-site accuracy, mean 60.4%

These are the accuracies of the **deployed checkpoints** —
`mlruns/depot_{1..4}/cnn_state_dict.pt`, the files that produced every one of the
4,367 predictions and decisions in the database. Measured by
`python -m fleetmind.decision.report`:

| Site | Motor load | Deployed accuracy | Correct / held out | Margin over the 50.0% floor | Train / held-out windows |
|---|---|---|---|---|---|
| Plant A | 0 HP | **53.81%** | 127 / 236 | **+3.81 pts** | 766 / 236 |
| Plant B | 1 HP | **69.49%** | 164 / 236 | +19.49 pts | 885 / 236 |
| Plant C | 2 HP | **61.02%** | 144 / 236 | +11.02 pts | 885 / 236 |
| Plant D | 3 HP | **57.20%** | 135 / 236 | +7.20 pts | 887 / 236 |
| **mean** | — | **60.38%** | 570 / 944 | **+10.38 pts** | — |

**This 53.8–69.5% range, mean 60.38%, is the project's accuracy claim.** Three
qualifications are part of the claim, not footnotes to it.

**It is a three-class figure.** Each site holds exactly one Normal recording, and
`stratified_experiment_split` assigns a single-recording class entirely to
training — so no healthy machine is ever held out at any site. The held-out set is
identical everywhere: `{OuterRace 118, Ball 59, InnerRace 59}`. The model emits
four classes and only three are ever tested.

**The floor is 50.0%, not 25%.** OuterRace is exactly half the held-out set, so
always predicting OuterRace scores 50.00%. Plant A's deployed model clears that
by 3.81 points, with InnerRace recall 0.0 and an OuterRace predicted share of
0.915 — it is very nearly the same majority-class predictor criticised in §3.2.

**The false-alarm rate on a healthy machine has never been measured** and cannot
be on this split. It is absent, not zero; `fleetmind/decision/quality.py` reports
it as `None` with a machine-readable reason rather than as `0.0`.

#### 3.1.1 Why these replace an earlier 62.7–72.5% claim

An earlier version of this report quoted 64.83 / 72.46 / 62.71 / 65.68% read from
`mlruns/depot_{1..4}/metrics.json`. That field is `best_val_acc`: the best
accuracy any *epoch* reached. `ml/train.py` tracks it with
`best_acc = max(best_acc, acc)` and then calls `torch.save(model.state_dict(), …)`
**after** the loop, outside any best-epoch guard — there is no best-model
checkpointing at all. The saved weights are therefore the **final** epoch's, and
the earlier figures describe models that were never written to disk.

| Site | `metrics.json` `best_val_acc` | the saved checkpoint actually scores | gap |
|---|---|---|---|
| Plant A | 0.6483 | **0.5381** | −0.1102 |
| Plant B | 0.7246 | **0.6949** | −0.0297 |
| Plant C | 0.6271 | **0.6102** | −0.0169 |
| Plant D | 0.6568 | **0.5720** | −0.0847 |
| mean | 0.6642 | **0.6038** | **−0.0604** |

Confirmed two independent ways: through `fleetmind/decision/probs.py`, and by
replaying `ml/train.py`'s *own* validation loop and its *own* `_TorchDataset`
against the saved checkpoints — both give 0.5381 / 0.6949 / 0.6102 / 0.5720, so
the gap cannot be an evaluation difference.

**The checkpoints are deliberately not retrained over.** All 4,367 stored
predictions and decisions, the six flagged machines and both LLM narratives were
produced by these exact files; retraining would silently invalidate every
downstream figure in this report. The honest measured number is quoted instead.
A properly split retrain is being done separately, to new `mlruns/depot_N_v2`
checkpoints, so that v1 and v2 can be compared on data neither has trained on.

### 3.2 Federated: FedProx converged to a majority-class predictor

Verified from `mlruns/federated/fedprox/federated.json`:

| Metric | Value |
|---|---|
| Strategy | FedProx, mu = 0.1, 30 rounds |
| Mean local baseline | **60.05%** |
| Mean federated final | **42.86%** |
| Gain vs baseline | **−17.19 points** |

The gap is well known in the federated-learning literature under non-IID class
imbalance. But the per-class recall in the same file says something sharper
and should be stated rather than hidden:

> At **every one of the four sites**, the federated model's per-class recall is
> **OuterRace 1.0, Normal 0.0, InnerRace 0.0, Ball 0.0.**

That is not a model that is somewhat less accurate. It is a model that has
**collapsed to predicting the majority class for every input**, and 42.86% is
simply the OuterRace share of the validation sets. Federated averaging across
four clients with different class distributions drove the global model to the
only prediction that is never badly wrong on average.

This is a genuine negative result and the most intellectually interesting
finding in the project. It should be presented as one.

### 3.3 The 91.44% figure — what it is, and what it is not

> ### ⚠ Do not quote 91.44% as accuracy.
> It is **in-sample model-vs-ground-truth agreement**, not validated accuracy.
> The accuracy figure is **53.8–69.5%, mean 60.38%** (§3.1). Conflating the two
> overstates the model by roughly 22–38 points.

Verified by SQL across all 4,367 predictions:

| | Count | Share |
|---|---|---|
| Prediction matches documented label | 3,993 | **91.44%** |
| Prediction disagrees | 374 | 8.56% |

Why it is not accuracy: `model_predictions` was populated by running each
site's trained model over **all** of that site's windows — including the ones
it was trained on. The figure therefore measures how well the pipeline
reproduces its own training data end to end, which is a useful *integration*
check (it confirms the stored predictions really came from the models) and a
useless *generalisation* claim.

The disagreements are structured, not random:

| Documented | Predicted | Count |
|---|---|---|
| InnerRace | Ball | 163 |
| Ball | OuterRace | 133 |
| InnerRace | OuterRace | 63 |
| OuterRace | Ball | 14 |
| OuterRace | InnerRace | 1 |

**A consequence worth volunteering.** Because `FAULT_SEVERITY` ranks OuterRace
highest (0.85), a Ball→OuterRace error escalates the risk score and can trip
the maintenance flag. Of the **6** equipment currently flagged by the pipeline,
**4 rest on a prediction that disagrees with the documented label**, all in the
direction of over-predicting severity. The AI-flagged list is therefore
*enriched* in the model's over-predictions. For a maintenance product that
failure direction is the safer one — a false alarm costs an inspection, a
missed fault costs a machine — but it is a real bias and should be named.

---

## 4. The product reframe: industrial, not vehicular

The project is a **predictive-maintenance and field-service platform for
rotating industrial machinery**. A machine builder or service provider monitors
motors and pumps installed in its customers' factories, detects bearing faults
from vibration, backs every finding with inspectable evidence, and dispatches
technicians. **It recommends; humans decide.**

It is explicitly **not** a vehicle fleet, ride-hailing or logistics product.

An audit (commit `f12db083`) found that `vehicle`, `Uber`, `Rapido` and `rider`
appeared **nowhere** in the code — the only legacy framing was the word `depot`
plus some transport-flavoured seed data. What changed:

- Navigation and headings: "Depot Operator" → **Site Operations**
- Table headers and filters: "Depot" / "All depots" → **Site** / **All sites**
- Seed customers: Acme Logistics / Global Freight / Rapid Transport →
  **Acme Manufacturing / Northwind Steel / Pinnacle Processing**
- Site display names: `depot_0hp…depot_3hp` → **Plant A–D (CWRU N HP)**, which
  keeps the dataset mapping visible on screen so every figure stays traceable

**Deliberately unchanged:** the `depots` table, `depot_id` columns and query
parameters, the `/depots` route, and `mlruns/depot_N/` paths. Renaming the
schema is high-risk for low benefit; the decision and its reasoning are
recorded in `docs/adr/0001-depots-to-sites-rename.md` as **Option D — defer the
rename, change only user-facing vocabulary.**

That ADR also records the deeper issue honestly: `depots` rows are *operating
conditions*, not places. The name is legacy for "site", the mismatch is
contained because the schema word is never rendered to a user, and the ADR
names the migration path if a real multi-site model is ever needed.

Re-audited 2026-10-03: zero vehicle/ride/logistics language in code, UI, README
or the live database, including `agent_decisions.decision_text`.

---

## 5. The dynamic replay system

### Why it exists

CWRU is a **static historical archive**. Putting a scheduler over it would
recompute identical results on a timer — automation theatre. Instead:

`signal_windows.released_at` (migration `0004`) marks **arrival**. The replay
engine progressively releases already-ingested windows as if sensors were
streaming them, and the pipeline processes only what has arrived.

### How it works

```
replay/engine.py     release / status / reset
replay/worker.py     asyncio tick loop, CPU work offloaded to a thread
pipeline.py          process_arrived_windows(): predict → agents → sync
/replay/{status,start,stop,step,reset}
```

`pipeline.py:process_arrived_windows()` chains the three stages so the
background worker and the manual trigger share exactly one code path.
`ml/predict.py:predict_depot(incremental=True)` processes only
arrived-and-unpredicted windows; the default `False` preserves the documented
batch behaviour byte-for-byte.

### How it differs from a live sensor feed — stated plainly

> **This is a replay of recorded data. It is not live telemetry, and no sensors
> are attached.** The UI says so on the panel itself.

What is genuinely real: the incremental processing, the ordering, the fact that
each tick's predictions, decisions, status changes and filed requests are
computed fresh from data the pipeline had not previously seen. What is
simulated: the arrival itself.

### Two agent input paths — know which produced a number

A subtlety that matters when quoting figures:

- `agents/run.py --all-pending` (the original batch CLI) feeds the agent chain
  each window's **ground-truth label**. This path flags **4** equipment.
- `pipeline.py:process_arrived_windows()` (the replay path) feeds it the
  **model's predicted label and confidence** — the honest sensor→model→decision
  flow the project claims. This path flags **6**.

The live database holds the **model-prediction** result. Only that path
demonstrates the model actually driving maintenance decisions, so it is the one
to quote — and to say so. `agents/run.py` is left unchanged so the older
documented batch figures stay reproducible.

---

## 6. The evidence and explainability layer

The question a reviewer asks first is *how do you know the model isn't just
reading the label?* This layer exists to answer it with something checkable.

### 6.1 Ground truth vs. independent prediction

`GET /service-requests/{id}/evidence` returns the CWRU-documented facts and the
model's output as **separate fields**, and the UI renders them as two
deliberately unlike panels:

- **Left — archival.** Dashed border, muted ink: documented fault type, machined
  defect diameter, sensor position, motor load, nominal rpm, window label, and
  the source `.mat` file, with the CWRU citation.
- **Right — live.** The fault class's own colour: predicted label, model
  confidence, checkpoint, recomputed RMS and peak-to-peak, diagnosis
  confidence, risk score.

A verdict strip states **Agreement** or **Disagreement** — and is as prominent
on a miss as on a match, because a panel that only celebrates matches is
evidence of nothing. The comparison uses the *base* class, so the diagnostic
agent's "(unconfirmed — low signal energy)" qualifier is not mistaken for a
different class.

### 6.2 The Frequency Lab

An interactive explorer that stages the transform deliberately: the raw
waveform is shown alone, the user presses an explicit button, and **the same
chart morphs** into the magnitude spectrum.

The staging is the argument. A second panel appearing would read as a second
measurement; transforming the chart already on screen reads as *that signal,
seen differently* — which is the true statement. Both domains arrive in one
API response (`GET /signal-window/{id}/spectrum`) so no fetch at the moment of
transform could suggest new data arrived. The copy states it outright:

> Frequency content is computed from this exact recording via Fourier
> Transform — nothing new is collected.

### 6.3 Bearing characteristic frequencies — and what they are not

`ml/bearing_freq.py` computes the four defect frequencies from published
bearing geometry and shaft speed:

| Bearing | BPFI | BPFO | BSF | FTF |
|---|---|---|---|---|
| SKF 6205-2RS JEM (DE) | 5.4152× | 3.5848× | 4.7135× | 0.3983× |
| SKF 6203-2RS JEM (FE) | 4.9469× | 3.0530× | 3.9874× | 0.3817× |

(multipliers of shaft speed in Hz; Ball faults are inspected at 2×BSF, since a
ball defect strikes both races once per revolution)

> **These are not an explanation of the model.** The CNN consumes a 512-bin
> log-scaled, z-normalised FFT magnitude and was never given these frequencies.
> They are *independent physical corroboration* a human can check the spectrum
> against. Presenting them as the model's reasoning would be false, and the UI
> says so wherever a band is drawn.

### 6.4 The weak-corroboration finding

The most useful thing this layer produced is a result that is *unflattering*.

On **signal window 2713** (Bearing Motor D2, Plant D, 3 HP — currently service
request **#54**; it was #41 before the last replay reset, which renumbered the
requests):

| | |
|---|---|
| Documented class | Ball |
| Model prediction | **Ball at 99.91% confidence** |
| Matched defect band | 2×BSF = 271.82 Hz (258.23–285.41 Hz) |
| Energy in that band | **0.000325** of total spectral energy |
| Overall spectral peak | **1,382.81 Hz** — far outside the band |

For scale: a perfectly flat spectrum would place ~0.45% of its energy in a
27 Hz band out of 0–6,000 Hz. The defect band holds roughly **one fourteenth**
of that. The physics does not merely fail to corroborate the call — there is
*less* energy at the defect frequency than nothing-in-particular would predict,
against a 99.91%-confident classifier.

Honest caveats on that statement: real spectra are not flat, so the
flat-spectrum comparison is a scale reference and not a proof the prediction is
wrong; and the classifier legitimately keys on whatever separates the classes
in the normalised spectrum, which need not be the defect frequency. What it
does show is that the *physical* corroboration for this particular call is
weak — and the UI surfaces it in an amber note rather than hiding it.

**A related bug this finding exposed.** The Frequency Lab initially reused a
spectrum helper that crops at 1 kHz. That crop would have hidden the dominant
peak entirely and made "overall peak" a statement about the crop rather than
the signal. Caught by a test using a synthetic 1,500 Hz tone that peaked at
0.0 Hz. The endpoint now runs to Nyquist, which reveals a resonance cluster
around 3,000–3,500 Hz that the cropped view never showed.

---

## 7. The agentic LLM layer

### The split, and why

The agent chain itself is **deterministic rule-based Python** — a LangGraph
`StateGraph` running Monitor → Diagnostic → Risk → Maintenance. Those four
functions produce the risk score that flips equipment into `maintenance` and
files a service request, and that must stay reproducible, inspectable and
testable.

So the LLM sits strictly **on top** and never originates a number. It writes
the prose a technician reads; the deterministic layer decides.

### How the grounding is enforced

Four mechanisms, not one:

1. **Tools are the only source of numbers.** `agents/tools.py` exposes four
   callable tools, each a thin read-only wrapper over code that already ran:
   `get_signal_statistics` (→ `_signal_stats`), `get_risk_assessment` (→
   `risk_agent` + `maintenance_agent`, returning the *stored* risk score
   alongside the recomputed one so a divergence would be visible),
   `get_bearing_frequency_match`, and `measure_spectral_band_energy(low, high)`
   — the one tool taking real arguments, so the model can check whether energy
   actually sits on the defect band rather than take the match on faith.
2. **The prompt carries no measurements.** Only identifiers — request id,
   equipment name, site, window id, sensor, predicted class. The model must
   call a tool to learn any figure. A test (`test_prompt_carries_no_measurements`)
   pins this, because the guard below is meaningless without it.
3. **A post-hoc numeric check.** `verify_numbers_are_grounded()` scans the
   returned prose for any number absent from every tool result and **stores**
   what it finds rather than rejecting the narrative — so the guard's own track
   record is inspectable.
4. **The system prompt forbids revision.** The model may explain the risk score
   and urgency; it may not contradict them.

### The guard has fired twice on real output — both times correctly

| Attempt | What the model wrote | Verdict |
|---|---|---|
| First live call | `0.0325 %` | It multiplied the tool's `0.000325` by 100. Derived, not hallucinated — but still the model originating a figure. **Caught.** |
| Current demo narrative | `260–285 Hz` | It rounded the tool's `258.23`. **Caught.** |

The first was fixed by **tightening the prompt** against unit conversion rather
than loosening the check. The second is deliberately left in place: a guard
that visibly fires during a demonstration is better evidence that it works than
a clean run would be.

**One limitation the guard cannot catch.** In one generation the model described
263.67 Hz as "far below the expected 271.82 Hz" — a 3% difference, comfortably
inside the ±5% band the tool itself reports. The numbers were correct; the
adjective overstated them. The guard constrains figures, not characterisations,
and nothing in this design claims otherwise.

### Provider: Groq, and why that is incidental

Current provider is **Groq**, model **`openai/gpt-oss-120b`**, cached per
service request in the `llm_narratives` table so a demo costs one call per
flagged machine rather than one per page view.

The layer was originally built against the Anthropic SDK targeting Claude. The
available API key turned out to be a Groq key (`gsk_` prefix), so it was
rewritten. **The grounding design did not change at all** — four deterministic
tools, a measurement-free prompt, a post-hoc numeric check — because none of it
depends on the provider. Only the call shape changed.

One provider-specific constraint, found by probing the live API rather than
assumed: Groq returns `400 json mode cannot be combined with tool/function
calling`. The loop therefore runs in two phases — tools first with no
`response_format`, then one final tool-free call with
`response_format={"type":"json_object"}`. A test asserts no single request ever
carries both, so the workaround cannot be undone by accident. Because JSON mode
guarantees *valid* JSON rather than *schema-conforming* JSON, the response keys
are validated in `_parse_final`.

With no key configured the endpoint returns a clear **503** and nothing else
changes: predictions, risk scores, equipment flagging and auto-filed requests
are all unaffected.

---

## 8. Authentication and the three-role scoping model

JWT bearer authentication (HS256, PyJWT), bcrypt password hashing, migration
`0006`. **Every route requires a token except `POST /auth/login` and the
unauthenticated `GET /livez` liveness probe** — including `/health`, which
reports database state and is therefore protected like everything else.

| Role | Scope |
|---|---|
| `customer` | Only its own equipment and service requests, at any site |
| `operator` | Only its own site's equipment, queue and dispatches |
| `admin` | Everything, including the replay controls |

### The design point

Authentication, authorization and **row scoping** are kept as three separate
things:

- `get_current_user` re-reads the user row on **every request**, so deactivating
  an account or changing a role takes effect immediately rather than when the
  token happens to expire.
- `require_roles` refuses a role outright.
- `scope_by_customer` / `scope_by_depot` return SQLAlchemy conditions that each
  endpoint composes into its own query.

> **The scope is always applied before the caller's own query parameters and is
> derived from their user row.** So `?customer_id=…` or `?depot_id=…` can only
> ever *narrow* a result set, never widen it past the caller's scope.

A row outside scope returns **404, not 403**, so a customer cannot enumerate
another tenant's equipment ids by reading status codes.

A CHECK constraint makes the role/scope pairing a **database invariant**: an
operator with no site, or an admin carrying one, cannot be inserted at all.

### Verified, not asserted

Live `curl` matrix against the running API, 8 equipment across 4 sites:

| | admin | operator.a | operator.c | customer Acme | customer Northwind |
|---|---|---|---|---|---|
| `GET /equipment` | 8 | 2 | 2 | 4 | 2 |
| `GET /work-orders` | 3 | 2 | 1 | 403 | 403 |
| `GET /replay/status` | 200 | 403 | 403 | 403 | 403 |

Scope-widening attempts return **0 rows**: `?depot_id=3` as the Plant A
operator, `?customer_id=1` as Northwind. A token with one character changed
returns `401 invalid token: Signature verification failed`.

The strongest test, `test_every_route_refuses_an_anonymous_caller`, walks the
FastAPI application's **own route table** rather than a hand-written list — so
a router added later without a guard fails the suite instead of shipping open.

The frontend guard is documented in its own docstring as a **usability guard,
not a security one**: deleting it would make the app ugly, not insecure.

---

## 9. What is real vs. what is simulated

Stated as plainly here as the project states it internally.

### Real

- **The dataset.** 4,367 windows from 64 real CWRU recordings. No synthetic
  data anywhere except the documented GAN augmentation step below.
- **The classifier and its accuracy.** 53.8–69.5% (mean 60.38%) on an
  experiment-level split — three classes, against a 50.0% majority-class floor.
- **The federated result**, including its failure (§3.2).
- **The predictions and decisions.** 4,367 of each, regenerated from the current
  models and verified by direct SQL, not app-level reads.
- **The incremental processing.** Each replay tick genuinely predicts, decides
  and syncs over windows the pipeline had not seen.
- **The bearing physics.** Published multipliers, real shaft speeds.
- **The LLM call.** A real API call with a real tool-use loop and a real audit
  trail.
- **The access control.** Enforced server-side and verified by calling the API
  directly as each role.

### Simulated — and labelled as such in the UI

- **The four "sites" are four CWRU motor load levels.** A real site could host
  machines at several loads; these cannot. Display names keep the mapping
  visible: *Plant A (CWRU 0 HP)*.
- **Sensor arrival is a replay.** `released_at` marks simulated arrival over a
  static archive. There are no sensors. The panel says so.
- **GPS / location fields on `Equipment` and `WorkOrder` are invented.** CWRU
  has no location telemetry. Never implied as real.
- **`Equipment.linked_experiment_id` is a simulated association** between a
  product row and a bench recording. This is the largest open modelling gap.
- **Demo credentials are published** in the README on purpose, so a reviewer can
  verify the access restrictions themselves.

### Known limitations

1. **The GAN augmentation has an unresolved domain mismatch.** TimeGAN
   generates *time-domain* synthetic windows, but since the §2.2 fix the
   classifier trains on their *spectrum*. A window can be plausible in time and
   unrealistic in frequency — the likely reason augmentation sometimes hurts
   accuracy. Documented, not fixed.
2. **Federated accuracy is a negative result** (§3.2), presented as a finding.
3. **Defect-band corroboration is often weak** (§6.4), surfaced rather than
   hidden.
4. **Auth is demo-grade.** The JWT lives in `localStorage` rather than an
   httpOnly cookie, because the API is a separate origin and a cookie would
   need `SameSite=None; Secure`, credentialed CORS and TLS. Every restriction
   is still server-side, so a stolen token cannot read another tenant's rows —
   but there is no refresh rotation, rate limiting or password reset.
5. **Docker is unverified end to end.** `docker compose config` resolves
   cleanly, but the Docker daemon is unreachable on the development laptop
   (WSL2/Hyper-V are blocked by policy), so `docker compose up` has never been
   run. Stated rather than implied.
6. **No `FleetMind_Literature_Review.md` exists.** `CLAUDE.md` referred to a
   six-paper literature review at this path; no such file is present in the
   repository or anywhere in its 44-commit history. The reference was stale and
   has been corrected. The review is outstanding work, not a missing file.

---

## 10. Verification status

As of 2026-10-03, commit `74fedc61`:

| Check | Result |
|---|---|
| Backend test suite | **257 passed** |
| `ruff check .` | clean |
| `ruff format --check .` | clean |
| `mypy fleetmind` | clean, 64 source files |
| `npx tsc --noEmit` | clean |
| `npm run build` | clean, 7 static routes |
| `docker compose config` | resolves (5 services) |
| Migration chain from empty | all 6 applied against real PostgreSQL; resulting table set identical to production |
| Browser walkthrough | 15 page-visits across 5 roles, **zero console errors** |

### Testing approach worth noting

Three tests exist specifically because a behavioural test could not have caught
the bug:

- **`test_dependencies.py`** AST-scans every import against `pyproject.toml`. It
  caught that `langgraph` was declared *nowhere* and `torch` only under an
  optional extra, while CI installed neither — meaning a clean clone could not
  start the application. Everything passed locally only because the development
  venv had the packages from earlier manual work.
- **`test_every_route_refuses_an_anonymous_caller`** enumerates the app's own
  route table, so new routes are closed by default.
- **`test_json_mode_is_never_combined_with_tools`** pins a provider workaround
  that would otherwise be silently undone.

---

## 11. Architecture summary

```
backend/fleetmind/
  ingestion/    CWRU download, manifest, windowing, ingest
  db/           SQLAlchemy models, session, seed scripts
  api/          FastAPI app + 11 routers
  auth/         bcrypt passwords, JWT tokens, role + row scoping
  eda/          exploratory data analysis reports
  ml/           model, dataset, train, evaluate, predict, bearing_freq
  augmentation/ GAN-based rare-class augmentation
  federated/    Flower FedAvg/FedProx simulation
  agents/       LangGraph chain, sync_equipment_status, tools.py, llm.py
  replay/       engine + background worker
  pipeline.py   predict → agents → sync, one code path
frontend/src/   Next.js 14 App Router, TanStack Query, Recharts, Tailwind
```

**Migrations:** `0001` core schema · `0002` product tables · `0003` AI-request
partial unique index · `0004` `released_at` arrival marker · `0005`
`llm_narratives` · `0006` `users`.

**Database:** 12 tables. PostgreSQL 18, native (not Docker — WSL2/Hyper-V are
blocked on the development machine).

---

*Report sourced from `CLAUDE.md`'s Progress log and all 44 commits, with every
quoted figure re-verified against the live system on 2026-10-03.*
