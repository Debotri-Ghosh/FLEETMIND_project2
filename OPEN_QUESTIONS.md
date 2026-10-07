# Open questions — decisions taken with a reasoned default

Per working rule 5 of `docs/plans/2026-09-21-fleetmind-implementation/00-overview.md`:
genuine product or design decisions are written here with a reasoned default, and
work proceeds on that default rather than stopping. Each entry says what was chosen,
why, and what it would take to change it. **Any of these can be overridden by the
project owner; none of them is hidden in code without being written down here.**

Opened 2026-10-08 for Phases 3–5 of
`docs/plans/2026-10-06-decision-intelligence/PLAN.md`.

---

### Q1. Which posterior does the decision policy consume?

**Default: the deployed v1 checkpoints, uncalibrated.**

- The owner decided on 2026-10-07 to keep the deployed v1 checkpoints and not retrain
  over them, so v1 is the model the live pipeline runs.
- Phase 2 measured temperature scaling and found it made test calibration **worse**
  (mean ECE 0.1082 → 0.1172 for v2-last), with the fitted temperature disagreeing
  about the direction of miscalibration across sites. Shipping it would make things
  worse on the evidence.
- **Consequence, stated rather than hidden:** v1 is over-confident (mean signed
  over-confidence +0.2101, ECE 0.2146 on held-out data). An expected-cost decision is
  only as good as the probabilities it is given. This is the policy's main weakness.

To change: swap the checkpoint path, or re-run Phase 2 with more calibration data.

### Q2. What does the cost matrix depend on?

**Default: healthy vs. faulted only — not the specific fault class.**

The heuristic baseline scored faults with an invented severity table (Ball 0.4,
InnerRace 0.7, OuterRace 0.85). Nothing in CWRU says how a ball fault and an
outer-race fault differ in time-to-failure, so a per-class cost would be the same
kind of invented number this work exists to remove. The policy uses only
`P(faulted) = 1 − P(Normal)`, which needs **no per-fault constant at all**.

To change: supply per-class consequence data (e.g. measured time-to-failure) and
extend `costs.py` to a 4-state matrix.

### Q3. What units are costs in?

**Default: relative cost units, normalised so an unplanned run-to-failure costs 100.**

An expected-cost decision depends only on the **ratios** between costs — multiplying
every entry by a constant changes no decision. So no currency or downtime figure is
needed, and none is invented. This also keeps the project clear of any ROI claim,
which `PRESENTATION_NUMBERS.md` forbids. The defaults are stated in `costs.py`, one
comment per entry, and every Phase 3 report shows how decisions move as the key
ratio is varied across two orders of magnitude.

To change: edit `costs.py`, or pass a different matrix to the policy.

### Q4. How is a false-alarm rate measured, when no Normal recording is held out?

**Default: cross-site Normal recordings.**

Each site has one Normal recording, kept in that site's training set. But site *k*'s
model has **never seen** site *j*'s Normal recording (*j* ≠ *k*). So each model is
scored against the other three sites' healthy recordings: 4 models × 3 = **12
genuinely unseen healthy-machine evaluations**. This is the first false-alarm
measurement this project has been able to make.

**Caveat:** a different motor load is a domain shift, so this probably overstates the
false-alarm rate a site would see on its own healthy machines. It is reported as
"cross-site", never as plain false-alarm rate.

### Q5. What are the SPRT error targets?

**Default: α = 0.01 (false alarm), β = 0.05 (miss).**

A false alarm sends a technician to a healthy machine; a miss lets a fault keep
running. The default makes false alarms the rarer error, as in most condition
monitoring. **Caveat:** Wald's guarantees assume independent observations, and
windows cut from one continuous recording are strongly autocorrelated, so the
achieved rates are expected to differ from the targets — they are measured and
reported, never asserted.

### Q6. Does the new policy replace the heuristic in the live pipeline?

**Default: yes, behind a setting, with the heuristic one switch away.**

`CLAUDE.md` already names the decision layer as authoritative. The pipeline switches
to it, and `decision_policy = "heuristic"` restores the old behaviour exactly, so the
documented batch figures stay reproducible. As with the 2026-10-01 load-factor fix,
this **changes which machines get flagged**; the before/after is measured and
recorded, not assumed.

### Q7. What is done with the Phase 2 techniques that did not help?

**Default: not shipped. Recorded as tested-and-rejected, not quietly dropped.**

PLAN.md Phase 5 lists exposing "the calibrated posterior and the conformal set". Phase 2
measured both and found neither helped on this data. Wiring them in would put two
techniques the project's own evidence argues against into the product. The plan item
is closed as *rejected on evidence*, with the evidence linked.

### Q8. Where are the full class probabilities stored?

**Default: a new nullable JSON column `model_predictions.class_probabilities`
(migration `0007`).**

The policy needs `P(Normal)`; today only the argmax label and its confidence are
stored. Nullable, so all 4,367 existing rows stay valid and the heuristic path is
unaffected. It also lets the evidence panel show the full distribution rather than
one number.
