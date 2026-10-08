# FleetMind — demo script

A click-through for the thesis defence. Roughly 10 minutes if you stop at the
end of §4; §5–§7 are there for whichever questions actually come.

Everything below has been run end to end; the figures are from the live
database, not from notes. Where a number is weak, this script says so and
tells you to raise it yourself — see §6, which is the part worth rehearsing.

---

## 0. Before they walk in

```bash
# backend (from backend/, venv active)
python -m uvicorn fleetmind.api.main:app --host 127.0.0.1 --port 8000

# frontend (from frontend/)
npm run dev
```

Then open **http://localhost:3000** and sign in as
`admin@fleetmind.example.com` / `admin-demo-2026`.

Two things to check before you start talking:

1. The **Live Automation** panel at the top says `4,367 / 4,367` and
   *archive fully replayed*. That is the finished state.
2. Press **Reset** once. It takes a second and drops everything back to
   `0 / 4,367`. **Do this before the demo, not during it** — reset deletes the
   predictions, decisions and AI-filed requests, and the LLM write-ups go with
   them. You will regenerate one in §4.

If you would rather not reset live, skip §1 and narrate the finished board
instead; everything else works either way.

---

## 1. Automation, where they can see it (dashboard, 2 min)

Open on the **Dashboard**. The Live Automation board is the first thing on the
page, deliberately — it used to be buried in the operator console.

**Press "Start live replay".** Then stop talking for about five seconds and
let them watch. Within three seconds:

- the counter climbs past `25`, `50`, `100` and the progress bar fills;
- the amber dot next to **Live Automation** starts pulsing and the label flips
  from `idle` to `live`;
- an **equipment card flashes amber** the moment the pipeline changes its
  status — measured at **+3.0 s** in testing (Bearing Motor A1 → maintenance);
- a **new AI-flagged request animates in** with a `NEW` badge — measured at
  **+3.3 s** (request #49, OuterRace, risk 66%).

Say, while it runs:

> Nothing here is on a timer pretending to be work. Each tick releases a batch
> of archived windows, runs the classifier on them, runs the agent chain over
> the predictions, and syncs equipment status. The cards change because the
> database changed.

**Then say the honest part before anyone asks it:**

> This is a replay of a recorded archive, not live telemetry. CWRU is a static
> dataset — there are no sensors attached. `signal_windows.released_at` marks
> *arrival*, and the replay engine progressively releases rows as if they were
> streaming in. The UI says so on the panel itself.

Press **Pause** when you have made the point. You do not need to wait for it
to finish — but if you want the full board for §3, press Start again and leave
it running while you talk through §2.

---

## 2. The Frequency Lab (dashboard, 2 min)

Scroll to **Frequency Lab**. It opens on the **time domain** — a raw vibration
waveform, nothing else.

**Press "Apply Fourier Transform ▶".** The chart dims, rescales and becomes the
spectrum *in the same box*.

The point to make, in these words or close to them:

> That is the same 2,048 samples. Nothing new was collected and no second
> measurement was taken — the frequency view is the waveform re-expressed. I
> made it transform in place rather than appear in a second panel, because a
> second panel would imply a second measurement.

Then point at the three tiles that appear underneath:

| Tile | What to say |
|---|---|
| **Defect band** | Bearing geometry — the frequency a Ball defect produces at this shaft speed. Computed from the SKF 6205 multipliers, not learned. |
| **Energy in band** | How much of the spectrum actually sits there. |
| **Overall peak** | Where the energy *really* is. |

**Say this out loud, do not wait to be asked:**

> The classifier never sees these frequencies. It consumes a 512-bin
> log-scaled normalised FFT magnitude. This band is independent corroboration a
> human can check — it is not an explanation of the model.

And if the amber warning is showing (it usually is):

> On this window the dominant energy is at ~3,369 Hz, nowhere near the defect
> band. The physics does not corroborate the label here, and the view says so.
> I would rather show that than crop the chart to hide it — which is what the
> earlier version did, by only plotting up to 1 kHz.

---

## 3. Role-based access, proved rather than claimed (1 min)

Press **Sign out**, then sign in as
`operator.c@fleetmind.example.com` / `operator-demo-2026`.

- The nav loses the **Customer** tab and gains `Site operator · Plant C`.
- The replay controls are gone — replay rewrites every site's data, so it is
  admin-only.
- Go to **Site Operations**: the Equipment Board shows **2 machines**, both at
  Plant C. The site filter says "My site" and offers nothing else.

Then say:

> This is not hidden UI. The API refuses it. If I pass `?depot_id=1` as this
> operator I get zero rows, not Plant A's machines, because the scope is read
> from the user row and applied before any query parameter the client sends.
> `backend/tests/test_auth.py` asserts that directly, and one of its tests
> walks the application's own route table so a route added later without a
> guard fails the suite instead of shipping open.

If they want it demonstrated rather than asserted, this is the one-liner:

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"operator.c@fleetmind.example.com","password":"operator-demo-2026"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -s "localhost:8000/equipment?depot_id=1" -H "Authorization: Bearer $TOKEN"   # []
curl -s  localhost:8000/replay/status        -H "Authorization: Bearer $TOKEN"    # 403
curl -s  localhost:8000/equipment                                                  # 401
```

---

## 4. Evidence behind an AI decision (operator console, 3 min)

Sign back in as **admin**. Go to **Site Operations** → **AI-Flagged Requests**
→ press **Show evidence** on any row.

### 4a. Ground truth vs. prediction — the first thing on the panel

Two deliberately unlike panels:

- **left, dashed and grey** — what the CWRU researchers machined into the
  bearing: fault type, defect diameter, sensor position, motor load, and the
  source `.mat` file. An archival record.
- **right, coloured and solid** — what the classifier said, with its
  confidence and checkpoint.

> The model is blind to the left panel. **Read the note under the right panel
> before saying anything about generalisation**: machines A1, B1, C1 and D1 run on
> recordings held out of their site model's training split; A2, B2, C2 and D2 run
> on recordings it *trained on*, and the panel says so. Agreement on those is
> in-sample. (Corrected 2026-10-08: this script used to claim every recording was
> held out.) The verdict
> strip underneath says Agreement or Disagreement, and it is just as prominent
> when the model is wrong — a panel that only shows matches is evidence of
> nothing.

### 4a′. The maintenance decision

Under the ground truth, **Maintenance decision** shows what was decided and why:

- the chosen action and P(faulted) = 1 − P(Normal), read from the classifier's
  full probability over all four classes (shown);
- the expected cost of every action in relative units (run-to-failure = 100) —
  point out that the cheapest wins, and that "stop machine" is marked *never
  optimal* because pricing it needs a degradation rate CWRU does not have;
- **Why the machine was flagged**: the sequential test, e.g. "flagged after 2 of
  59 windows, evidence 6.46 against boundary 4.55";
- one line with the heuristic baseline's verdict on the same window, labelled as
  comparison only. On Bearing Motor A2 the baseline says MEDIUM — that is why the
  old rule never flagged it.

> Say: "Every threshold here is derived from stated costs, not tuned. On
> held-out data this ties the old rule — zero missed faults, zero cross-site false
> alarms — because fault *presence* is easy on CWRU. What it changes is that the
> model's probability now actually reaches the decision."

### 4b. The signal and its spectrum

Same two plots as the Frequency Lab, bound to the window this decision was
actually made from. The RMS and peak are **recomputed live** by the same
function that produced the stored decision — the panel says so.

### 4c. The LLM write-up

Scroll to **Engineer's write-up**. If the panel says none has been generated,
press **Generate write-up** — it is a real API call to
`openai/gpt-oss-120b` and takes a few seconds.

> The model gets no raw signal and no arithmetic. It can only call four tools
> that wrap the same deterministic functions that produced the decision, and
> the prompt it is handed contains no measurements at all — only identifiers.
> So every figure in that paragraph had to come back from a tool.

**Then open the "Tool calls behind this write-up" disclosure** and show the raw
JSON. That is the audit trail.

**And if the amber grounding warning is showing — point at it, do not skip it:**

> That is the guard working. On this one the model wrote "260–285 Hz" when the
> tool returned 258.23. It rounded. That is not a fabrication, but it is still
> the model originating a figure, and the check caught it and stored it rather
> than letting it through. An earlier run converted 0.000325 into "0.0325 %"
> and was caught the same way; I tightened the prompt instead of loosening the
> check.

---

## 5. If they ask about the ML itself

Dashboard → **Model Performance: Local vs. Federated**.

| | |
|---|---|
| Per-site accuracy, **deployed checkpoints** | 53.8% / 69.5% / 61.0% / 57.2% — mean **60.4%** |
| Majority-class floor (always OuterRace) | **50.0%** |
| Mean local baseline, federated study | 60.0% |
| FedProx, 30 rounds, mu=0.1 | 42.9% |

> Say the floor out loud. OuterRace is exactly half the held-out set, so always
> guessing OuterRace scores 50%. Plant A's deployed model beats that by 3.8
> points; Plant B by 19.5. And it is a **three-class** number — each site has one
> Normal recording and the split keeps it entirely in training, so no healthy
> machine is ever held out. The false-alarm rate on a healthy machine is
> therefore **unmeasured**, not zero.

> The federated result is *worse* than local training, and that is the real
> number. It is a documented phenomenon under non-IID per-site class imbalance,
> it is in the literature review, and plain FedAvg was worse still — it
> oscillated round to round, which is why FedProx is the primary result rather
> than FedAvg.

Three bugs worth mentioning if the accuracy numbers are questioned, because all
three *lowered* them:

1. **Data leakage.** The original split was per-window, so windows from the
   same recording landed on both sides and accuracy read as a fake 100%. Fixed
   to an experiment-level split.
2. **InnerRace was unlearnable.** The CNN's `AdaptiveAvgPool1d` collapsed the
   whole time-domain window into one global average, discarding the
   fault-repetition rate that separates InnerRace from OuterRace — the
   confusion matrix showed InnerRace 100% misclassified as OuterRace at every
   site. Fixed by feeding a 512-bin FFT magnitude instead of raw samples.
3. **The saved checkpoint was the last epoch, not the best one.** `metrics.json`
   records `best_val_acc` — the best accuracy any epoch *reached* — but
   `ml/train.py` calls `torch.save()` after the loop with no best-epoch guard.
   So the earlier 62.7–72.5% figures described models that were never written to
   disk. The checkpoints that are actually deployed score a mean 6.0 points
   lower, which is why §5 now quotes 53.8–69.5%. **If you are asked why the old
   number is gone, this is the answer — and it is a strong one**: we measured the
   model that is running, not the one the log remembered. The checkpoints were
   deliberately left in place rather than retrained, because all 4,367 stored
   predictions and decisions came from them.

---

## 6. The caveats — say these before you are asked

This is the section to rehearse. Volunteering a limitation is worth more than
surviving a question about it.

**1. The 91.4% agreement figure is in-sample. Do not quote it as accuracy.**

The database holds 4,367 predictions and 3,993 match the documented label —
91.44%, with 374 disagreements (mostly InnerRace→Ball, 163, and
Ball→OuterRace, 133). But each site's model predicted over *all* of its own
windows, including ones it trained on. **The honest generalisation numbers are
the 53.8–69.5% (mean 60.4%) accuracies in §5, three-class, against a 50.0%
floor.** If someone quotes 91% back at you as model accuracy, correct them.

**1a. The decision layer ties the old rule on held-out data; it does not beat it.**

Both miss 0 of 16 held-out faults and raise 0 of 12 cross-site false alarms. On
the live demo the policy flags 8 machines to the old rule's 6, but both extra
machines run on training recordings — in-sample, not an improvement. The
probabilities feeding the policy are uncalibrated (ECE 0.2146), and calibration
was tried and made it worse.

**2. The GAN augmentation has a known mismatch, unfixed.**

TimeGAN generates *time-domain* synthetic windows, but since the FFT fix the
classifier trains on their *spectrum*. A window can be plausible in time and
unrealistic in frequency, which is the likely reason augmentation sometimes
hurts accuracy. It is documented as a limitation rather than quietly dropped.

**3. The defect-frequency corroboration is often weak, and the UI shows it.**

On the sampled windows the dominant spectral energy does not sit on the defect
band. Say what that does and does not mean: it weakens the *physical*
corroboration of that particular call; it does not invalidate the classifier,
which keys on whatever separates the classes in the normalised spectrum. The
evidence layer exists precisely to surface this.

**4. Several things are simulated, and labelled as such.**

The four "sites" are four CWRU motor load levels. GPS on equipment and work
orders is invented — CWRU has no location data. `Equipment.linked_experiment_id`
is a simulated association between a product row and a bench recording. The
replay is a replay.

**5. Auth is demo-grade.**

The JWT lives in `localStorage`, not an httpOnly cookie, because the API is a
separate origin. Every restriction is enforced server-side, so a stolen token
still cannot read another tenant's rows — but there is no refresh rotation, no
rate limiting, no password reset, and the demo passwords are published in the
README on purpose.

---

## 7. Numbers you may be asked for

| | |
|---|---|
| Signal windows | 4,367 (2,048 samples each, 12 kHz, drive end) |
| Predictions / agent decisions | 4,367 each |
| Equipment flagged by the pipeline | 6 of 8 |
| AI-filed service requests | 6 |
| Backend tests | 257 passing |
| Migrations | 6, verified by running the whole chain from empty against real Postgres |
| Classifier input | 512-bin FFT magnitude, log-scaled, z-normalised |
| Agent chain | LangGraph: Monitor → Diagnostic → Risk → Maintenance (rule-based) |
| LLM layer | `openai/gpt-oss-120b` via Groq, tool-grounded, prose only |

**If asked why Groq and not Claude:** the integration was built against the
Anthropic SDK first; the available key turned out to be a Groq key, so it was
rewritten for Groq. The grounding design is provider-independent — four
deterministic tools, a measurement-free prompt, and a post-hoc numeric check —
and only the call shape changed.

---

## 8. Recovering if something breaks

| Symptom | Fix |
|---|---|
| Every page says "Failed to load" | The backend died. Restart uvicorn; the pages have Retry buttons. |
| Pages 404 on their own JS | `npm run build` was run while `npm run dev` was live. Stop dev, `rm -rf .next`, restart. |
| "Start live replay" is greyed out | The archive is fully replayed. Press **Reset** first. |
| Write-up returns 503 | `GROQ_API_KEY` is missing from `backend/.env`. Everything else still works — the rule-based decision does not depend on it. |
| Backend changes seem ignored | uvicorn runs without `--reload`. Restart it. |
