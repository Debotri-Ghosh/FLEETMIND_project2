# Frequency analysis — explained for the defence

**A study guide, not a technical reference.** Every number in this document came
from running this project's own code on 2026-10-03. Nothing here is a
hypothetical example. If you quote a figure from this file, it is defensible.

The question your reviewer asked — *"the dataset is static, so how can there be
frequency data?"* — has a clean one-sentence answer, and the rest of this
document is there so you can defend that sentence from any angle:

> **Only one thing was ever physically recorded: vibration amplitude over time.
> Frequency was never measured — it is calculated afterwards from that same
> recording, using mathematics.**

---

## Q1: "The dataset is static — how is there frequency data at all?"

### The short answer

There is no separate frequency measurement, and there never was. The CWRU rig
had **one accelerometer** bolted to the motor housing. It recorded **one single
quantity**: how much the housing was shaking, 12,000 times per second.

That is the entire physical measurement. One number, repeated 12,000 times a
second.

Everything about "frequency" in this project is **computed from those numbers
afterwards** by a mathematical operation called the Fourier Transform. No extra
sensor, no extra recording, no extra data collection. The same numbers,
re-described.

### The analogy to use out loud

Think of a piano chord.

You can describe that chord in two completely different ways:

1. **As air pressure over time** — "the air pressure went up, down, up, down,
   in this exact wiggly pattern for two seconds." That is a recording. It is
   what a microphone captures.
2. **As notes** — "that was a C, an E and a G played together."

Here is the key point: **nobody measured the notes separately.** There was no
second microphone that recorded "notes". The notes were *worked out* from the
pressure wiggle, by mathematics. Both descriptions contain exactly the same
information — one is just far more useful to a musician.

Our accelerometer recorded the wiggle. The Fourier Transform tells us the
notes. A bearing with a defect in it has a characteristic "note", and that is
what makes the whole thing diagnosable.

### If they push: "so you generated data that wasn't there?"

No — and this is the important distinction to hold onto.

Nothing is created. The Fourier Transform is **lossless and reversible**: you
can transform the waveform into the spectrum and transform it straight back and
get the original samples. If it were inventing information, that would be
impossible.

It is a change of description, not an addition of content. Exactly like saying
"half" instead of "0.5".

### Where this is visible in the system

The **Frequency Lab** on the dashboard was built specifically to make this
undeniable on screen. It shows the raw waveform first, on its own. You press a
button. The *same chart* — same panel, same box — morphs into the spectrum.

It is staged that way deliberately: if the spectrum simply appeared in a second
panel next to the first, it would look like a second measurement. Transforming
the chart that is already on screen shows that it is the same recording, seen
differently. The copy on the panel says it in words too:

> *"Frequency content is computed from this exact recording via Fourier
> Transform — nothing new is collected."*

**Show them this. It answers the question better than any sentence you can
say.**

---

## Q2: "What exactly gets computed, step by step, from the raw numbers?"

This is `to_fft_magnitude()` in `backend/fleetmind/ml/model.py`. Here is what it
actually does, in order, with this project's real numbers.

### Step 0 — what we start with

| | |
|---|---|
| Samples in one window | **2,048** |
| Sampling rate | **12,000 Hz** (12,000 readings per second) |
| So one window covers | 2048 ÷ 12000 = **0.1707 seconds** (170.67 ms) |

One window is about a sixth of a second of shaking. At 1,730 rpm the shaft turns
roughly **5 times** in that window — so a defect that strikes once per
revolution gets caught several times. That is why the window length is a
reasonable choice, and it is a good thing to be able to say.

Real values from window #2713 (the one used throughout this guide):

```
first 5 raw samples : [0.10542, -0.10737, -0.16341, 0.118903, 0.184039]
range               : -0.4629 to +0.4759
```

Those are acceleration values in **g**. That is the whole raw recording: a list
of 2,048 numbers like these.

### Step 1 — the Fourier Transform (`np.fft.rfft`)

The FFT asks one question, over and over:

> *"How much of a pure wave at frequency f is present in this signal?"*

It answers that for a whole ladder of frequencies at once, and returns a
magnitude for each — how strongly that frequency is present.

`rfft` is the "real" FFT. Our data is real numbers (not complex), and that makes
the output symmetric, so it returns only the useful half:

| | |
|---|---|
| Input | 2,048 samples |
| Output | **1,025 values** (2048 ÷ 2 + 1) |
| Highest frequency it can describe | **6,000 Hz** (the Nyquist limit, half the sampling rate) |

**Nyquist in one line, if asked:** you need at least two samples per cycle to
see a wave at all, so sampling at 12,000 Hz you can never resolve anything above
6,000 Hz. That is physics, not a design choice.

### Step 2 — the bins, and the real resolution

The 1,025 outputs are evenly spaced "bins" from 0 Hz up to 6,000 Hz. The spacing
is the number you should memorise:

```
bin width = sampling rate ÷ number of samples
          = 12,000 ÷ 2,048
          = 5.859375 Hz
```

So bin 0 is 0 Hz, bin 1 is 5.86 Hz, bin 2 is 11.72 Hz, and so on. **Our
frequency resolution is about 5.86 Hz** — we cannot distinguish two frequencies
closer together than that.

### Step 3 — keeping 512 bins

The function keeps only the **first 512 bins**:

```
512 × 5.859375 = 3,000 Hz
```

So the model sees **0 to 3,000 Hz**, and the top half of the spectrum
(3,000–6,000 Hz) is discarded.

**Why 512 is enough** — this is the justification to have ready. Every bearing
defect frequency in this project is **under 300 Hz**. At Plant D they land here:

| Frequency | Hz | Which of the 512 bins |
|---|---|---|
| FTF | 11.48 | bin 2 |
| BPFO | 103.36 | bin 18 |
| BSF | 135.91 | bin 23 |
| BPFI | 156.14 | bin 27 |
| 2×BSF | 271.82 | bin 46 |

Everything diagnostically relevant sits in roughly **the first 46 of 512 bins**.
Keeping 512 gives the model the defect region *and* a large margin of context
above it, at a quarter of the data.

### Step 4 — log scaling (`np.log1p`)

Raw FFT magnitudes are wildly uneven. In window #2713:

```
raw magnitudes range from 0.0099 to 47.5504   -> a factor of about 4,800
```

A neural network handles that badly: the huge values dominate and the small ones
get ignored, even when the small ones matter.

`log1p(x)` computes log(1+x), which squashes that range:

```
after log1p: 0.0670 to 3.8826   -> a factor of about 58
```

The loud parts are still the loudest — **the ordering is preserved** — but the
quiet parts are now visible to the network instead of being numerically
drowned. (`log1p` rather than plain `log` because log(0) is undefined and a bin
can legitimately be zero.)

### Step 5 — z-normalisation

Finally, subtract the mean and divide by the standard deviation:

```
spectrum = (spectrum - mean) / std
```

Verified on the real output for window #2713:

```
final mean = 1.18e-16  (that is zero, to floating-point precision)
final std  = 1.000000
final range = -1.6777 to +4.1290
```

**Why this is needed:** a heavily loaded motor vibrates harder overall than a
lightly loaded one. Without this step the network could "cheat" by learning
*loudness* — effectively guessing the site rather than the fault. Normalising
every window to mean 0 and standard deviation 1 removes absolute loudness and
forces the model to learn from the **shape** of the spectrum: which frequencies
are prominent *relative to each other*. That is the actual fault signature.

### The whole pipeline in one line

> 2,048 raw samples → FFT → 1,025 magnitudes at 5.86 Hz spacing → keep the first
> 512 (0–3,000 Hz) → log-scale → normalise to mean 0, std 1 → that 512-number
> vector is what the CNN sees.

---

## Q3: "Where do BPFI/BPFO/BSF/FTF come from — are they read from the data?"

### The short answer

**No. Absolutely not. Zero data from the recording is used.**

They are calculated **in advance**, from two things only:

1. the **physical geometry of the bearing** (fixed, published by the
   manufacturer), and
2. the **shaft speed** (how fast the motor turns).

You could compute them with a calculator before the rig was ever switched on.

### What each frequency physically means

A rolling-element bearing is an inner ring, an outer ring, and a set of balls in
a cage between them. When a ball rolls over a defect, it produces an impact. How
*often* those impacts happen depends purely on geometry and rotation speed:

| Name | Stands for | The physical event |
|---|---|---|
| **BPFO** | Ball Pass Frequency, Outer race | A ball passes over a pit on the **stationary outer** ring |
| **BPFI** | Ball Pass Frequency, Inner race | A ball passes over a pit on the **rotating inner** ring |
| **BSF** | Ball Spin Frequency | How fast an individual **ball** spins on its own axis |
| **FTF** | Fundamental Train Frequency | How fast the **cage** holding the balls rotates |

A defect in one of those locations produces impacts at *that* rate. So the
location of the damage determines the frequency of the knocking. That is the
whole physical basis of this field.

### The standard formulas

The textbook formulas depend on the number of balls (n), ball diameter (d),
pitch diameter (D), contact angle (φ) and shaft speed (f):

```
BPFO = (n/2) × f × (1 − (d/D)·cos φ)
BPFI = (n/2) × f × (1 + (d/D)·cos φ)
BSF  = (D/2d) × f × (1 − ((d/D)·cos φ)²)
FTF  = (1/2)  × f × (1 − (d/D)·cos φ)
```

Where: **n** = how many balls, **d** = diameter of one ball, **D** = pitch
diameter (the circle the ball centres travel on), **φ** = contact angle (the
angle at which the ball touches the races), **f** = shaft speed in Hz.

### ⚠ Be precise about this bit — it is the thing you could be caught on

**This codebase does not evaluate those formulas from n, d, D and φ at runtime.**
It stores the **published multipliers** for the two specific bearings CWRU used,
which are those formulas *already worked out* for that exact geometry:

```python
# backend/fleetmind/ml/bearing_freq.py
BEARING_MULTIPLIERS = {
    "DE": {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.3983},  # SKF 6205-2RS JEM
    "FE": {"BPFI": 4.9469, "BPFO": 3.0530, "BSF": 3.9874, "FTF": 0.3817},  # SKF 6203-2RS JEM
}
```

Read "BPFO = 3.5848" as **"the outer race is struck 3.5848 times per shaft
revolution"** — a pure geometric ratio for that bearing.

**If asked "show me where you compute BPFO from ball diameter":** the honest
answer is *"we don't — we use the published constant for this specific bearing,
which is standard practice for CWRU work, because the bearing model is fixed and
known."* That is a correct and defensible answer. Do not claim the code derives
them from geometry at runtime; it doesn't, and the file is short enough for
anyone to check.

### The live calculation — Plant D, real numbers

Run on the actual system. **These are the numbers to memorise.**

Plant D is the 3 HP site, nominal **1,730 rpm**:

```
Step 1 — rpm to Hz:
    shaft speed = 1730 ÷ 60 = 28.8333 Hz
    (the shaft turns about 28.8 times every second)

Step 2 — multiply by each geometric constant:
    BPFI = 5.4152 × 28.8333 = 156.1383 Hz  ->  156.14 Hz
    BPFO = 3.5848 × 28.8333 = 103.3617 Hz  ->  103.36 Hz
    BSF  = 4.7135 × 28.8333 = 135.9059 Hz  ->  135.91 Hz
    FTF  = 0.3983 × 28.8333 =  11.4843 Hz  ->   11.48 Hz
```

That is the entire computation. Two lines of arithmetic. **No recording was
consulted.**

All four sites, if you want the full picture:

| Site | Load | rpm | Shaft Hz | BPFI | BPFO | BSF | FTF |
|---|---|---|---|---|---|---|---|
| Plant A | 0 HP | 1797 | 29.95 | 162.19 | 107.36 | 141.17 | 11.93 |
| Plant B | 1 HP | 1772 | 29.53 | 159.93 | 105.87 | 139.21 | 11.76 |
| Plant C | 2 HP | 1750 | 29.17 | 157.94 | 104.56 | 137.48 | 11.62 |
| **Plant D** | **3 HP** | **1730** | **28.83** | **156.14** | **103.36** | **135.91** | **11.48** |

Notice the frequencies drop as load increases — because a loaded motor turns
slightly slower. Being able to point that out shows you understand that these
track the physics rather than being arbitrary constants.

### One extra detail worth knowing: why "2×BSF"

For a **Ball** fault the system checks **twice** the ball spin frequency:

```
BSF at Plant D   = 135.91 Hz
2×BSF            = 271.82 Hz   <- this is the band actually checked
```

Because a defect on a ball strikes the inner race *and* the outer race once per
ball rotation — two impacts per spin, so the knocking occurs at double the spin
rate. This is conventional practice in vibration analysis and it is in the code
as `BALL_USES_SECOND_HARMONIC = True`.

---

## Q4: "How do you check if the model's prediction is actually physically supported?"

This is `measure_spectral_band_energy()` in `backend/fleetmind/agents/tools.py`.

### The logic in plain words

1. The model predicts a fault class — say **Ball**.
2. We look up the frequency a Ball defect *should* produce — **2×BSF =
   271.82 Hz** at Plant D.
3. We draw a **±5% band** around it: **258.23 – 285.41 Hz**. (±5%, not an exact
   line, because real shaft speed drifts and the balls slip slightly. A
   zero-width line would look precise and be misleading.)
4. We measure **how much of the recording's total energy falls inside that
   band**.
5. If the energy is concentrated there → the physics agrees with the model. If
   it isn't → the physics does not support that call, and we say so.

Energy is computed as the sum of squared magnitudes (squaring is standard —
energy goes as amplitude squared), band total divided by spectrum total.

### The complete worked example — run live, end to end

**Service request #60**, currently in the database. (Request IDs shift when the
replay is reset — the stable reference is **signal window #2713**.)

```
WHAT WAS RECORDED
  equipment            : Bearing Motor D2
  site                 : Plant D (CWRU 3 HP), 1730 rpm
  sensor               : drive-end accelerometer @ 12,000 Hz
  window               : #2713, 2,048 samples (0.1707 s)
  RMS vibration        : 0.105521 g
  peak-to-peak         : 0.938875 g

WHAT THE MODEL SAID
  predicted fault      : Ball
  confidence           : 0.9991128   (99.91%)
  rule-based risk      : 0.4914  -> urgency HIGH

WHAT THE PHYSICS EXPECTS
  shaft speed          : 28.83 Hz
  BSF                  : 135.91 Hz
  2×BSF (checked)      : 271.82 Hz
  band (±5%)           : 258.23 – 285.41 Hz   (27.18 Hz wide)

WHAT WAS ACTUALLY MEASURED IN THAT BAND
  energy in band       : 0.000325  of total  (0.0325%)
  loudest point in band: 263.67 Hz, magnitude 2.2024
  loudest point OVERALL: 1382.81 Hz, magnitude 47.5504   <-- nowhere near
```

### The verdict, and how to read it

Here is the yardstick that makes the number meaningful. The band is 27.18 Hz
wide out of a 0–6,000 Hz spectrum. If energy were spread *perfectly evenly*
across all frequencies, you would expect:

```
27.18 ÷ 6000 = 0.4530%  of the energy to land in that band by chance alone
```

We measured **0.0325%**.

```
0.0325 ÷ 0.4530 = 0.072x
```

**There is about fourteen times LESS energy at the defect frequency than random
noise would put there** — and the actual loudest content in the recording sits
at 1,382.81 Hz, which is not a bearing defect frequency at all.

So for this window: **the physics does not corroborate the model's call.** The
classifier is 99.91% confident it is a Ball fault, and the defect frequency is
essentially absent.

### Be ready for the obvious follow-up

If they ask *"so is your model wrong?"* — the honest answer has three parts:

1. **This is one window, not the model's overall performance.** Validated
   accuracy is 62–72%, measured separately.
2. **The classifier is not required to use the defect frequency.** It learns
   whatever separates the classes in the normalised spectrum — which may be
   broadband resonance, like the energy at 1,382 Hz. That can be a perfectly
   valid discriminative signal.
3. **What this genuinely shows is that the confidence is not calibrated.**
   99.91% confidence with no physical corroboration means the number should not
   be read as 99.91% probability of being correct.

And then the point worth making on your own initiative:

> *"The system surfaces this contradiction instead of hiding it. The Frequency
> Lab prints an amber warning when the dominant energy falls outside the defect
> band. I would rather the tool tell me when the evidence is weak than have it
> only agree with the model."*

### Honest detail about thresholds

**There is no hard-coded numeric threshold for strong vs weak.** Be straight
about this if asked. What exists is:

- the tool **reports** the energy fraction and both peak locations as numbers;
- the UI **automatically flags** it when the overall peak falls outside the
  defect band (that is an explicit coded rule);
- the LLM write-up labels it `strong` / `partial` / `weak` / `none` as a
  judgement over those reported numbers.

The flat-spectrum comparison above (0.4530%) is an *interpretive yardstick* for
reading the number — it is reasoning you can defend, not a constant in the
source.

---

## Q5: "Does the AI model use these defect frequencies to make its prediction?"

### The short answer

**No. Never. The classifier has never seen BPFI, BPFO, BSF or FTF.**

The frequency check is **independent verification running alongside the model**,
not part of how the model decides. That independence is what makes it worth
anything — a check built from the model's own inputs would be circular and would
prove nothing.

### The proof, from the code

**1. What the model is fed.** Every place a window reaches the CNN — training,
federated training, augmentation, and prediction — it goes through
`to_fft_magnitude()` first and nothing else:

```
ml/train.py:76-77         train_rows = [(to_fft_magnitude(v), lab) ...]
ml/predict.py:86          values = to_fft_magnitude(values)
federated/run.py:97-98    "train": [(to_fft_magnitude(v), lab) ...]
augmentation/run.py:195   fft_train_rows = [(to_fft_magnitude(v), lab) ...]
```

And `to_fft_magnitude()` returns exactly one thing: a plain 512-number spectrum.
It takes no bearing parameters, no rpm, no site — its only arguments are the raw
samples and the bin count.

**2. Where the defect frequencies actually live.** Searching the entire backend
for `BPFI|BPFO|BSF|FTF|bearing_freq` returns **five files**:

```
fleetmind/ml/bearing_freq.py              <- where they are defined
fleetmind/agents/tools.py                 <- the evidence tool
fleetmind/api/routers/service_requests.py <- the evidence API
fleetmind/api/routers/windows.py          <- the Frequency Lab API
fleetmind/api/schemas.py                  <- API response shapes
```

**Not in that list: `ml/model.py`, `ml/train.py`, `ml/predict.py`,
`ml/dataset.py`.** Every one of the four files that defines or feeds the
classifier contains **zero** references to any defect frequency. `model.py`
itself scores exactly 0 occurrences.

That is the whole argument, and anyone can verify it in ten seconds with grep.

### How to say it

> *"The classifier consumes a generic 512-bin normalised spectrum — the same
> representation for every site and every fault. It was never given the defect
> frequencies, and the code shows that: they appear only in the evidence and API
> layers, never in the four files that define or feed the model. So when I check
> whether energy sits on the defect frequency, I am testing the model against
> independent physics, not against itself."*

The project's own source says this too, in the module docstring of
`bearing_freq.py`:

> *"the CNN does **not** use these frequencies … The characteristic frequencies
> here are independent, human-inspectable evidence for a reviewer —
> corroboration, not an explanation of the model's internals. Do not present
> them as 'what the model looked at'."*

---

## The numbers to actually memorise

If you remember nothing else, remember these six:

| | |
|---|---|
| Samples per window | **2,048** |
| Sampling rate | **12,000 Hz** → window = **0.17 s** |
| FFT bin width | **5.86 Hz** (= 12000 ÷ 2048) |
| Bins kept for the model | **512** → covers **0–3,000 Hz** |
| Plant D shaft speed | **1730 rpm ÷ 60 = 28.83 Hz** |
| Plant D BPFO / BPFI | **103.36 Hz / 156.14 Hz** |

And one worked verdict: window #2713, Ball predicted at 99.91%, 2×BSF band
258–285 Hz, energy in band **0.0325%**, real peak at **1,382.81 Hz** — **weak
corroboration, and the system says so**.

---

## If sir asks this live — the answer to say out loud

> "Only one thing was ever physically recorded: a single accelerometer measuring
> vibration amplitude, 12,000 times a second. There is no separate frequency
> measurement, and there never was one. Frequency is *calculated* from that same
> recording using the Fourier Transform — the same way a piano chord can be
> described either as air pressure over time or as the notes C, E and G. Both
> descriptions hold identical information; nobody measured the notes separately.
> Concretely: each window is 2,048 samples at 12,000 Hz, about a sixth of a
> second. The FFT turns that into 1,025 frequency bins spaced 5.86 Hz apart, and
> we keep the first 512, covering 0 to 3,000 Hz, then log-scale and normalise
> them so the model learns the shape of the spectrum rather than how loud the
> machine is. Separately — and this part uses *no* data from the recording at
> all — I calculate what frequency a bearing defect *should* produce, purely from
> the published bearing geometry and the shaft speed. At Plant D the shaft turns
> at 1,730 rpm, which is 28.83 Hz, and that gives an outer-race frequency of
> 103.36 Hz and an inner-race frequency of 156.14 Hz. Then I check whether the
> recording's energy actually sits where the physics says it should. On the
> window I use as my example the model predicted a Ball fault at 99.91%
> confidence, but only 0.0325% of the spectral energy fell in the expected band
> — roughly fourteen times less than you would get by chance — while the real
> peak was at 1,382 Hz. So on that window the physics does *not* corroborate the
> model, and the system displays exactly that rather than hiding it. The
> important point is that this check is genuinely independent: the classifier is
> only ever shown the generic 512-bin spectrum and has never seen these defect
> frequencies — they appear nowhere in the four files that define or feed the
> model."

---

*Every figure above was produced by running this project's code on 2026-10-03
against the live database. No code was modified to produce this document.*
