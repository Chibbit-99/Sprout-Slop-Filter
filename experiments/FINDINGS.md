# How many training examples does Sprout need?

**Short answer: roughly 150–400 well-varied labelled examples is the practical
target — and *vocabulary coverage* matters more than raw count. That answer
only became measurable after fixing a trainer bug that made dataset size
irrelevant (word vectors were never updated). The fix has now landed in
`subsystems/`, and every figure below was re-measured on 2026-10-18 against
the corrected real code.**

Everything below is measured against the real code in `subsystems/`, loaded the
same way `test.html` loads it. Reproduce with:

```bash
node experiments/learning_curve.mjs    # word-vector audit + learning curves
node experiments/diagnostics.mjs       # convergence and the coverage ceiling
node experiments/realistic_hard.mjs    # fluent slop, shared vocabulary
node experiments/ambiguity.mjs         # effect of debatable labels
```

All four experiments now train through the real subsystem pipeline
(`train_model` → `make_prediction_for_training` → `update_parameters`).
`experiments/lib/trainer_v2.mjs` — the reference implementation that
prototyped the fix — survives only as an independent cross-check inside
`learning_curve.mjs`.

---

## 1. The blocker (FIXED): word vectors were never trained

### The bug

`subsystems/update_parameters.js` always documented that

> Word vectors get a small share of the error too, weighted by how often they
> participated in the representation.

but the function never received `wordVectors` and never wrote to them. The
call site in `train_model.js` passed only the averaged `representation`, so
the only parameters that ever changed were the **8 classifier weights and the
bias — 9 numbers total**. Every word vector stayed at its initial random
value forever, and the model could only learn one linear boundary in a fixed
random projection of the text. Measured pre-fix, held-out accuracy was flat
from 8 examples to 3,200:

| training examples | 8 | 32 | 100 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|
| held-out accuracy (pre-fix) | 60.7% | 66.1% | 57.6% | 67.6% | 62.0% | 64.0% | 66.8% |

And it was not a case of "train longer": at a fixed 800 examples, running
100× more epochs changed nothing (69.7% at 50, 200, 1000, and 5000 epochs
alike). This was also why the README's "thesaurus-based vocabulary expansion"
idea would have done nothing on its own — teaching an unknown word to map
onto a *known* word's vector only helps if those vectors carry learned
meaning, and pre-fix they did not.

### The fix

The update the comments always described is now implemented in the subsystems
themselves:

- `make_prediction_for_training` returns `tokenIndices` alongside `score` and
  `representation`: the vocabulary index of every token that contributed to
  the representation, mirroring `get_word_vector`'s lookup rules (unknown
  tokens map to the shared `<UNK>` entry).
- `update_parameters` accepts those indices as an optional fifth parameter
  and applies `v_t -= lr · g · w / n` to every participating word vector,
  where `g = p − y` and `n` is the number of contributing tokens. The vector
  step is computed from the forward-pass weights, before the classifier
  weights change — the mathematically correct backpropagation order. The
  parameter is optional, so older four-argument callers still work and simply
  update the classifier only.
- `train_model` passes the indices through at the call site the old version
  of this document flagged.
- `test.html` gained regression tests: token-index mapping, participating
  vectors moving (and `<UNK>` staying put when it did not participate),
  four-argument backwards compatibility, and word vectors changing across a
  full `train_model` run.

### Verification

The audit in `learning_curve.mjs` confirms the real subsystem trainer now
changes all 1,184 word-vector values on a 148-word vocabulary, and the real
subsystem curve sits on top of the reference trainer's curve at every dataset
size (largest gap across 10 sizes × 5 seeds: 0.1 points). Learning responds
to data again — see §2.

---

## 2. Learning responds to data

Real subsystem trainer throughout: 5 seeds, 200 epochs, learning rate 0.2.

### Easy, cleanly separable data (gibberish vs. concrete nouns)

| training examples | 8 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| held-out accuracy | 52.9% | 78.5% | 91.6% | **99.4%** | 100% |

Accuracy crosses 90% by ~100 examples and saturates by ~200–400 (pre-fix,
the same regime was flat at ~60–67% at every size). The saturation applies
to *seen* vocabulary, though: on test text built entirely from held-out word
pools, accuracy climbs only to ~72.5% and then stops, no matter how large
the training set gets. That is the coverage ceiling quantified in §3.

### Realistic difficulty (fluent corporate slop, shared vocabulary)

This is the case that matters — slop like *"drive alignment across teams"* is
grammatical and uses ordinary English words, so the model cannot win by
spotting junk characters.

| training examples | 20 | 40 | 80 | 160 | 320 | 640 |
|---|---|---|---|---|---|---|
| accuracy, same pools as training | 75.0% | 87.2% | 91.4% | 96.7% | 99.2% | 100% |
| accuracy, fully held-out wordings | 58.0% | 72.0% | 75.0% | 82.0% | 81.0% | 82.0% |
| false-positive rate | 1.3% | 1.3% | 1.9% | 1.3% | 0.0% | 0.0% |

Useful by ~40, solid by ~80–160. These figures reproduce the reference
trainer's pre-fix-landing measurements to within 0.3 points — the fix changed
*where* the learning happens (real subsystem code), not what the model can
learn on clean data. The held-out-wordings row plateaus at ~82%: once every
distinct phrase in the pools has been seen, more examples only repeat them.

> **Caveat:** these are synthetic sets where the same phrases recur, so they
> are optimistic. Treat **150–400** as the realistic planning figure for real
> data, not 80.

---

## 3. The ceiling is vocabulary coverage, not data volume

Any word the model never saw in training collapses to the single shared
`<UNK>` vector, so unknown content words carry no information at all.
Holding training size fixed at 400 examples and varying how much of the test
wording is new (real subsystem trainer, 5 seeds):

| unknown-word rate | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| held-out accuracy | 99.9% | 87.1% | 74.7% | 61.6% | 49.4% |

Identical to the pre-fix reference measurements, and almost perfectly linear:
0% → 100% unknown drops the model to chance. The coverage ceiling is a
property of the *encoding*, not the optimiser — fixing the trainer did not
touch it. **The real relationship stands: what you need is enough examples
that your training vocabulary covers the words people actually type.** Rough
guidance:

- aim for **~10–20 occurrences per content word** you expect in real input
- aim for the training vocabulary to cover **≥95%** of tokens in real input
- adding more examples that repeat words you already cover buys you very
  little (see the flat 72.5% unseen curve in §2)

The epoch sweep also changed meaning. Pre-fix it proved the capacity ceiling:
flat 69.7% from 50 to 5,000 epochs. With the fixed trainer at the same 800
examples, the model converges within 50 epochs (100% on seen wording) and
stays there to 5,000 epochs — on this synthetic data extra epochs neither
help nor hurt. Once the model can express the boundary, more optimisation is
neutral; only better coverage moves the unseen ceiling.

---

## 4. The ceiling you cannot fix with data: label ambiguity

The README flags this itself — `Do something` is "possibly slop" depending on
the intended definition. Mixing genuinely debatable phrases into training
with random labels, re-run with the real subsystem trainer and 5 seeds
(raised from 3, because contradictory labels make per-seed results noisy):

| ambiguous share | 0% | 10% | 25% | 50% |
|---|---|---|---|---|
| best held-out accuracy | 71.3% | 77.5% | 76.3% | 75.6% |
| held-out accuracy at 1,280 examples | 71.3% | 76.3% | 76.3% | 75.6% |

The test set includes six debatable phrases under *both* labels, which caps
achievable accuracy at 81.25% by construction. Every share converges into
the same ~71–78% band, and the differences between shares are within seed
noise (sd 1.3–6.1 points). Contradictory labels slow early learning (at 40
examples, 25% ambiguity costs ~5 points), but the deeper result is unchanged:
**no amount of data gets past the band.** Even perfectly clean training data
(0% share) plateaus at ~71% on a test set containing debatable items. That
is a **definition** problem, not a data problem.

One honest correction to the previous version of this document: it reported a
sharper penalty (71.9% → ~72% → ~65% → ~58%, destabilising at larger sizes).
Those numbers were measured with the reference trainer `trainer_v2.mjs`,
which shuffles training order each epoch and applies the word-vector step
with the just-updated weights. Under contradictory labels that combination
oscillates and degrades with more data; the subsystem trainer — fixed
presentation order, vector step computed from the forward-pass weights — is
measurably more stable. On *clean* data the two trainers agree to within 0.1
points everywhere; they diverge only in this contradictory-label regime.
`learning_curve.mjs` runs both side by side, so the comparison can be
re-checked at any time.

---

## Recommendation

1. **Fix the trainer before collecting more data.** ✅ Done — the
   word-vector update now lives in `subsystems/update_parameters.js` (fed by
   `tokenIndices` from `make_prediction_for_training`, wired through
   `train_model`), with regression tests in `test.html`. Extra examples now
   actually teach the model something.
2. **Start with ~150–300 labelled examples**, balanced, with varied wording.
   That is enough to tell whether the approach works at all.
3. **Grow toward ~1,000 only if coverage demands it** — specifically, if your
   real inputs contain content words that rarely or never appear in training.
4. **Write down a precise slop definition first**, and label borderline cases
   consistently. §4 shows that even the fixed trainer cannot get past the
   ambiguity band; consistency of labels matters more than their volume.
5. **Reconsider the architecture before scaling past ~1,000.** Two limits
   survive the trainer fix, and the reruns confirm both are encoding
   problems, not optimisation problems:
   - `encode_text` averages word vectors, so word order is discarded —
     `buy milk` and `milk buy` are identical, and negation is invisible.
   - Unknown words all share one vector. §3 shows accuracy falling almost
     linearly with the unknown-word rate, and §2 shows unseen-wording
     accuracy flat at ~72–82% no matter how much training data is added.
     The README's synonym plan is one way to address this; character-level
     subword features are another.

### Staging table

| examples | expectation | verdict |
|---|---|---|
| 0–50 | ~75% on clean data, unstable | prototype only |
| 150–300 | ~90%+ on familiar wording | **good results, start here** |
| 500–1,000 | plateau on familiar wording | add only to raise coverage |
| 2,000+ | no gain without architecture changes | fix encoding/subwords instead |

All figures are single-machine measurements on synthetic data generated by
`experiments/lib/data.mjs`, re-measured on 2026-10-18 with the fixed
subsystem trainer. They bound the *shape* of the answer; your real numbers
will depend on how separable your definition of slop turns out to be.
