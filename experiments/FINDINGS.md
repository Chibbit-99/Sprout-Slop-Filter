# How many training examples does Sprout need?

**Short answer: right now, no number would work. There is a bug that makes the
dataset size irrelevant. After a one-function fix, roughly 150–400 well-varied
labelled examples is the practical target — and *vocabulary coverage* matters
more than raw count.**

Everything below is measured against the real code in `subsystems/`, loaded the
same way `test.html` loads it. Reproduce with:

```bash
node experiments/learning_curve.mjs    # as-is vs patched learning curves
node experiments/diagnostics.mjs       # the two structural ceilings
node experiments/realistic_hard.mjs    # fluent slop, shared vocabulary
node experiments/ambiguity.mjs         # effect of debatable labels
```

---

## 1. The blocker: word vectors are never trained

`subsystems/update_parameters.js` documents that

> Word vectors get a small share of the error too, weighted by how often they
> participated in the representation.

but the function never receives `wordVectors` and never writes to them. The
call site in `train_model.js` line 78 passes only the averaged `representation`:

```js
update_parameters(model, errorInfo, learningRate, representation);
```

So the only parameters that ever change are the **8 classifier weights and the
bias — 9 numbers total**. Every word vector stays at its initial random value
forever. Proof:

```
word vector for 'buy' BEFORE: [-0.014840609580278398, 0.09896458913572133, ...]
word vector for 'buy' AFTER : [-0.014840609580278398, 0.09896458913572133, ...]
identical across 500 epochs: true
```

### Why that makes dataset size irrelevant

The model is `sigmoid(w · r + b)` where `r` is the average of *fixed random*
8-dimensional vectors. All it can learn is one linear boundary in a random
projection of the text. Held-out accuracy is flat from 8 examples to 3,200:

| training examples | 8 | 32 | 100 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|
| held-out accuracy | 60.7% | 66.1% | 57.6% | 67.6% | 62.0% | 64.0% | 66.8% |

Ten times more data, zero improvement. And it is not a case of "train longer":
at a fixed 800 examples, running 100× more epochs changes nothing.

| epochs | 50 | 200 | 1000 | 5000 |
|---|---|---|---|---|
| held-out accuracy | 69.7% | 69.7% | 69.7% | 69.7% |

The model converges to the best boundary it can express and stops. **This is
why the README's "thesaurus-based vocabulary expansion" idea would also do
nothing on its own** — teaching an unknown word to map onto a *known* word's
vector only helps if those vectors carry learned meaning, and today they don't.

---

## 2. After fixing the trainer

`experiments/lib/trainer_v2.mjs` implements the update the comments already
describe — `v_t -= lr · g · w / n`, so word vectors learn alongside the
classifier. Nothing in `subsystems/` is modified; this is a reference
implementation for measurement.

### Easy, cleanly separable data (gibberish vs. concrete nouns)

| training examples | 8 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| held-out accuracy | 52.9% | 78.4% | 91.5% | **99.5%** | 100% |

Learning finally responds to data. Accuracy crosses 90% by ~100 examples and
saturates by ~200–400.

### Realistic difficulty (fluent corporate slop, shared vocabulary)

This is the case that matters — slop like *"drive alignment across teams"* is
grammatical and uses ordinary English words, so the model cannot win by
spotting junk characters.

| training examples | 20 | 40 | 80 | 160 | 320 | 640 |
|---|---|---|---|---|---|---|
| accuracy on new wordings | 75.0% | 86.9% | 91.4% | 96.7% | 99.2% | 100% |
| false-positive rate | 1.3% | 1.9% | 1.9% | 1.3% | 0.0% | 0.0% |

Useful by ~40, solid by ~80–160.

> **Caveat:** these are synthetic sets where the same phrases recur, so they are
> optimistic. Treat **150–400** as the realistic planning figure for real data,
> not 80.

---

## 3. The ceiling is vocabulary coverage, not data volume

Any word the model never saw in training collapses to the single shared `<UNK>`
vector, so unknown content words carry no information at all. Holding training
size fixed at 400 examples and varying how much of the test wording is new:

| unknown-word rate | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| held-out accuracy | 99.9% | 87.1% | 74.7% | 61.6% | 49.4% |

Almost perfectly linear, and 0% → 100% unknown drops the model to chance. **This
is the real relationship: what you need is enough examples that your training
vocabulary covers the words people actually type.** Rough guidance:

- aim for **~10–20 occurrences per content word** you expect in real input
- aim for the training vocabulary to cover **≥95%** of tokens in real input
- adding more examples that repeat words you already cover buys you very little

---

## 4. The ceiling you cannot fix with data: label ambiguity

The README flags this itself — `Do something` is "possibly slop" depending on
the intended definition. Mixing genuinely debatable phrases into training with
random labels (3 seeds, 300 epochs):

| ambiguous share | 0% | 10% | 25% | 50% |
|---|---|---|---|---|
| best held-out accuracy | 71.9% | ~72% | ~65% | ~58% |

With a quarter of labels debatable, accuracy stalls around 65% no matter how much
data you add, and results become unstable at larger sizes. That is not a data
problem — it is a **definition** problem. Note that my held-out test set capped
at ~73% by construction, so read this as "ambiguity caps you", not as precise
numbers.

---

## Recommendation

1. **Fix the trainer before collecting more data.** Today, extra examples are
   wasted. Implement the word-vector update in `update_parameters.js` (needs
   `model` and the token indices, not just the averaged representation).
2. **Then start with ~150–300 labelled examples**, balanced, with varied wording.
   That is enough to tell whether the approach works at all.
3. **Grow toward ~1,000 only if coverage demands it** — specifically, if your
   real inputs contain content words that rarely or never appear in training.
4. **Write down a precise slop definition first**, and label borderline cases
   consistently. Ambiguity caps accuracy far harder than dataset size does.
5. **Reconsider the architecture before scaling past ~1,000.** Two limits remain
   after the trainer fix, and neither is fixed by more data:
   - `encode_text` averages word vectors, so word order is discarded —
     `buy milk` and `milk buy` are identical, and negation is invisible.
   - Unknown words all share one vector. The README's synonym plan is one way to
     address this; character-level subword features are another.

### Staging table

| examples | expectation | verdict |
|---|---|---|
| 0–50 | ~75% on clean data, unstable | prototype only |
| 150–300 | ~90%+ on familiar wording | **good results, start here** |
| 500–1,000 | plateau on familiar wording | add only to raise coverage |
| 2,000+ | no gain without architecture changes | fix encoding/subwords instead |

All figures are single-machine measurements on synthetic data generated by
`experiments/lib/data.mjs`. They bound the *shape* of the answer; your real
numbers will depend on how separable your definition of slop turns out to be.
