# Sprout: Slop Filter

A small machine-learning project exploring how to detect meaningless or low-effort text ("slop"), starting with task descriptions in a task tracker.

## Goal

Given a string, predict how likely it is to be slop, returning a score from **0 to 1**:

- `0` = the model thinks the text is genuine.
- `1` = the model thinks the text is slop.

"Slop" means meaningless, nonsensical, or empty filler—not something unethical or bad. For example, `Steal someone's wallet` describes a genuine intention even though the action is unethical.

## Running it

```bash
npm start          # serve the folder at http://localhost:8000
```

Then open <http://localhost:8000>. The page loads the trained `model.json`
and the `training.json` dataset, and lets you train a fresh model, watch it
learn, score text, and save or load models — all in the browser, with no
build step and no dependencies.

The page has to be served over HTTP: browsers block `fetch` for pages opened
straight from disk (`file://`), and the page needs to read `training.json`
and `model.json`. If you open it that way anyway, it falls back to a small
built-in demo set and says so.

To retrain the shipped model from the command line:

```bash
npm run train                       # training.json -> model.json
node train.mjs --epochs 200 --lr 0.2 --vector-size 24 --seed 7
node train.mjs --help               # every option
```

| File | What it is |
| --- | --- |
| `index.html` | The browser app: train, evaluate, test, save. |
| `training.json` | 524 labelled examples (262 genuine, 262 slop). |
| `model.json` | The trained model, ready to use — the app loads it on start. |
| `train.mjs` | Command-line trainer that produces `model.json`. |
| `serve.mjs` | A tiny dependency-free static server. |
| `subsystems/` | The model itself: tokenising, encoding, training, scoring. |
| `test.html` | The subsystem test suite. |
| `experiments/` | Measurements behind `experiments/FINDINGS.md`. |

### Checking it works

```bash
npm test          # the learning-curve experiment (subsystem regression check)
npm run measure   # the measurements in FINDINGS.md section 5
npm run test:app  # a headless pass over the browser app
```

The app smoke test needs jsdom, which is the only thing in this project that
is not dependency-free:

```bash
npm install --no-save jsdom
npm run test:app
```

Without jsdom it prints a note and exits, so it never blocks anything.

## The browser app

`index.html` is the main way to work with Sprout.

- **Train** — set the epochs, learning rate, vector size and seeds, then
  watch the loss and validation accuracy move as the model learns. Training
  runs in small chunks so the page stays responsive.
- **Evaluate** — the test split is held back from training and scored
  afterwards: accuracy, how much slop is caught, how much genuine text is
  wrongly flagged, and a confusion matrix. Every held-out example is listed
  with its score.
- **Test** — type a task description and read the score, the verdict, and
  which of its words the model actually knows (anything outside the
  training vocabulary collapses to a single `<UNK>` vector, which is the
  model's main blind spot).
- **Edit the data** — add labelled examples in the page and retrain, or
  import and export a dataset as JSON.
- **Save the model** — download it as JSON, keep it in the browser, or load
  one back in.

The app uses the real subsystem code in `subsystems/`, the same files the
test page and the experiments use, so what you train in the browser is what
the command-line trainer produces.

### Score softening

The training data is cleanly separable, so a freshly trained model is very
confident: almost every score lands on 0.000 or 1.000. Dividing the
classifier weights and the bias by a temperature spreads the scores out
again, so the gauge is worth reading. It cannot change a single decision —
the sign of the logit is untouched, so the 0.5 boundary and every metric
stay exactly where they were.

## Basic architecture plan

1. **Training examples** — collect text samples and label each one as either genuine or slop.
2. **Split text into words** — turn the input string into words the model can work with.
3. **Represent words with numbers** — give words numerical representations (vectors). These can be learned during training.
4. **Handle unfamiliar words** — explore looking up unknown words in a thesaurus or word API and connecting them to known related words. Related words are clues, not guaranteed replacements.
5. **Combine the words** — use the word representations together to make a representation of the full input.
6. **Make a prediction** — use the combined representation to produce a slop score between 0 and 1.
7. **Learn from mistakes** — compare predictions with the training labels and adjust the model's numbers to improve future predictions.
8. **Test it** — check how well it handles examples it was not trained on, including genuine text it should not incorrectly flag.

This is an initial plan, not a claim that all these components have been implemented. The first version should stay small and understandable.

## Synonym lookup experiment

One idea is to use the [Datamuse API](https://www.datamuse.com/api/) to find synonyms and help Sprout deal with words it has not learned yet.

For example, a lookup for `purchase` can return `buy`, while a lookup for `buy` can return `purchase`. This could help Sprout connect different wording that expresses a similar idea.

A small JavaScript example of the lookup:

```js
async function getSynonyms(word) {
    const url = new URL("https://api.datamuse.com/words");
    url.searchParams.set("rel_syn", word);
    url.searchParams.set("max", "20");

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Datamuse API error: ${response.status}`);
    }

    const results = await response.json();
    return results.map(result => result.word);
}

getSynonyms("purchase").then(console.log);
```

### How Sprout could use the results

1. Encounter a word that is missing from the model's vocabulary.
2. Look up its synonyms with Datamuse.
3. Check whether any returned words are already known to the model.
4. Use those matches as clues that the unfamiliar word may be related to a known word.

Synonyms are **clues, not perfect replacements**. Results can include different meanings of a word: for example, a lookup for `buy` may also return `steal` or `bribe` because words can have multiple meanings and uses. Sprout should not automatically assume every returned word means exactly the same thing in the current sentence.

To keep the experiment simple and manageable, cache lookup results so the same word is not repeatedly requested, and start with one round of lookups instead of searching synonyms of synonyms forever.

The useful test is to compare the model **with synonym connections** against the same model **without them**, using examples it was not trained on. That will show whether this idea actually improves predictions.

Datamuse currently documents that an API key will be required from **1 February 2027**. Keeping synonym lookup in a separate part of the project would make it easier to change services later if needed. If Sprout uses Datamuse in a public app, its documentation should acknowledge the API.

The lookup itself is now implemented in `subsystems/map_synonyms.js` and used by `classify_text` when a model has `options.useSynonyms` switched on — the app has a checkbox for it. What has **not** been done yet is the measurement above: there is no experiment showing whether the synonym connections actually improve held-out predictions, so treat the feature as untested.

## Example labels

| Text | Label |
| --- | --- |
| `Buy milk` | Genuine |
| `Finish maths homework` | Genuine |
| `Do something` | Depends on the intended definition; possibly slop |
| `asdfghjkl` | Slop |
| `Steal someone's wallet` | Genuine text, despite describing an unethical action |

The labels used for training will determine what the model learns, so examples should cover varied wording and avoid treating vagueness or ethics as automatically meaning slop.

`training.json` follows that table. The definition it uses is:

> **Genuine** — names a concrete action on a concrete target, so someone could pick it up and do it.
> **Slop** — keyboard mash, or filler that names no target: `asdfghjkl`, `do something`, `align stakeholders`.

Being unethical is not the same as being slop, so the dataset keeps deliberately awkward genuine examples (`Steal someone's wallet`, `Delete the customer records`). Borderline phrases are labelled one way and stay that way — `experiments/FINDINGS.md` §4 shows that mixed labels cap accuracy no matter how much data you add.

## Possible future use

The same general training code could later be used for other text-classification tasks, such as checking whether GitHub commit messages are informative. That would need its own examples and labels: a message that is acceptable as a task description may still be a poor commit message.

## Current status

The model, the training data and the browser app are implemented. A trained
model ships in `model.json` and scores **96.3%** on a held-out test split
with a 0% false-positive rate — see `experiments/FINDINGS.md` §5 for the
measurement and its limits.

Still open:

- The synonym lookup is wired up but unmeasured (§Synonym lookup experiment).
- The encoding still averages word vectors, so word order is lost and
  negation is invisible.
- Unknown words still collapse to a single `<UNK>` vector, which is the
  ceiling on how the model handles wording it has not seen.

