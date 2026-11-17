# Offline tenfold lexical coverage (historical checkpoint; now augmented with real issue titles)

At the original lexicon-only checkpoint, the single-mode task model recognised **34,402 distinct alphabetic tokens**: **3,402 were observed in labelled training**, and the rest come from a licensed WordNet 3.1 lookup. Counting tokenizer punctuation/numeric tokens, the union is **34,424**, versus 3,424 observed tokenizer tokens before. These are *not* 34,402 labelled training words. The number of actual task training rows remains 3,686 (786 base + 2,900 training-only); displayed examples remain 1,124. No held-out row or developer-authored unfamiliar probe was added to training. WordNet supplies lexical categories and direct first-sense synonyms, **not** labels or a universal definition of actionable text.

## How it works

`experiments/build_wordnet_lexicon.py` selects 31,000 previously unobserved single-word ASCII lemmas (3–20 letters) from `wordnet-db@3.1.14`, prioritised by WordNet tag count. Along with 3,037 training-observed lemmas, these form 34,037 direct lookup keys. Each entry stores its WordNet lexicographer class (45 classes are represented) and, where available, a *direct first-sense synonym already in the training vocabulary*. A conservative plural fallback also recognises some inflected forms. Lexical coverage is English-centric; many rare entries lack sense-frequency evidence, senses can be wrong in context, and names, code, non-English text and compound phrases are not generally covered. The lookup does not pad `model.vocabulary`.

`subsystems/richer_features.js` uses the class as a learned sparse feature in the existing affix hash bank, and a known synonym as a scaled existing word feature. Character, word-order and subword features still run for every token. This keeps **524,289 potential trainable scalars** (including bias). The then-shipped task checkpoint had **51,393 nonzero feature weights**, ~2.36 MB of JSON; no network or situation modes are required. The page colours training-observed tokens green, lexicon-only tokens gold and neither red. The CLI reports each category separately. A lexical match is a cue, not a guarantee of meaning or correctness. The WordNet notice is embedded in the shipped model metadata and also at `WORDNET_LICENSE.txt`; redistribute the standalone lookup with that notice.

## Selection and diagnostics

A fresh v4 trial trained directly with lexical cues, using the existing 45% baseline anchor, **regressed** on both inspected validation partitions: original **96/102** (FP 3, FN 3) versus base **97/102** (FP 2, FN 3), and expanded **162/168** (FP 3, FN 3) versus base **163/168** (FP 2, FN 3). Its expanded held-out result was unchanged **168/170** (FP 0, FN 2). The trial is **not shipped**. Merely attaching a dictionary to the previously trained model made no measured change because the new class slots had not learned the task. Instead, `lexicon_class_weights.json` contains only the 45 trained category coefficients from the trial; `assemble_lexicon_model.mjs` adds these to the better, previously selected anchored checkpoint and bundles the lookup. Existing feature weight slots (not extra parameters) are reused; collisions with other feature keys remain possible. This was selected only after looking at validation and diagnostics, so it is **not an independent accuracy result**.

| Inspected diagnostic | Before lookup | Bundled class head |
| --- | --- | --- |
| Original validation | 97/102, FP 2/FN 3 | 97/102, FP 2/FN 3 |
| Expanded validation | 163/168, FP 2/FN 3 | 163/168, FP 2/FN 3 |
| Expanded held-out | 168/170, FP 0/FN 2 | 168/170, FP 0/FN 2 |
| Original held-out | 102/104, FP 0/FN 2 | 102/104, FP 0/FN 2 |
| Five unfamiliar-word groups | 35/40, FP 5 | 35/40, FP 5 |
| Helpdesk hard, task model zero-shot | 9/16, FP 6/FN 1 | 9/16, FP 6/FN 1 |
| Agenda hard, task model zero-shot | 13/16, FP 3/FN 0 | 13/16, FP 3/FN 0 |

The cue is **not inert**: on an authored example, “Verify the ammeter before Monday”, the score changes from 0.2195 to 0.2137, but **no accuracy improvement has been demonstrated**. The four actionable non-English challenge messages still fail; this is not a multilingual model. Existing splits and probes were inspected repeatedly and cannot estimate real-world performance, and a commit message's *authenticity* cannot be inferred from text alone. Production use requires independently collected, author-diverse task/commit text, new locked tests, calibration and review of uncertain predictions. See `npm run test:lexicon` for exact comparisons. No developer-authored examples used in this report should be mistaken for independent test data.

## Reproduce

```sh
npm install --no-save wordnet-db@3.1.14  # only needed to rebuild the lookup
python3 experiments/build_wordnet_lexicon.py node_modules/wordnet-db/dict
# Optional: regenerate the trial/class-head coefficients, *not* the shipped checkpoint:
node train.mjs --extra experiments/task_supplement.json --anchor-model experiments/task_baseline_model.json --anchor-mix 0.45 --lexicon experiments/wordnet_lexicon.json --out experiments/task_lexicon_trial_model.json
node experiments/extract_lexicon_head.mjs experiments/task_lexicon_trial_model.json
node train.mjs --extra experiments/task_supplement.json --anchor-model experiments/task_baseline_model.json --anchor-mix 0.45 --out /tmp/lexicon_historical.json
node experiments/assemble_lexicon_model.mjs /tmp/lexicon_historical.json /tmp/lexicon_historical.json
# For the CURRENT default with real issue titles, use npm run train instead.
npm run test:model && npm run test:lexicon && npm run test:app
```

The trial model is a temporary experiment, not part of the release; its recipe is recorded so the 45 coefficients are reproducible. The historical two-command reproduction and the current `npm run train` require only checked-in data, baseline checkpoint, class-head coefficients and lookup, with no install step. When retraining the untouched default corpus in the browser with the full supplement, the same 45-slot prior is copied from its current shipped model. Edited/imported corpora never inherit the task-specific lexical prior. Custom v4 CLI training may explicitly use `--lexicon` to train its own lexical coefficients, but doing so alone should not be presumed to improve accuracy.

The current default additionally trains on 131 attributed real issue titles. Its observed/recognised counts and quality comparison are in [REAL_ISSUE_GENERALISATION.md](REAL_ISSUE_GENERALISATION.md); the numbers above are historical, not the current default.
