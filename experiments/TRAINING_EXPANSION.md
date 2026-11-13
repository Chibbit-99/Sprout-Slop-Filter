# Task data expansion: 3,402 observed words and 3,686 training rows

**This records the unanchored expansion checkpoint** (`task_unanchored_model.json`); the current default uses [same-size baseline anchoring](ANCHOR_OPTIMISATION.md) and scores 35/40 on the unfamiliar diagnostic without increasing parameters. The counts and dataset provenance below still apply to the default.

“1,000 training datasets” is interpreted as **at least 1,000 labelled examples actually used for fitting**, not 1,000 independent datasets. Run `npm run build:task-data`, `npm run train`, `npm run test:model`, `npm run test:app` to reproduce. `npm run stats -- model.json` reports its trainable/nonzero parameter counts. The browser has **no situation modes**: it loads the expanded `training.json` and `model.json` directly.

## Counting, provenance and split

- `training.json` contains **1,124 displayed labelled examples**, 562 per label. Its seed-1 split has **786 train**, 168 validation, 170 test rows.
- `experiments/task_supplement.json` contains **2,900 additional training-only rows**, 1,450 per label. It is added **after** the base split, bringing fitting to **3,686 actual rows**. Neither the browser nor CLI adds it to validation/test. Disable the supplement checkbox to train on the base split alone. The original 140 supplementary rows are unchanged at the start of that file.
- `build_vocabulary` runs on the actual training partition plus supplement, yielding **3,424 distinct observed tokenizer tokens, including 3,402 alphabetic words** (the serialized vocabulary also contains a separate `<UNK>` sentinel, for 3,425 entries). The count is not padded with dummy slots or unused words. V4 still has 524,289 *trainable parameters*, a different quantity from vocabulary size.
- The original 684 examples and v4 model are frozen in `task_baseline_training.json` and `task_baseline_model.json`. `build_task_expansion.mjs` inverts the deterministic seed-1 shuffle so that **all 102 original validation and 104 original test rows stay out of training** and the old 104 test rows remain in the expanded test split. The extra 440 base rows are split 308/66/66. Regression checks assert no original held-out row appears in the expanded training partition; annotated OOV targets stay unseen.
- New task examples are **synthetic templates**, not thousands of independently labelled real requests. The generator uses 1,600 two- to four-word names of physical artifacts selected from Princeton WordNet 3.1 `noun.artifact` lemmas (packaged as [`wordnet-db`](https://www.npmjs.com/package/wordnet-db) version 3.1.14). The exact selected list is `task_artifact_terms.json`; its copyright/license is `WORDNET_LICENSE.txt`. Each term appears once in an actionable inspection/inventory request and once in a vague non-actionable request. This controls for rare target nouns being a one-sided label cue, but repeated sentence templates are still an **artificial style cue**. No WordNet dependency or network access is needed at runtime or to rebuild from the checked-in terms list.

## Measurements (developer-authored, not deployment accuracy)

| Model/slice | Correct | FP / FN | Notes |
| --- | ---: | ---: | --- |
| Unanchored expanded v4, expanded seed-1 test | 168/170 (98.8%) | 1 / 1 | Includes 104 old locked test and 66 new templated rows. Easy style overlap; not independent. |
| Unanchored expanded v4, original locked test only | 102/104 (98.1%) | 1 / 1 | Original baseline v4 scored 101/104 (97.1%, FP 0 / FN 3). Not a trustworthy +1 claim; model and data were developed with these probes visible. |
| Unanchored expanded v4, 20 transfer probes | 20/20 | 0 / 0 | 24/120 tokens unseen, down from 35/120 in the historical model. |
| Unanchored expanded v4, 40 unfamiliar-word probes | 31/40 | 9 / 0 | **Worse** than the historical model's 34/40. Inflection group falls to 5/8; all four actionable non-English inputs remain false positives. |

The v4 architecture is unchanged: **524,289 trainable scalars**, **51,358 nonzero weights** after fitting, **1,624,771 bytes** for the *unanchored* sparse JSON checkpoint (current anchored `model.json`: 1,633,459 bytes). The old baseline used 15,719 nonzero weights and 480,798 bytes. The expanded model remains offline and loads in the browser, but is substantially larger; more observed words do **not** confer general language understanding or automatically improve cross-domain accuracy. For example, its zero-shot helpdesk ordinary score fell from 18/24 to 17/24; the agenda ordinary score rose from 21/24 to 23/24. See `npm run test:domains` for breakdowns. A real deployment needs diverse human-labelled tasks and a fresh locked external test set, plus threshold/false-positive review. Do not use the model to verify truth, Git commits or authorship.
