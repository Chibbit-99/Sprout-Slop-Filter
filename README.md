# Sprout: Slop Filter

A small **offline** classifier for generic filler versus informative text, implemented in plain JavaScript for Node and browsers. Its default v4 model has **524,289 trainable parameters** (524,288 feature weights + one bias), while the expanded task and commit model files are **1.63 MB and 300 KB** respectively. Capacity is not knowledge: most coefficients remain zero after training.

## Run it

```bash
npm start                     # open http://localhost:8000
node score.mjs "Replace the kitchen tap washer"
node score.mjs --domain commits "Fix timeout when fetching user profiles"
node score.mjs --domain helpdesk "The printer jams on two-sided pages"
node score.mjs --domain agendas "Decide who signs the contract"
git log -1 --pretty=%s | node score.mjs --domain commits
npm run stats -- model.json    # exact parameter count and nonzero weights
npm run stats -- commit_model.json
```

`index.html` has **no situation modes or preset selector**: it loads the task-trained `model.json` and `training.json` and scores text with that single label definition. You can score individual or batch text, edit/import labelled examples, retrain, and save/load models. The default dataset contains **1,124 labelled examples**, and the shipped model fits **3,686 training rows** (786 from the seed-1 base split plus 2,900 training-only examples) with **3,402 distinct alphabetic words** (3,424 observed tokenizer tokens including punctuation and numbers), excluding the `<UNK>` sentinel. The original 684-row held-out partitions are preserved. Default retraining blends in a same-size, old task checkpoint as a training-time prior (45% weights) to mitigate synthetic-template overfitting; inference still uses one model and the same parameter count. See [anchor optimisation](experiments/ANCHOR_OPTIMISATION.md). The browser's supplement option reproduces that setup on the unedited base dataset; disable it to train on base rows only. On imported/edited datasets, supplements are skipped. All scoring is local and needs no word API; optional Datamuse applies only to the legacy v1 encoder when explicitly enabled. Other domain-specific models remain available as separate CLI experiments, not as modes in the page. Most new examples are templated from WordNet artifact names, not real user text; see [data provenance and limitations](experiments/TRAINING_EXPANSION.md).

**A commit-message score measures informativeness, not authenticity.** It cannot know if code changed, whether the message describes it truthfully, or who wrote it. `genuine` in `commit_training.json` means a specific change is named; `slop` means the message is generic. The task policy instead concerns an actionable request with a concrete target, regardless of ethics. Helpdesk and meeting agendas use *different* policies: reports with concrete symptoms/errors versus requests needing diagnostic detail, and specific decisions/subjects versus vague agenda placeholders. Each experimental domain model uses its own 60 training-only rows, not the task/commit labels. Neither shipped domain model has an independent held-out accuracy estimate; the page does not load them automatically. See [domain-model diagnostics](experiments/DOMAIN_MODELS.md) and their limitations.

## Architecture and parameter count

- v4 encodes word unigrams, character 3-, 4-, and 5-grams, ordered adjacent and skip word pairs, first/last-word position, and prefixes/suffixes into **eight disjoint 65,536-weight hash banks**. Banks prevent collisions between feature *types* (collisions within a bank remain). These are genuine addressable trainable coefficients, not inert padding. Pairs provide limited local context and subwords provide spelling clues for unseen words. Input is capped at 128 tokens and 64 characters per token.
- A logistic classifier uses per-feature AdaGrad updates. **8 × 65,536 + 1 bias = 524,289 trainable scalars per model.** But only **51,358 expanded task weights / 9,797 commit weights are nonzero** after training. Vocabulary strings and optimizer accumulators are *not* trainable parameters. Sparse JSON stores only the nonzero weights; loading expands them into a dense array. In-memory numeric payload for weights plus training accumulators is roughly 8 MiB, excluding JS object overhead. See [parameter accounting and v3 comparison](experiments/ARCHITECTURE_V4.md).
- The default task recipe uses **same-size weight-space anchoring**, not extra inference parameters or a second runtime model. After fitting the expanded corpus, it blends 45% of the frozen pre-expansion v4 checkpoint into the coefficients. See [validation selection and tradeoffs](experiments/ANCHOR_OPTIMISATION.md). Other domain models are not blended.
- Older v1/v2/v3 models remain loadable; `--encoder legacy`, `--encoder hashed-subword-v2` and `--encoder structured-subword-v3` still train them. V4 is `--encoder richer-subword-v4` (the default). The v4 width is fixed by its eight banks. `model_parameter_stats` in `subsystems/` is shared by Node, tests and the browser. The `vocabulary` array in v2–v4 records observed training tokens for coverage/display; a new token still receives subword features.

**Subword overlap is not semantic understanding.** Non-English actionable examples are still wrongly flagged. Project jargon, negation, misleading detail, and new contexts remain difficult. A score is not a calibrated probability of truth. Review borderline cases and test independently labelled real data before using this as a gate.

## Retrain and test

```bash
npm run build:task-data        # regenerate JSON from frozen baseline + attributed artifact terms
npm run train                  # expanded task + 2,900 training-only rows; 45% same-size prior
npm run train:commits          # v4 commit model: fixed base split + 80 supplemental train rows
node train.mjs --data my_data.json --out my_model.json  # no supplement by default
node train.mjs --encoder structured-subword-v3 --data my_data.json --out old_model.json
node train.mjs --help
npm run test:model             # versioned load/save, counts, OOV and round trips
npm run test:anchor            # validation-only prior selection; report post-selection diagnostics
npm run test:architecture      # historical baseline v3/v4 comparison, frozen 684-row set
npm run test:vocab             # 0 / partial / full supplemental vocabulary, frozen splits
npm run train:contexts         # deterministic helpdesk and agenda artifacts (60 rows each)
npm run test:domains           # leakage-checked ordinary/hard/new domain diagnostics
npm run test:contexts          # new-word and non-commit-context challenge
npm run test:app               # browser smoke test (npm install --no-save jsdom)
npm test                       # historical legacy learning curve
```

`train.mjs` uses seed-1 70/15/15 base splits and builds the observed vocabulary from **only** the training partition plus optional `--extra` rows. `--extra-count N` takes the first N supplement rows; the provided files alternate genuine/slop examples so prefixes remain balanced. The trainer rejects duplicate supplement sentences from *any* base split. Default training is 120 epochs, learning rate 0.8 and score-softening ×4. Softening scales weights and bias together and does not affect decisions at 0.5. `--vector-size` applies to v1 only. Dataset files are JSON arrays of `{ "text": "...", "label": "genuine" | "slop" }`. The scoring CLI accepts `--domain tasks|commits|helpdesk|agendas`, `--model path` and `--threshold 0..1` and prints JSON; the verdict describes the selected *text policy*, not authenticity. The web page must be served over HTTP to fetch JSON assets.

## Measurements (diagnostic, not deployment estimates)

| Model | Trainable (nonzero weights) | Held-out split | Transfer probes | On disk |
| --- | ---: | --- | ---: | ---: |
| Expanded anchored tasks: 786 base train + 2,900 supplement | 524,289 (51,358) | **98.8%**, TP 78 / FP 0 / TN 90 / FN 2 (170 cases) | 20/20 | 1,633,459 bytes |
| Historical task baseline (archived) | 524,289 (15,719) | **97.1%**, TP 48 / FP 0 / TN 53 / FN 3 (104 cases) | 20/20 | 480,798 bytes |
| Commits: 139 base train + 80 supplement | 524,289 (9,797) | **93.5%**, TP 13 / FP 1 / TN 16 / FN 1 (31 cases) | 20/20 | 299,609 bytes |

The expanded split retains the 104 historical task test rows, on which the expanded model scores **102/104 versus the historical model's 101/104**; these data and probes have been inspected during development, so this is **not an independent real-world improvement**. Without anchoring, synthetic vocabulary expansion worsened the separate 40-case unfamiliar-word diagnostic from **34/40 to 31/40**. The same-size prior brings it to **35/40**, but all four actionable non-English inputs remain false positives. On the developer-authored validation partition, anchoring holds accuracy at 97/102 while trading one false negative for one fewer false positive. Neither vocabulary size nor parameter count guarantees accuracy. The [training expansion report](experiments/TRAINING_EXPANSION.md) documents the provenance, split-locking assertions, false positives and tradeoffs. The [historical v4 vocabulary-growth report](experiments/ARCHITECTURE_V4.md) and [context report](experiments/CONTEXT_CHALLENGE.md) use the frozen baseline; `npm run test:domains` compares the current task model against it and the separately trained helpdesk/agenda models. All are developer-authored diagnostics, **not deployment estimates**.

## Files

| Path | Purpose |
| --- | --- |
| `training.json`, `commit_training.json` | Labelled corpora, split before supplemental training. |
| `experiments/task_baseline_training.json`, `experiments/task_baseline_model.json`, `experiments/task_unanchored_model.json` | Frozen pre-expansion and unanchored checkpoints for reproducible comparisons; not needed at inference. |
| `subsystems/blend_models.js` | Same-size task weight-space prior, used during default CLI/browser retraining. |
| `experiments/task_artifact_terms.json`, `experiments/build_task_expansion.mjs`, `experiments/WORDNET_LICENSE.txt` | Attributed artifact-term list, deterministic expansion recipe and source license. |
| `helpdesk_training.json`, `agenda_training.json` | Separate 60-row training-only domain corpora. |
| `experiments/task_supplement.json`, `experiments/commit_supplement.json` | Training-only labelled vocabulary growth. |
| `model.json`, `commit_model.json`, `helpdesk_model.json`, `agenda_model.json` | Shipped v4 models; sparse on disk, no remote dependencies. |
| `subsystems/richer_features.js`, `subsystems/model_parameter_stats.js` | v4 feature banks and parameter accounting. |
| `train.mjs`, `score.mjs`, `model_stats.mjs` | Node training, inference and model inspection. |
| `index.html`, `serve.mjs`, `test.html` | Browser app, static server and subsystem tests. |
| `experiments/`, `TODO.md` | Reproducible measurements and development checklist. |
