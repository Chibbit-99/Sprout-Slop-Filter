# Sprout: Slop Filter

A small **offline** classifier for generic filler versus informative text, implemented in plain JavaScript for Node and browsers. Its default v4 model has **524,289 trainable parameters** (524,288 feature weights + one bias), while the expanded task and commit model files are **2.55 MB and 300 KB** respectively. Capacity is not knowledge: most coefficients remain zero after training.

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

`index.html` has **no situation modes or preset selector**: it loads the task-trained `model.json` and `training.json` and scores text with that single label definition. You can score individual or batch text, edit/import labelled examples, retrain, and save/load models. The default dataset contains **1,124 labelled examples**, and the shipped model fits **3,817 training rows** (786 seed-1 base train + 2,900 synthetic training-only rows + 131 manually title-labelled real GitHub issues) with **3,867 distinct alphabetic words** (3,946 observed tokenizer tokens including punctuation and numbers), excluding the `<UNK>` sentinel. The original 684-row held-out partitions are preserved. In addition, a licensed offline WordNet lookup gives **34,679 recognised alphabetic tokens total**: **only 3,867 were observed in labelled training**. Each lexical match supplies a small, trained category/synonym cue, not a labelled example or an accuracy guarantee. [Lexical coverage, licensing and diagnostics](experiments/LEXICON_COVERAGE.md). Default retraining blends in a same-size, old task checkpoint as a training-time prior (50% weights) to mitigate synthetic-template overfitting; inference still uses one model and the same parameter count. The 45% recipe in [anchor optimisation](experiments/ANCHOR_OPTIMISATION.md) is historical; [real issue-title generalisation](experiments/REAL_ISSUE_GENERALISATION.md) records the new selection and its limits. The browser's supplement option reproduces that setup on the unedited base dataset and reuses the shipped 45-slot lexical class head; disable it to train on base rows only. **Correcting a scored task** now adds a training-only row without dropping supplements or changing the original validation/test partition; click Train model to apply it, and download/import feedback to keep it across visits. Existing base/supplement/real-title duplicates and split changes with feedback are blocked. On imported datasets, task-specific supplements and priors are skipped. [Correction workflow and rejected context-feature experiment](experiments/QUALITY_FOLLOWUP.md). All scoring is local and needs no word API; optional Datamuse applies only to the legacy v1 encoder when explicitly enabled. Other domain-specific models remain available as separate CLI experiments, not as modes in the page. Most of the supplemental rows remain templated from WordNet artifact names; the 131 real issue titles are attributed, manually selected and still insufficient for a deployment estimate. See [data provenance](experiments/TRAINING_EXPANSION.md) and [real-title diagnostics](experiments/REAL_ISSUE_GENERALISATION.md).

**A commit-message score measures informativeness, not authenticity.** It cannot know if code changed, whether the message describes it truthfully, or who wrote it. `genuine` in `commit_training.json` means a specific change is named; `slop` means the message is generic. The task policy instead concerns an actionable request with a concrete target, regardless of ethics. Helpdesk and meeting agendas use *different* policies: reports with concrete symptoms/errors versus requests needing diagnostic detail, and specific decisions/subjects versus vague agenda placeholders. Each experimental domain model uses its own 60 training-only rows, not the task/commit labels. Neither shipped domain model has an independent held-out accuracy estimate; the page does not load them automatically. See [domain-model diagnostics](experiments/DOMAIN_MODELS.md) and their limitations.

## Architecture and parameter count

- v4 encodes word unigrams, character 3-, 4-, and 5-grams, ordered adjacent and skip word pairs, first/last-word position, and prefixes/suffixes into **eight disjoint 65,536-weight hash banks**. Banks prevent collisions between feature *types* (collisions within a bank remain). These are genuine addressable trainable coefficients, not inert padding. Pairs provide limited local context and subwords provide spelling clues for unseen words. Input is capped at 128 tokens and 64 characters per token.
- A logistic classifier uses per-feature AdaGrad updates. **8 × 65,536 + 1 bias = 524,289 trainable scalars per model.** But only **57,444 expanded task weights / 9,797 commit weights are nonzero** after training. Vocabulary strings and optimizer accumulators are *not* trainable parameters. Sparse JSON stores only the nonzero weights; loading expands them into a dense array. In-memory numeric payload for weights plus training accumulators is roughly 8 MiB, excluding JS object overhead. See [parameter accounting and v3 comparison](experiments/ARCHITECTURE_V4.md).
- The default task recipe uses **same-size weight-space anchoring**, not extra inference parameters or a second runtime model. After fitting the expanded corpus, it blends 50% of the frozen pre-expansion v4 checkpoint into the coefficients. See [real-title selection and tradeoffs](experiments/REAL_ISSUE_GENERALISATION.md); the original 45% selection is historical. Other domain models are not blended.
- Older v1/v2/v3 models remain loadable; `--encoder legacy`, `--encoder hashed-subword-v2` and `--encoder structured-subword-v3` still train them. V4 is `--encoder richer-subword-v4` (the default). The v4 width is fixed by its eight banks. `model_parameter_stats` in `subsystems/` is shared by Node, tests and the browser. The `vocabulary` array in v2–v4 records observed training tokens for coverage/display; a new token still receives subword features.

**Subword overlap is not semantic understanding.** Non-English actionable examples are still wrongly flagged. Project jargon, negation, misleading detail, and new contexts remain difficult. A score is not a calibrated probability of truth. Review borderline cases and test independently labelled real data before using this as a gate.

## Retrain and test

```bash
npm run build:task-data        # regenerate JSON from frozen baseline + attributed artifact terms
npm run train                  # task + 2,900 synthetic rows + 131 real titles; 50% prior + offline lookup
npm run train:commits          # v4 commit model: fixed base split + 80 supplemental train rows
node train.mjs --data my_data.json --out my_model.json  # no supplement by default
node train.mjs --extra experiments/task_supplement.json --feedback experiments/real_issue_training.json --feedback task_feedback.json --anchor-model experiments/task_baseline_model.json --anchor-mix 0.5 --out model.json
node experiments/assemble_lexicon_model.mjs   # reattach offline class head after CLI feedback training
node train.mjs --encoder structured-subword-v3 --data my_data.json --out old_model.json
node train.mjs --help
npm run test:model             # versioned load/save, counts, OOV and round trips
npm run test:anchor            # historical 45% prior selection, before real-title training
npm run test:architecture      # historical baseline v3/v4 comparison, frozen 684-row set
npm run test:vocab             # 0 / partial / full supplemental vocabulary, frozen splits
npm run test:lexicon           # isolates the lexical head within the new checkpoint
npm run test:real              # real issue-title cross-repository diagnostic and provenance guards
npm run test:feedback          # feedback stays training-only; no split or duplicate leakage
npm run train:contexts         # deterministic helpdesk and agenda artifacts (60 rows each)
npm run test:domains           # leakage-checked ordinary/hard/new domain diagnostics
npm run test:contexts          # new-word and non-commit-context challenge
npm run test:app               # browser smoke test (npm install --no-save jsdom)
npm test                       # historical legacy learning curve
```

`train.mjs` uses seed-1 70/15/15 base splits and builds the observed vocabulary from **only** the training partition plus optional `--extra` and repeatable `--feedback` rows. Feedback rows must not duplicate any base or supplemental example (including punctuation-only variants); the task anchor requires seed 1. `--extra-count N` takes the first N supplement rows; the provided files alternate genuine/slop examples so prefixes remain balanced. The trainer rejects duplicate supplement sentences from *any* base split. Default training is 120 epochs, learning rate 0.8 and score-softening ×4. Softening scales weights and bias together and does not affect decisions at 0.5. `--vector-size` applies to v1 only. Dataset files are JSON arrays of `{ "text": "...", "label": "genuine" | "slop" }`. The scoring CLI accepts `--domain tasks|commits|helpdesk|agendas`, `--model path` and `--threshold 0..1` and prints JSON; the verdict describes the selected *text policy*, not authenticity. The web page must be served over HTTP to fetch JSON assets.

## Measurements (diagnostic, not deployment estimates)

| Model | Trainable (nonzero weights) | Held-out split | Transfer probes | On disk |
| --- | ---: | --- | ---: | ---: |
| Real-title augmented anchored tasks: 786 base + 2,900 synthetic + 131 real | 524,289 (57,444) | **98.8%**, TP 78 / FP 0 / TN 90 / FN 2 (170 cases) | 20/20 | 2,551,535 bytes |
| Historical task baseline (archived) | 524,289 (15,719) | **97.1%**, TP 48 / FP 0 / TN 53 / FN 3 (104 cases) | 20/20 | 480,798 bytes |
| Commits: 139 base train + 80 supplement | 524,289 (9,797) | **93.5%**, TP 13 / FP 1 / TN 16 / FN 1 (31 cases) | 20/20 | 299,609 bytes |

The expanded split retains the 104 historical task test rows, on which the expanded model scores **102/104 versus the historical model's 101/104**; these data and probes have been inspected during development, so this is **not an independent real-world improvement**. Without anchoring, synthetic vocabulary expansion worsened the separate 40-case unfamiliar-word diagnostic from **34/40 to 31/40**. The same-size prior brings it to **35/40**, but all four actionable non-English inputs remain false positives. On the developer-authored validation partition, anchoring holds accuracy at 97/102 while trading one false negative for one fewer false positive. Neither vocabulary size nor parameter count guarantees accuracy. The new real-title training improved a small, handpicked cross-repository diagnostic from **12/30 to 23/30** while preserving task test scores; this is **not** a random or deployment estimate ([report](experiments/REAL_ISSUE_GENERALISATION.md)). The freshly trained WordNet-cue candidate **regressed** on validation (162/168 vs 163/168); the released class-head graft preserves the previous inspected binary results (including the 35/40 unfamiliar-word diagnostic) but has **not demonstrated an accuracy gain**. [Coverage report](experiments/LEXICON_COVERAGE.md). The [training expansion report](experiments/TRAINING_EXPANSION.md) documents the provenance, split-locking assertions, false positives and tradeoffs. The [historical v4 vocabulary-growth report](experiments/ARCHITECTURE_V4.md) and [context report](experiments/CONTEXT_CHALLENGE.md) use the frozen baseline; `npm run test:domains` compares the current task model against it and the separately trained helpdesk/agenda models. All are developer-authored diagnostics, **not deployment estimates**.

## Files

| Path | Purpose |
| --- | --- |
| `training.json`, `commit_training.json` | Labelled corpora, split before supplemental training. |
| `experiments/task_baseline_training.json`, `experiments/task_baseline_model.json`, `experiments/task_unanchored_model.json`, `experiments/task_real_unanchored_model.json` | Frozen historical and current pre-blend checkpoints for reproducible comparisons; not needed at inference. |
| `subsystems/blend_models.js` | Same-size task weight-space prior, used during default CLI/browser retraining. |
| `experiments/task_artifact_terms.json`, `experiments/build_task_expansion.mjs`, `experiments/WORDNET_LICENSE.txt` | Attributed artifact-term list, deterministic expansion recipe and source license. |
| `helpdesk_training.json`, `agenda_training.json` | Separate 60-row training-only domain corpora. |
| `experiments/task_supplement.json`, `experiments/commit_supplement.json`, `experiments/real_issue_training.json` | Training-only labelled examples; real issue-title sources and limits in the real-title report. |
| `model.json`, `commit_model.json`, `helpdesk_model.json`, `agenda_model.json` | Shipped v4 models; sparse on disk, no remote dependencies. The task model bundles the offline WordNet lookup and its notice. |
| `experiments/wordnet_lexicon.json`, `experiments/lexicon_class_weights.json` | Offline lexical resource and 45 learned class weights; builder/assembler/reproduction in the coverage report. |
| `subsystems/richer_features.js`, `subsystems/model_parameter_stats.js` | v4 feature banks and parameter accounting. |
| `train.mjs`, `score.mjs`, `model_stats.mjs` | Node training, inference and model inspection; `--feedback` appends training-only corrections. |
| `index.html`, `serve.mjs`, `test.html` | Browser app, static server and subsystem tests. |
| `experiments/`, `TODO.md` | Reproducible measurements and development checklist. |
