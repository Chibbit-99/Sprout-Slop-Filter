# Model improvement plan — completed

- [x] Inspect existing browser/CLI training, dataset, scoring path and evaluation limits (v1 averaged word vectors; one `<UNK>` for unseen words; task-only examples).
- [x] Implement lightweight offline v2 with subword, word and adjacent-word features; keep v1 models loadable and trainable.
- [x] Wire v2 through CLI, shared subsystems and browser; add a task/commit use-case selector and a scoring CLI.
- [x] Expand tasks to 684 labelled examples (1,050 distinct tokens); create a separate 199-example commit corpus and ship both trained models.
- [x] Add round-trip, OOV, order and transfer regression probes; run browser smoke and old subsystem checks; measure model size/speed. See `npm run test:model` and README for results/caveats.
- [x] Document reproduction, architecture, label policy and limitations.

## Follow-up: unfamiliar vocabulary and other use cases — completed

- [x] Add a fixed, labelled challenge set covering morphology, jargon, novel names/identifiers, multilingual text and changed context; assert what is actually out of training vocabulary.
- [x] Test zero-shot scoring and separate, domain-trained scoring on helpdesk reports and meeting agendas, with non-overlapping examples and ordinary/hard held-out sets.
- [x] Compare v2 with v1 where possible; record false positives, failures and artifact size without turning challenge cases into training data. Earlier `test:model` still reports inference timing.
- [x] Run regression and browser checks, publish reproduction commands and limits (including what 'genuine' cannot mean) in `experiments/CONTEXT_CHALLENGE.md`.

## Longer-term work

- [ ] Collect independently labelled real task-tracker text and real project commit subjects from multiple authors, with a **new** locked external test set. The current splits and hand-written probes were inspected during development, so they are not deployment-grade accuracy estimates.
- [ ] Validate confidence calibration, threshold and false-positive costs on each target workflow. Commit messages cannot prove authorship or correspondence with a code diff; that would require a different, diff-aware model.

## Parameter-budget architecture upgrade — completed

- [x] Add a versioned 131,073-trainable-parameter architecture with four disjoint feature banks; retain v1/v2 load and scoring.
- [x] Expose reproducible total vs nonzero parameter counts for shipped models in CLI (`npm run stats -- model.json`), browser and tests; document what each count means (9,874/4,971 nonzero weights).
- [x] Retrain task and commit artifacts; compare to v2 on identical training partitions and pre-existing held-out, transfer, OOV and cross-domain probes. More capacity did **not** raise held-out accuracy; commit false positives increased by one.
- [x] Run model, architecture, context, browser and legacy tests; document size, speed, counts, tradeoffs and limitations in `experiments/ARCHITECTURE.md`.

## 500k-parameter and vocabulary-growth follow-up — completed

- [x] Implement v4 with 524,289 trainable scalars across eight meaningful feature banks, preserving v1–v3 compatibility and counting nonzero weights separately.
- [x] Add balanced labelled task/commit supplements, freeze the original splits, assert no base/challenge leaks, and measure 0/partial/full vocabulary growth with `npm run test:vocab`.
- [x] Compare v3/v4 on the same selected training rows, plus OOV/context challenges. Task held-out stays 97.1%; commit rises from 90.3% to 93.5% (one example), with caveats and unchanged multilingual failure.
- [x] Update CLI/browser, artifacts, docs and regression tests; run model/architecture/vocabulary/context/browser/legacy suites. See `experiments/ARCHITECTURE_V4.md`.

## Browser compatibility and multiple-context accuracy

- [x] Verify `index.html` against the real static server (model/dataset/supplement fetches, boot, scoring, retrain, save/load and use-case switching); `npm run test:app`.
- [x] Give helpdesk reports and meeting agendas their own label policy, training-only data and versioned lightweight v4 models; do not train on their existing ordinary/hard tests.
- [x] Add 24 further non-overlapping, OOV-annotated examples per domain and compare zero-shot task/commit scoring against context-specific models, including mistakes and false positives (`npm run test:domains`). These are developer-authored, not an independent real-world test.
- [x] Wire browser and CLI context selection, update documentation and run regression/browser/legacy tests.

## Mode-free browser follow-up — completed

- [x] Remove situation dropdown and automatic dataset/model switching from `index.html`; keep one task-trained default and explicit import/load tools.
- [x] Keep domain models and their label-policy comparisons in the CLI and experiments, without implying the default browser model generalizes to them.
- [x] Update the real-server browser smoke test to assert the single workflow; recheck retraining, scoring, saving, reset and reload.

## Expanded default training coverage (requested)

- [x] Interpret "1000 training datasets" as ≥1,000 labelled examples actually in the training partition; keep a real, non-padded observed vocabulary of ≥3,000 alphabetic words (excluding `<UNK>` and punctuation).
- [x] Expand the default task corpus with balanced, labelled, reproducible examples; freeze the existing baseline and keep evaluation/probe rows out of training.
- [x] Retrain the shipped browser model, verify artifact size, browser counts and actual train-only vocabulary, and measure accuracy/false positives rather than assuming size helps.
- [x] Update regression tests and docs for the new default, rerun real-server browser (31/31), subsystem (49 pass, 4 skipped) and legacy tests. Freeze the original test split and document synthetic-data limitations in `experiments/TRAINING_EXPANSION.md`.

## Architecture efficiency and generalisation follow-up

- [x] Compare same-parameter training/weight-regularisation candidates against the expanded v4 baseline using original validation and independent-style challenge diagnostics; do not tune on held-out/probe scores.
- [x] Ship only a change that improves at least the chosen validation objective while preserving v1–v3 compatibility, ≥3,000 observed alphabetic words, ≥1,000 actual training examples and offline browser scoring.
- [x] Re-run model, architecture, vocabulary, context, domain, anchor, browser (31/31), subsystem (50 passed, 4 skipped) and legacy suites; report counts, FP/FN, unchanged cross-domain/multilingual limits and reproducible recipe in `experiments/ANCHOR_OPTIMISATION.md`. `git diff --check` passed.
