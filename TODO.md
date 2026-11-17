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

## Tenfold vocabulary and quality follow-up — coverage implemented; accuracy remains open

- [x] User delegated the choice: keep 3,402 genuinely training-observed words distinct from a ≥34,020-word licensed offline lexical resource; make lexical relationships affect inference rather than padding `model.vocabulary`. Validate real quality impact before shipping.
- [x] Preserve the 524,289-parameter offline browser model, existing 3,686 labelled training examples and frozen evaluation partitions while implementing the selected coverage approach. WordNet's 34,402 recognised alphabetic words are distinct from 3,402 actually observed in labelled training.
- [x] Compare against the prior checkpoint on held-out task, unfamiliar-word, multilingual and cross-domain diagnostics; document the failed fresh-training candidate and unchanged accuracy of the grafted lexical head in `experiments/LEXICON_COVERAGE.md`.
- [x] Add browser/CLI lexical coverage and uncertainty disclosures, licensed reproducible build, and regression/browser checks; no accuracy improvement was demonstrated. Rebuilt with `npm run train`; model/anchor/lexicon/architecture/vocabulary/context/domain/legacy tests passed, browser smoke 32/32, subsystem test page 50 passed/4 API-dependent skipped, `git diff --check` passed.
- [ ] Collect real, independently labelled examples and a new locked external test set to improve generalisation and test whether lexical cues actually help. Existing developer-authored task and context examples are insufficient to claim an accuracy gain.

## Context-sensitive quality follow-up — correction path complete; default accuracy remains open

- [x] Try training-only context/specificity features rather than vocabulary-count changes; choose on the original validation partition before examining protected test/probe results. The best head improved validation 97/102 → 100/102 but regressed held-out 168/170 → 167/170 and unfamiliar 35/40 → 34/40; reject it rather than tuning on test results.
- [x] Make genuine user corrections learnable in the offline browser: keep the seed-1 base validation/test partitions fixed, include the existing 2,900 supplement rows and same-size anchor/lexical head, reject duplicate protected examples, and add `--feedback` for reproducible CLI training. See `experiments/QUALITY_FOLLOWUP.md`.
- [x] Verify one authored correction changes its own prediction (0.660 → 0.435) with the 170-case test still 98.8%; that is adaptation, **not a general accuracy claim**. CLI feedback regression and browser smoke (38/38, including atomic feedback reimport) pass. Model, anchor, lexicon, architecture, vocabulary, context, domain, legacy and subsystem tests (50 pass/4 skipped) passed; `git diff --check` passed.
- [ ] Collect independently labelled real-world corrections and a new locked external test; only then can a more accurate default checkpoint be selected without overfitting the inspected synthetic partitions.

## Real-text generalisation experiment — shipped, but independent accuracy remains open

- [x] Manually title-label and attribute 131 real Vite/Node training issues; keep Axios validation (46 rows after excluding a punctuation-only duplicate) and SvelteKit/Express diagnostic (30 rows) out of training. These are selected, not random or independently multiply annotated.
- [x] Select a 50% same-size anchor on original task + Axios validation, then inspect held-out sources once: real-title diagnostic 12/30 → 23/30, expanded task held-out 168/170 unchanged, unfamiliar authored set 35/40 unchanged, helpdesk-hard zero-shot 9/16 → 13/16. Do not tune on those test results.
- [x] Ship the selected one-model v4 checkpoint with 3,817 actual training rows, 3,867 training-observed alphabetic words / 34,679 recognised with WordNet, 524,289 potential parameters, and the single-mode offline browser. Preserve the previous tests/splits and add `npm run test:real` with URL/provenance/leakage guards. Rebuilt via `npm run train`, tested model/real/feedback/lexicon/anchor/architecture/vocab/context/domain/browser (38/38)/legacy/subsystem (50 pass, 4 skipped), and checked `git diff --check`. See `experiments/REAL_ISSUE_GENERALISATION.md`.
- [ ] Commission a genuinely sampled, multiple-rater, project-diverse locked test and more non-English/commit-specific data before claiming deployment accuracy or authenticity detection.

## Generalisation beyond curated issue-title splits — audit complete, smarter model still open

- [x] Label an attributed, fixed Fastify/ESLint 120-title audit before scoring; include both labels in each project, exclude nine ambiguous titles, assert no token-normalised overlap with all checked-in training/diagnostics. This is a selected-page, single-rater audit, **not** a representative or independently adjudicated deployment test. See `experiments/PROJECT_AUDIT.md` and `npm run test:blind`.
- [x] Diagnose confidence, training-OOV coverage, length, project and label. The shipped model gets only 55/111 unambiguous titles right (44 FP, 12 FN), and a fixed uncertainty-review rule leaves only 13/28 accepted titles correct. Historic curated task splits hid this weakness.
- [x] Test a training-only 131-title nearest-neighbour prior, selected on original/Axios validation, then inspect the now-reused audit: 55/111 → 58/111; original task validation 97/102 unchanged, Axios 32/46 unchanged. **At that stage**, reject the modest, extra-component alternative and keep the v4 browser model unchanged; a later correction-trained same-budget model is documented below.
- [ ] Obtain multiple-rater, source-balanced real task/commit/helpdesk data and a *new* locked, project/author/time-separated test before attempting a next release; do not treat this inspected audit as fresh evaluation. Compare balanced hard-negative training with same-budget context features before selecting on the new validation split.

## Correcting inspected cross-project failures — completed, real-world quality remains open

- [x] Confirm pre-correction Fastify/ESLint 55/111 (44 FP/12 FN), then move 111 unambiguous rows into **training only**; preserve the original audit as a historical pre-correction measurement and label its new 110/111 as training fit, not held-out improvement.
- [x] Before scoring, save source-attributed Babel *selection* and React/TypeScript/VS Code *once-locked test* title labels. Exclude 26 unclear/duplicate test rows; verify source/count/canonical and cross-partition near-duplicate guards. These are API-page-selected, one-developer labels, not a representative multi-rater study.
- [x] Fit one same-budget corrected model (3,928 actual training rows); select the same 50% anchor using only old original-task/Axios and new Babel validation. Then open the test once: **75/135 → 82/135**; FP **58→50**, FN **2→3**, each project improves. Original task test **168/170** and unfamiliar **35/40** unchanged, while inspected helpdesk-hard **13/16→12/16** regresses. The test is now inspected; no further tuning on it.
- [x] Ship a reproducible offline single-model v4 checkpoint at **524,289 scalars**, 60,167 nonzero weights, 4,080 training-observed alphabetic words and 34,791 recognised with the licensed lookup. Browser default retraining includes both real-title files, prevents duplicate user feedback, leaves imported corpora untouched; tests including real-server browser **41/41** passed. See `experiments/ISSUE_CORRECTION.md` for remaining severe false positives and limits.
- [ ] Get an independently annotated, source-/author-/time-diverse *new* locked test with enough vague titles, hard negatives and other policies before attempting another quality release. The inspected 135-title set cannot support further unbiased selection.

## Vocabulary-independent real-world generalisation — structural trial rejected for release

- [x] Freeze URL-attributed Vitest selection and Home Assistant/Flutter/Playwright/Deno locked title/commit partitions before scoring; exclude ambiguous rows and guard duplicates. These are single-rater, selected-page, title-only labels, not deployment estimates.
- [x] Prototype same-budget offline role/condition/failure features behind opt-in `structureVersion: 1`; verify unchanged old v4/v1–v3 scores and save/load. Select 0.4 mix on development rows only.
- [x] Open the first locked set once. Despite fewer genuine-title false positives (issues 37→18; commits 19→11), **reject version-1 release**: the original expanded-task test fell 168/170→167/170, below the predeclared floor. A later separately sourced, now-inspected Rust/CPython/Flask confirmation improved 54/85→67/85, but could not reverse the failed gate. At that point `model.json` remained correction-trained; see `experiments/STRUCTURAL_GENERALISATION.md`.

## Evidence-gated structural follow-up — release passed; deployment quality unproven

- [x] Freeze a separate four-project Go/Rails/Kubernetes issue + Bun commit subject set (230 titles, 188 clear) and validate sources and canonical separation before fitting. The set is selected by pages and one-rater, with only nine vague rows; it is not a deployment sample.
- [x] Restrict structural cues by symptom/code/condition evidence (version 2); reject on development data alone. Version 3 also enables abstract role patterns on longer text; select the 0.4 anchor on original task and commit validation, authored ordinary tasks and Vitest selection. Do not score the new set until selected.
- [x] Apply the unchanged first-trial release gate and predeclared new gate once: historical task **168/170 unchanged**, old project issues **47/89→66/89**, commits **38/57→46/57**; new issues **42/88→57/88**, commits **75/100→82/100**, combined FP **69→47**, FN **2→2**. Ship the single-mode offline v4 version-3 checkpoint with **524,289 scalars**, **60,881 nonzero weights** and reproducible default CLI/browser recipe. See `experiments/STRUCTURE_V3_RELEASE.md`.
- [x] Continue with opt-in version-4 clause-relation features and balanced training-only hard guards; select using separate Angular/Terraform/FastAPI development rows. The plain candidate failed development selection. The guarded candidate passed development but failed both predeclared release gates: new Pandas/Electron **104/143→127/143**, FP **37→13** but FN **2→3**; original task test FP **0→1**, and inspected four-project FN **2→6**. **Reject, do not ship** or revise thresholds post hoc. See `experiments/RELATIONAL_TRIAL.md`.
- [ ] Obtain multiple-rater, source-/author-/time-diverse data with enough vague and ambiguous hard cases before claiming deployment accuracy or using scores as automatic blocking decisions. Every present title set is now inspected. Any next candidate needs a new untouched evaluation partition.
