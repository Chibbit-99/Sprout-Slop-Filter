# Clause-relation fine-tuning trial — rejected for release

**Decision: keep `model.json` and the default browser/CLI recipe unchanged.** A same-budget experimental `structureVersion: 4` encoder and a development-selected, balanced-guard training candidate were built, but the candidate **failed both the historical class-wise release floor and the predeclared new-project gate**. Do not use the higher headline accuracy alone to promote it. All sets described here have now been inspected.

## Hypothesis and partition discipline

The shipped version-3 model still flags informative component-prefixed real titles as filler (for example, a `module: behaviour when condition` report). A prefix such as `fix:` or a package name is not by itself a reason to classify the remainder as actionable: the model should attend to the *relation between prefix, clause, target, failure and condition*. `structureVersion: 4` adds hashed, word-identity-independent features for the role of the post-colon clause, coarse detail count, code-shaped prefix and failure/condition order. It retains all version-3 features and uses the **same 524,289 scalar budget and 0.5 threshold**. Old v4 checkpoints without this option and the shipped version-3 checkpoint retain their original feature maps and scores. The role inventory still contains English trigger words, and length and punctuation are crude cues, not semantic reasoning.

Before fitting or opening new scores, `relational_development.json` and `relational_locked.json` were saved from disjoint GitHub API pages: Angular and Terraform issue titles plus FastAPI commit subjects for development; Pandas issue titles plus Electron commit subjects for the locked partition. Single-rater labels were assigned from titles/subjects, ambiguous and canonical-duplicate rows excluded. The natural locked sample contains **146** rows: **135 informative, 8 vague, 3 unclear**, so only **143** were scored. Development has 150 rows but only 67 clear (61 informative, 6 vague). The separate authored, source-separated `relational_*_pairs.json` files contain **8 development pairs and 10 locked pairs** of unknown identifier-bearing specific reports versus generic requests. These controlled negatives are **not** actual project titles and were scored separately. `relational_partition_guard.mjs` checks sources, counts, URLs, duplicates against all prior fit/selection/test sets, and locked-file SHA-256 digests. The final locked data were never used in fitting or anchor selection.

Two candidates were fitted with the same original task/issue rows as version 3, using the new feature view:

1. **Plain version 4:** anchor mix 0.2 improved development issues **34/47→40/47**, commits **12/20→14/20**, but task validation fell **97/102→96/102**. No fraction met the development rule; the locked set remained closed.
2. **Guarded version 4:** before refitting, 24 balanced, authored **training-only** rows (`relational_training_guards.json`) contrasted concrete ordinary requests and generic corporate filler. This is an explicit synthetic fitting intervention, *not* evidence of external performance. The selected anchor mix was **0.2** on development only: task validation **97/102 unchanged**, commit validation **19/29→21/29**, issues **34/47→40/47** (FP 10→4, FN 3→3), commits **12/20→14/20** (FP 8→6, FN 0→0), and authored development pairs **15/16 unchanged**. The selected model remained in ignored `.cache/`; it never replaced `model.json`.

The fresh release rule was written in `relational_release_trial.mjs` **before** scoring the Pandas/Electron and authored locked data. It required ≥2 extra correct on each natural source, ≥6 extra correct and ≥5 fewer FPs combined, at most one extra FN in each source, no project loss >2, and no drop in authored pair accuracy or correct ordering. The inspected regression floor additionally required the historical expanded task test to remain ≥168/170 **with zero false positives**, unfamiliar authored contexts ≥36/40, helpdesk-hard ≥12/16, and the previously opened four-project set not to lose >2 correct or gain >1 FN. A failure cannot be rescued by changing the gate after seeing results.

| Diagnostic (shipped v3 → guarded v4) | Correct | False positives | False negatives |
| --- | ---: | ---: | ---: |
| New Pandas issue titles (43 clear) | **40 → 41** | 3 → 1 | 0 → 1 |
| New Electron commit subjects (100 clear) | **64 → 86** | 34 → 12 | 2 → 2 |
| New natural titles combined (143) | **104 → 127** | **37 → 13** | **2 → 3** |
| New authored contrast pairs (20) | 18 → 18 | 1 → 1 | 1 → 1 |
| Previously inspected expanded task test (170) | 168 → 168 | **0 → 1** | 2 → 1 |
| Previously inspected four-project set (188) | 139 → 171 | 47 → 11 | **2 → 6** |
| Previously inspected unfamiliar authored contexts (40) | 36 → 36 | 4 → 4 | 0 → 0 |
| Previously inspected helpdesk-hard (16) | 12 → 14 | 2 → 0 | 2 → 2 |

The locked authored pairs retained **10/10 correct relative order**, but their class accuracy did not improve. The fresh issue requirement also failed: only **+1** correct (rather than the predeclared +2), with a new missed vague title. More decisively, the historic task zero-FP floor failed and the earlier four-project set gained **four** false negatives. This is the same kind of specificity–recall trade-off the previous release gate was meant to catch. **Both gates failed.** No second candidate was selected using the opened Pandas/Electron set. New natural titles, the authored pairs, and previous project diagnostics are *all inspected now* and cannot be called a fresh test for future tuning.

## Reproduce and limitations

From a clean checkout, use:

```bash
mkdir -p .cache
node experiments/relational_partition_guard.mjs       # no scoring
node train.mjs --structure-version 4 --extra experiments/task_supplement.json --feedback experiments/real_issue_training.json --feedback experiments/issue_corrections_training.json --feedback experiments/relational_training_guards.json --anchor-model experiments/task_baseline_model.json --anchor-mix 0.5 --out .cache/relational-guarded-half.json
node experiments/assemble_lexicon_model.mjs .cache/relational-guarded-half.json
node experiments/relational_selection_trial.mjs --guards
node experiments/relational_release_trial.mjs         # deliberately exits nonzero: rejected
```

The unguarded candidate was fitted by omitting the last `--feedback` file and writing `.cache/relational-plain-half.json`; `relational_selection_trial.mjs` without `--guards` rejects it before any locked scoring. No new checkpoint was shipped; `npm run train`, `model.json`, the offline browser and all older-model compatibility remain version 3. The role inventories were moved from per-prediction allocations to shared constants; a side-by-side comparison against the pre-refactor feature extractor found **0 exact-score differences on 92 inputs** for the shipped model. Paired local timings varied, so no speedup is claimed. `npm run test:model`, `npm run test:release` and `npm run test:relational-data` pass. The GitHub API pages were manually selected and title-labelled by one rater; even the new natural set has only eight vague examples and no adjudication or checked code diffs. The controlled pairs are synthetic. Do not call these accuracies production estimates or use the model as an automatic gate. A future trial needs a new untouched, sufficiently both-class, independently reviewed evaluation partition; simply shifting the class trade-off or lowering the zero-FP floor would not be a demonstrated improvement.
