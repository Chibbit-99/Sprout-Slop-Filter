# Real issue-title generalisation (curated diagnostic)

The default offline task model now includes **131 public GitHub issue titles** as additional *training-only* rows: 117 title-specific bug reports/requests and 14 vague titles from Vite and Node.js. They are hand-labelled by the *title alone* under the existing task policy: `genuine` means the title names a concrete symptom, target or requested change, not that the issue is true; `slop` means a vague title with insufficient target/detail, not that its author is dishonest. Each row has a source URL in `real_issue_training.json`. The 1,124 displayed base examples and their seed-1 partitions remain unchanged; training now uses **786 base + 2,900 synthetic supplement + 131 real titles = 3,817 rows**. The model still has **524,289 trainable scalars**, uses one classifier, works offline in `index.html`, and preserves older model versions. The sparse artifact is 2,551,535 bytes, with 57,444 nonzero feature weights. **3,867 alphabetic words actually occurred in labelled training** (3,946 tokenizer tokens); the union with the offline WordNet lookup is 34,679 recognised alphabetic words. These counts are distinct.

## Source separation and selection

We manually picked and title-labelled public issues from mature projects, recorded the issue URLs, rejected exact/token-normalised duplicates across base, supplement, real training, validation and held-out data, and kept the validation/held-out projects out of training. This is **not a representative/random sample**: many genuine examples are technical, vague examples were deliberately selected, the training labels are imbalanced, and the held-out labels correlate with source project (20 specific SvelteKit titles, 10 vague Express titles). Labels are our judgements from titles, not independent human annotation. The titles are short attributed excerpts; source projects and their contributors retain their rights. They are not evidence of a valid issue, author identity or an actual code change.

- **Training only:** Vite and Node.js, 131 rows; no original base, supplement, challenge, validation or held-out duplicates.
- **Selection validation:** Axios, originally 47 rows. After candidate selection, a punctuation-only title was removed because it overlaps the base corpus under token-normalised matching; the reported set is **46 rows** (24 specific, 22 vague). No model parameters were changed after that correction.
- **Post-selection diagnostic:** SvelteKit + Express, 30 rows. It was evaluated **only after** choosing the real-training recipe, and was **not** used to pick the blend fraction.

A v4 candidate was trained with the same seed, 120 epochs, learning rate and temperature as the previous task model, with these 131 real rows appended after splitting. We compared anchor fractions on the *old task validation* and *Axios validation*, requiring no old-validation regression while seeking fewer real-title false positives. The chosen **50%** old task checkpoint mixes into the trained weights (one classifier at inference). The previous **45%** recipe is retained as a historical checkpoint; its real-title diagnostic was 22/46 after the punctuation-only row correction. The earlier WordNet class head is reattached to 45 slots, as before. The shipped model was not tuned on either task test split, the unfamiliar-word probes or the Svelte/Express diagnostic.

| Diagnostic | Previous model | New model |
| --- | ---: | ---: |
| Original task validation (inspected) | 97/102, FP 2/FN 3 | **97/102**, FP 2/FN 3 |
| Expanded task validation (inspected) | 163/168, FP 2/FN 3 | **163/168**, FP 2/FN 3 |
| Axios issue-title validation (curated) | 22/46, FP 22/FN 2 | **32/46**, FP 12/FN 2 |
| SvelteKit/Express post-selection titles (curated) | 12/30, FP 18/FN 0 | **23/30**, FP 6/FN 1 |
| Original task test | 102/104, FP 0/FN 2 | **102/104**, FP 0/FN 2 |
| Expanded task test | 168/170, FP 0/FN 2 | **168/170**, FP 0/FN 2 |
| Developer-authored unfamiliar contexts | 35/40, FP 5 | **35/40**, FP 5 |
| Helpdesk hard, task model zero-shot | 9/16, FP 6/FN 1 | **13/16**, FP 2/FN 1 |
| Agenda hard, task model zero-shot | 13/16, FP 3/FN 0 | **13/16**, FP 3/FN 0 |

These are **small, selected, repeatedly inspected diagnostics**, not unbiased accuracy estimates. In particular the issue-title test composition is confounded with project and label; it cannot establish deployment accuracy or prove the classifier understands semantics. The four actionable non-English unfamiliar probes still fail, and a generic title with a plausible issue number can still be misclassified. A credible next step is an independently sampled, multiply annotated, author/project-diverse corpus with a locked evaluation partition and application-specific false-positive costs. A commit-message score still cannot establish authenticity or diff correspondence.

## Reproduce (no network at runtime)

```sh
npm run train           # base + synthetic supplement + 131 real titles; 50% prior + offline lexicon
npm run test:real       # attributed dataset partition guards and diagnostic comparison
npm run test:model && npm run test:feedback && npm run test:app
```

`experiments/task_real_unanchored_model.json` is the deterministic pre-blend checkpoint for weight-space regression. `npm run train` uses only checked-in files; Node needs no installed packages. Browser default retraining fetches the same real-title training-only file (and the synthetic supplement), retains the split and one-model anchor, and copies the compact lexical head. Imported/custom corpora do not inherit those rows or the prior. To add your own labelled corrections **without** losing the 131 real titles, either use the browser's Import feedback workflow, or pass `--feedback experiments/real_issue_training.json --feedback your_feedback.json` to `train.mjs` with the same `--extra` and anchor settings. `--feedback` is repeatable and rejects base/supplement/feedback duplicates.
