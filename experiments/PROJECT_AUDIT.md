# Historical pre-correction audit (2026-11-11)

**Historical snapshot, not a test of the current model.** This audit originally led us to retain the then-current `model.json` (524,289 potential parameters, 3,817 training rows, 3,867 training-observed alphabetic words and 34,679 recognised words) because a retrieval head offered too little benefit. Subsequently, its 111 unambiguous titles were put into **correction training** and a separate three-project test was labelled and opened once; see [the follow-up and its remaining errors](ISSUE_CORRECTION.md). The figures below describe the *old* checkpoint unless explicitly marked as training fit. Do not report these 111 titles as a current held-out accuracy score. Adding recognised words alone does not teach which real reports are informative.

## Reproducible, source-attributed title audit

`blind_issue_audit.json` records the issue URL, exact API title and a single developer's title-only label. We collected non-PR issue listings from GitHub's Fastify pages 10/20/30/40 and ESLint pages 20/30/40, retained the **first 60** in each concatenated listing in API order, then labelled **all 120 before running this audit**. These are selected API pages, **not** a uniform or prospective sample of current submissions; titles alone omit the issue body. The title-only task policy is: `genuine` identifies a particular symptom, component, outcome or requested change; `slop` is too vague to identify the change/diagnostic target. Neither label judges truth, authorship or whether the issue is worth fixing. We excluded nine `unclear` cases from metrics rather than forcing a binary answer. Counts: Fastify 44 genuine / 12 slop / 4 unclear; ESLint 46 / 9 / 5. Labels are subjective, single-rater judgements, particularly for concise feature requests and issue-template prefixes; some could reasonably be reversed by a project maintainer.

`npm run test:blind` reconstructs the exact **pre-correction** checkpoint and the historical pre-real-title checkpoint; it also prints the current model's **training fit**, not a new test. Its assertions check both sources, counts, allowed labels and token-normalised title uniqueness against each other and earlier checked-in task/commit training and diagnostics (including nested context and unfamiliar-word probes). This does **not** check semantic near-duplicates, issue bodies or authorship. The selected pages and labels have now been used for training; **never reuse them as a fresh independent test.**

| Project and label | Historical correct | Current correct | Current mistakes |
| --- | ---: | ---: | ---: |
| Fastify genuine (44) | 2 | 19 | **25 false positives** |
| Fastify vague (12) | 9 | 8 | **4 false negatives** |
| ESLint genuine (46) | 1 | 27 | **19 false positives** |
| ESLint vague (9) | 9 | 1 | **8 false negatives** |
| **Both, unambiguous (111)** | **21** | **55** | **44 FP / 12 FN** |

The new training did improve this diagnostic relative to the historical checkpoint, but **55/111 is not good enough** to trust on unfamiliar project titles. The old 97/102 task validation or 168/170 expanded task test was disproportionately authored from other templates and cannot override these failures. Nor is 55/111 an estimate of real-world prevalence-adjusted accuracy: genuine titles outnumber vague ones 90:21 here; the page ranges and label policy were chosen by us.

A score of **0.677** for the concrete Fastify title “docs: missing docs about warnings or deprecation warnings” and **0.630** for “Method 'HEAD' already declared for route - regression in 4.24.0” are genuine false positives. “Bug: (fill in)” at **0.432** and “Lots of broken links in the docs” at **0.386** are vague false negatives. All scores are model outputs, **not calibrated probabilities**. Original-training-vocabulary OOV tokens comprise 126/472 Fastify genuine and 125/597 ESLint genuine tokens; the model's larger *recognised* lexicon does not turn those into labelled examples. Among genuine titles with at least 25% training-OOV tokens, **27/47** were false positives versus **17/43** for less novel titles. Length also matters: **6/7** genuine titles shorter than six tokenizer tokens were falsely flagged, compared with **38/83** longer ones. These are post-hoc descriptive slices, not tuned classifier thresholds or causal effects. Vague titles often contain technical words too: 9/12 low-OOV vague titles were false negatives.

A fixed abstention trial (review scores in **[0.4, 0.6]** or titles with at least **50%** training-OOV tokens) deferred **83/111**, yet only **13/28** accepted titles matched these single-rater labels. This is not a calibrated risk policy or a cure for misclassification. No change to the browser's threshold or confidence label follows from it.

## Compact alternative tested but not shipped

`node experiments/retrieval_trial.mjs` builds an IDF-weighted word/adjacent-pair/character-trigram similarity prior from the **131 Vite/Node training-only** titles and adds its signed nearest-neighbour evidence to the existing model's logit. It tests 72 compact configurations using **only the old original-task and Axios selection validation**; no audit title is added to the retrieval index or used to choose the displayed candidate. The best validation-compatible config (`k=1`, similarity floor 0.1, weight 0.25) leaves original task validation at **97/102**, and Axios at **32/46**, exchanging one false positive for one false negative. On the **already inspected** Fastify/ESLint audit, it changes **55/111 → 58/111** (FP 44→42; FN 12→11); the older inspected SvelteKit/Express set remains **23/30**. This small, source-fragile effect comes at the cost of a second inference component and an extra dataset/index; the audit has also been examined during development. **It was rejected.** No new model is saved, and browser/parameter compatibility are unchanged.

## Historical recommendations (partially acted on in the follow-up)

The correction release acted on part of item 2 using the audit as development data and a newly labelled test. That test has since been inspected and is **not** a new locked test for further tuning; its labels remain single-rater and page-selected. The multi-rater, representative study in item 1 is still outstanding.

1. **Define the policy with users, then obtain independently labelled real examples.** Stratify *within* several projects, authors and time periods for concrete/vague titles, including short generic bug reports and concise but genuinely actionable change requests. Give annotators the title and separately the full issue context; report disagreements and `unclear` explicitly. Include genuine/slop from **each** project to avoid learning repository identity as the label. Collect real task, helpdesk and commit examples under their *distinct* policies rather than calling all real text the same task. For commits, a message alone cannot establish diff correspondence or authenticity.
2. **Freeze new partitions before fitting.** Partition by project/author/time, deduplicate exact, token-normalised and near-duplicate titles across training, selection and a *new* locked test. At the time of this recommendation the audit was not training data; it was subsequently repurposed as explicit correction training, so its old metrics can no longer validate a new model. Do not train on the old Axios/SvelteKit diagnostics. Choose model/hyperparameters on training/validation only, then run the locked test once, with per-project FP/FN and bootstrap uncertainty. Get an independent annotation for borderline titles. The existing audit is now a development diagnostic, not the future locked test.
3. **Train for context, not vocabulary counts.** Compare the existing hashed word/subword/pair model to a same-budget regularised candidate using balanced real positives/negatives, hard negative mining *from training sources only*, and explicit evidence that a title names a failure condition, component or requested behaviour. Measure whether new spelling/combinations generalise to *unseen* projects instead of merely recognising more terms. Any extra features must preserve v1–v3 loading, v4's 524,289-scalar budget and offline `index.html` scoring.
4. **Only then calibrate the decision.** On new per-workflow validation, measure FP/FN costs, calibration and review coverage; keep human review as an explicit option. Do not present the present 0.5 score or the margin trial as a probability of truth or safe auto-approval.

Reproduce locally (no network needed):

```sh
npm run test:blind
node experiments/retrieval_trial.mjs
npm run test:real
```
