# Correcting inspected issue-title errors without changing the offline model budget

The default v4 task classifier now includes **111 previously inspected Fastify/ESLint issue titles** as attributed training-only corrections (90 concrete, 21 vague). The older `PROJECT_AUDIT.md` records the *pre-correction* diagnosis; its 111 unambiguous titles are **no longer a test**. No old task validation/test or old Axios/SvelteKit examples were added to training. This is still a title-*specificity* classifier, **not** an authenticity detector, an issue-validity judge, or a calibrated probability.

## Data, separation and predeclared decision

- **Fitting:** 786 original task training + 2,900 synthetic training-only examples + 131 Vite/Node issue titles + **111 Fastify/ESLint correction titles = 3,928 actual training rows**. The nine ambiguous Fastify/ESLint audit rows remain excluded. Correction URLs and the frozen original labels are in `issue_corrections_training.json`; the old 120-row audit is retained for historical reproducibility. The browser and CLI default recipes load both real-title files only for the unedited task corpus. On user-imported corpora they are skipped. New user feedback cannot duplicate any correction, synthetic row, base row or protected task partition.
- **Selection, not training:** Axios's already-inspected 46 titles and 34 Babel issue titles (page 30 of GitHub's all-state issue listing, excluding pull requests; 27 concrete, 2 vague, 5 title-ambiguous). We labelled the Babel list *before scoring it*. One historical original-task validation slice (102 examples) is also a selection guard. The 45 WordNet class coefficients are reattached after training/anchoring, without another inference component.
- **Once-locked test:** 56 React issue titles from pages 20/30, the first 60 TypeScript issue titles from page 30, and 45 VS Code issue titles from page 50 of their respective all-state issue listings. We saved URLs/titles/labels in `fresh_issue_test.json` **before scoring a candidate**. We excluded 26 `unclear` rows (off-topic/spam, ambiguous titles, and duplicate/near-duplicate titles, with reasons), leaving **135 evaluated titles: 114 concrete and 21 vague**. Each project contains both labels: React 40/3, TypeScript 45/5, VS Code 29/13. The sample was **selected by API page**, not randomly or prospectively sampled, and **one developer** assigned all title-only labels. Some auto-generated TypeScript error-summary titles may be actionable with issue-body context even though the title itself is vague. No author or issue-body data was reviewed.

`issue_partition_guard.mjs` checks source URLs, class counts, exclusions, cross-partition canonical duplicates against prior training/challenges and pairwise Jaccard >=0.85 for nontrivial cross-partition near-duplicates. It does not establish independent labelling or eliminate all semantic paraphrases. The fresh test was opened only after selecting one candidate. It is **now inspected and must not be used to select future changes**.

Training kept the same 120 epochs, learning rate 0.8, seed 7, score temperature 4, v4 features and **524,289 trainable scalars**. The unanchored checkpoint is `task_corrections_unanchored_model.json`. `select_issue_candidate.mjs` evaluates weight-space anchor fractions 0.25/0.35/0.45/0.50/0.60/0.70/0.80 using **only** original task/Axios/Babel validation. It requires at least the old original-task result and FP count, at least the old Axios result and no higher Axios FP, and no more than one additional Babel error or FP. It maximises combined Axios+Babel accuracy (then FP, then stronger anchor). **0.50** was selected: original task **97/102 unchanged**, Axios **32/46 → 34/46** (FP 12→9, FN 2→3), Babel **17/29 unchanged** (FP 12, FN 0). The Babel sample is especially small and mostly genuine; this is weak evidence for parameter selection.

Before opening the test, the go/no-go rule required **at least five fewer test false positives**, **at most two extra false negatives**, no project losing more than two correct titles, and no regression on the previously inspected 168/170 expanded-task test and 35/40 unfamiliar-context probes. The chosen candidate passed that gate:

| Once-locked project/title test | Old correct | New correct | Old FP/FN | New FP/FN |
| --- | ---: | ---: | ---: | ---: |
| React (43 labelled) | 22 | 26 | 21 / 0 | 17 / 0 |
| TypeScript (50) | 26 | 27 | 24 / 0 | 22 / 1 |
| VS Code (42) | 27 | 29 | 13 / 2 | 11 / 2 |
| **Total (135)** | **75** | **82** | **58 / 2** | **50 / 3** |

This is a **modest seven-title net improvement**, not a solved classifier or deployment accuracy estimate. Across the 111 correction-*training* titles, the fit rises **55/111 → 110/111** (FP 44→1, FN 12→0), which mostly demonstrates adaptation/memorisation and **must not be quoted as unseen generalisation**. One correction still fails: Fastify [“Add hooks in app.register() options.”](https://github.com/fastify/fastify/issues/5146) scores **0.521** (filler). Prior inspected task test stays **168/170** (FP 0/FN 2), unfamiliar contexts stay **35/40**, Axios changes as above and SvelteKit/Express stays **23/30**. The previously inspected helpdesk-hard task zero-shot diagnostic **regresses 13/16 → 12/16** (FP 2→3). The 4/8 multilingual challenge is unchanged. The commit model is unchanged; commit-message scores still cannot verify identity or diff correspondence.

Failures remain severe: the new checkpoint falsely flags the concrete React title [“View transitions don't work in Firefox”](https://github.com/react/react/issues/34952) with a slop score of **0.709**, and TypeScript's [“Contextual typing doesn't work when mixing up discriminated unions with function variant”](https://github.com/microsoft/TypeScript/issues/62306) at **0.696**. It misses vague VS Code [“The Terminal”](https://github.com/microsoft/vscode/issues/343792) at **0.457**. Several labels are debatable from a title alone. The model is **not safe as an automatic issue gate**; the score is not calibrated as a probability of truth.

## Size, browser behaviour and reproduction

`model.json` is **2,634,540 bytes**, with **60,167 nonzero weights** of 524,288 slots and one bias; this remains a **single offline classifier** in a browser with **no situation modes**. It observed **4,080 distinct alphabetic words** (4,170 tokenizer tokens including punctuation/numbers) in labelled training; with the separately licensed WordNet resource it recognises **34,791 alphabetic words**. Lexical recognition is not labelled training. Old v1/v2/v3 checkpoint compatibility and domain-specific CLI experiments are unchanged. The real-server browser smoke exercises loading, default retraining with both issue files, correction deduplication, reset, save/load and custom-dataset isolation (41 passed).

```sh
npm run train                     # reproduces one same-budget checkpoint offline
npm run test:issues               # 135-title formerly locked test; now inspected
npm run test:blind                # historical pre-correction audit + CURRENT TRAINING fit
npm run test:real && npm run test:model && npm run test:lexicon
npm run test:app                  # install optional jsdom to actually run browser smoke
```

To reproduce the *selection* rather than merely the chosen checkpoint, run the same trainer arguments into an ignored scratch artifact, attach the fixed lexical head, then select on validation (not test):

```sh
mkdir -p .cache
node train.mjs --extra experiments/task_supplement.json --anchor-model experiments/task_baseline_model.json --anchor-mix 0.5 --feedback experiments/real_issue_training.json --feedback experiments/issue_corrections_training.json --out .cache/candidate-050.json --quiet
node experiments/assemble_lexicon_model.mjs .cache/candidate-050.json .cache/candidate-050.json
node experiments/select_issue_candidate.mjs
```

`.cache/` is ignored. **Do not rerun selection against the now-inspected test.**

## Further improvement requires new evidence

The high false-positive count on three new projects is the key unsolved issue. Get multiple independent raters and real issue *body* context from a source-/author-/time-balanced sample, include hard negatives and concise genuine requests within every training project, and freeze a **new** test from other sources before fitting again. Measure per-class FP/FN, disagreement and calibration under the intended workflow. For a commit authenticity claim, add the diff and provenance to the input rather than pretending a text-only classifier can infer them. Do not tune further on these React/TypeScript/VS Code titles or the historical audit and describe it as an independent gain.
