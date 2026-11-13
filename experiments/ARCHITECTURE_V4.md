# V4: 524,289 trainable parameters and training-only vocabulary growth

**Historical baseline snapshot (before the default task corpus was expanded).** Task figures below refer to `experiments/task_baseline_training.json` and `experiments/task_baseline_model.json`, not the current default page. See [the expansion report](TRAINING_EXPANSION.md) for current counts and caveats.

Reproduce: `npm run stats -- experiments/task_baseline_model.json`, `npm run stats -- commit_model.json`, `npm run test:architecture`, `npm run test:vocab`, `npm run test:contexts`. The same `model_parameter_stats` function powers CLI, browser and tests. This document records **developer-authored diagnostics**, not an externally validated accuracy claim. The earlier 131k v3 snapshot is [ARCHITECTURE.md](ARCHITECTURE.md).

## Capacity versus learned weights

V4 uses eight *reachable*, disjoint 65,536-slot hashed feature banks (word identity; separate character 3-, 4-, and 5-grams; adjacent and skip pairs; first/last-word position; and prefixes/suffixes). Thus **524,288 learnable coefficients + 1 bias = 524,289 trainable parameters**. Feature families cannot collide with one another, but arbitrary features **can** collide within a bank. The weights start at zero and update only on training examples. The number of nonzero weights is much smaller than the capacity, and it is *not* a count of learned vocabulary words or semantic concepts. The inactive slots can be updated by new input/training; they are not dummy padding.

| Historical v4 recipe | Capacity | Nonzero weights after training | Disk bytes | Observed train tokens |
| --- | ---: | ---: | ---: | ---: |
| Task (478 base train + 70 extra) | 524,289 | 15,719 | 480,798 | 930 |
| Commit (139 base train + 80 extra) | 524,289 | 9,797 | 299,609 | 545 |

By family, task nonzero counts are: word 922, char3 1,833, char4 2,982, char5 3,007, adjacent 1,432, skip 1,251, position 637, affix 3,655. Commit nonzero counts: 544 / 1,378 / 1,936 / 1,898 / 793 / 626 / 266 / 2,356. Every family has 65,536 potential coefficients. `npm run stats` recomputes the counts from the **loaded** file. Sparse JSON omits zero weights; inference expands them to 524,288 dense values (~4 MiB numeric double payload plus JS overhead). A training accumulator uses another ~4 MiB, but **is not a trainable parameter**. Input is capped to 128 tokens and 64 characters per token. Inference needs only plain JavaScript, no web/API lookup.

## Is a larger model better?

`npm run test:architecture` retrains **v3 and v4 on the same rows**: the original seed-1 base split plus the selected supplement prefix. Hyperparameters are held constant (120 epochs, learning rate 0.8, temperature ×4). The script asserts exact v4 weight/bias reproduction of the archived task and current commit files, plus roundtrip scoring. Times below are approximate sandbox measurements and include feature extraction. Challenge phrases are excluded from training.

| Domain / training rows | Model | Params (nonzero) | Original validation | Original held-out test | Transfer | 40 unfamiliar tasks | 1,000 scores |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Task: 478 + 70 | v3 | 131,073 (11,229) | 95.1% | 97.1%, TP 48 / FP 0 / TN 53 / FN 3 | 20/20 | 34/40 | ~95 ms |
| Task: 478 + 70 | **v4** | **524,289 (15,719)** | **95.1%** | **97.1%, TP 48 / FP 0 / TN 53 / FN 3** | **20/20** | **34/40** | ~150 ms |
| Commit: 139 + 80 | v3 | 131,073 (7,092) | 93.1% | 90.3%, TP 12 / FP 1 / TN 16 / FN 2 | 20/20 | n/a | ~79 ms |
| Commit: 139 + 80 | **v4** | **524,289 (9,797)** | **93.1%** | **93.5%, TP 13 / FP 1 / TN 16 / FN 1** | **20/20** | n/a | ~140 ms |

The task result is unchanged; on commits v4 correctly identifies **one** more generic message without an additional false positive. It costs more time and disk space. No general accuracy guarantee follows from the parameter count.

## Does adding labelled vocabulary help?

`npm run test:vocab` freezes the **original** train/validation/test split before adding any supplemental examples, and enforces no exact duplicates with any base example (including validation/test), no duplicates within the supplement, and no annotated task challenge token in the supplement. The supplemental JSON is ordered genuine/slop in pairs. The script tests v3 and v4 with 0, partial and all extra rows, without changing the test set. The specific supplement sizes were explored during development, so even the original test split has been inspected; **do not treat this as an independent performance estimate**.

| Domain | Encoder / extra train rows | Observed train tokens | Validation | Original test | Test OOV tokens | Unfamiliar task challenge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tasks | v4 +0 | 813 | 92.2% (4 FP, 4 FN) | 97.1% (0 FP, 3 FN) | 123/429 | 33/40 |
| Tasks | **v4 +70** | **930** | **95.1% (2 FP, 3 FN)** | **97.1% (0 FP, 3 FN)** | **114/429** | **34/40** |
| Tasks | v4 +140 | 1,071 | 93.1% (3 FP, 4 FN) | 97.1% (0 FP, 3 FN) | 96/429 | 34/40 |
| Commits | v4 +0 | 376 | 89.7% (2 FP, 1 FN) | 90.3% (2 FP, 1 FN) | 66/137 | n/a |
| Commits | v4 +40 | 457 | 86.2% (3 FP, 1 FN) | 90.3% (2 FP, 1 FN) | 55/137 | n/a |
| Commits | **v4 +80** | **545** | **93.1% (2 FP, 0 FN)** | **93.5% (1 FP, 1 FN)** | **48/137** | n/a |

The selected task recipe (+70) improves *validation* and one unfamiliar probe, but does not improve the fixed test score. More words (+140) reduce OOV rate further yet validation worsens. The commit recipe (+80) improves the fixed split by one example; 31 examples are far too few for a reliable claim. Vocabulary is only an observed-token list for v4: adding word strings without labelled training would not update weights or convey meaning. The supplement adds **labelled, varied sentences**, not a bare dictionary. The held-out originals and unfamiliar probes remain excluded from training. See `test:contexts` for failures on new domains and multilingual input.

### Selection and limitations

The base split, supplemental rows, challenge cases and model configurations were hand-written or inspected during development. The apparent improvement may be due to style overlap. More data and more capacity can both *hurt* performance; for instance, on a separate 16-case hard meeting-agenda slice, domain-trained v4 scores 15/16 versus v3's 16/16. The archived task model zero-shot scores only 9/16 on hard helpdesk reports. Report-specific/authenticity claims cannot be inferred from text alone. Collect independently labelled text from multiple sources and use a new locked external test set before deployment.
