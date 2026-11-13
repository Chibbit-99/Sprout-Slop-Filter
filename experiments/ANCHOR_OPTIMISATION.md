# Same-size v4: baseline-anchored weight interpolation

The expanded task model gained vocabulary and training rows but lost accuracy on the 40 manually written unfamiliar-word probes (31/40 versus the historical 34/40). Instead of adding parameters, the default recipe now blends a **45% weight-space prior** from the older, more varied task-trained v4 checkpoint into the expanded checkpoint. Both classifiers share the *same* v4 feature map and task label policy. A convex combination of two linear logits is **exactly one linear classifier**: `w = 0.55 × expanded.w + 0.45 × baseline.w` and likewise for the bias. The result keeps **524,289 trainable scalars** and **3,402 observed alphabetic words** (3,424 tokenizer tokens). It is not a two-model ensemble at inference time; the optional archived checkpoint is used only while fitting/retraining. Older v1–v3 versions remain loadable.

## Selection and reproduction

`npm run test:anchor` compares six predetermined anchor fractions, using **only the frozen original 102-row validation partition to choose** accuracy first, then fewer false positives, then the smallest anchor on a tie. The original held-out test and challenge probes are reported *after* selection, not used to choose the fraction. The old validation data and probes had already been inspected in earlier project work, so this is an **illustrative developer diagnostic, not an independent estimate**.

| Anchor fraction | Original validation | False positives | False negatives |
| ---: | ---: | ---: | ---: |
| 0 (unanchored expansion) | 97/102 | 3 | 2 |
| 0.15 | 97/102 | 3 | 2 |
| 0.30 | 96/102 | 3 | 3 |
| **0.45 (chosen)** | **97/102** | **2** | **3** |
| 0.60 / 0.75 | 97/102 | 2 | 3 |

`npm run train` first fits `training.json` (786 base train + 2,900 training-only rows), softens scores ×4 and blends the result with `experiments/task_baseline_model.json`. The frozen unanchored checkpoint, used solely for comparison, is `experiments/task_unanchored_model.json`. `subsystems/blend_models.js` is the common Node/browser implementation; it rejects incompatible feature maps or label policies. Default browser retraining does the same post-fit blend if all training-only examples are present and the dataset is untouched. For imported/edited data or other encoders it does **not** apply the task prior; if the archived checkpoint cannot be fetched, it logs the reason and continues with the plain model. Training-curve charts show pre-blend epochs; final evaluated results use post-blend weights. All scoring is local.

## Tradeoffs, not deployment guarantees

| Diagnostic | Unanchored expansion | Anchored default | Historical 684-row baseline |
| --- | ---: | ---: | ---: |
| Expanded 170-case test | 168/170 (FP 1, FN 1) | **168/170 (FP 0, FN 2)** | Not comparable: 66 new rows not in its original test |
| Original locked 104-case test | 102/104 (FP 1, FN 1) | **102/104 (FP 0, FN 2)** | 101/104 (FP 0, FN 3) |
| 20 transfer cases | 20/20 | 20/20 | 20/20 |
| 40 unfamiliar cases | 31/40 (FP 9) | **35/40 (FP 5)** | 34/40 (FP 6) |
| Inflection / technical / names / multilingual / genericity (8 each) | 5 / 7 / 7 / 4 / 8 | **8 / 8 / 7 / 4 / 8** | 8 / 8 / 7 / 4 / 7 |

The total held-out accuracy **does not increase**; the change swaps one false positive for one false negative on the expanded test. “Rebuild libquartz.so for riscv64” and all four actionable non-English texts still receive false positives. This technique mitigates template overfitting but cannot teach semantics or verify authenticity. The shipped sparse JSON is **1,633,459 bytes** (51,358 nonzero weights; 524,289 potential parameters), versus 1,624,771 bytes unanchored. The default model is larger than the old 480,798-byte baseline because it saw many more training examples, *not* because of this optimisation. The browser smoke test (`npm run test:app`) loads and retrains it on the actual static server, while `npm run test:model` checks exact one-array reproduction and v1–v3 compatibility. Real-world accuracy needs independent human-labelled sources and a fresh locked test set.
