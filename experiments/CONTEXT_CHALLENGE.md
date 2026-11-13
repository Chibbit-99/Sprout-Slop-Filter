# Unfamiliar vocabulary and non-commit contexts

This is a **historical baseline snapshot**, using `task_baseline_training.json` and `task_baseline_model.json`; current default task results are in [TRAINING_EXPANSION.md](TRAINING_EXPANSION.md). Run `npm run test:contexts`. This checks annotated unseen tokens with the real tokenizer and training vocabularies, excludes duplicate sentences from training, scores the shipped models and trains transient helpdesk and agenda models with the same subsystem code. **These manually authored, developer-inspected probes are not an independent or production benchmark.** Threshold is 0.5; no challenge row is trained on. Both label policies are about *informativeness*, not truth, authorship or ethical quality.

## Unfamiliar task vocabulary

`unfamiliar_cases.json` has five balanced groups of eight. Every row specifies at least one token absent from the full original task corpus **and the 70 selected supplemental rows**. With the archived v4 task model, **113/230 tokens were unseen** (the designated targets all remain unseen). For comparison, v1 is trained on the same base training split plus the 70 supplement rows. The historical v3 task model without supplements scored 33/40 here; v3 **with** the same 70 rows scores 34/40 (`npm run test:architecture`).

| Challenge | v4 correct | legacy v1 correct | v4 false positives | v4 unseen tokens |
| --- | ---: | ---: | ---: | ---: |
| Inflections / derivations | 8/8 | 7/8 | 0/4 | 22/40 |
| Scientific / technical | 8/8 | 8/8 | 0/4 | 19/48 |
| Names / identifiers | 7/8 | 6/8 | 1/4 | 20/52 |
| Multilingual / code switching | **4/8** | 4/8 | **4/4** | 28/29 |
| Genericity with technical nouns | 7/8 | 7/8 | 1/4 | 24/61 |
| **Total** | **34/40 (85%)** | **32/40 (80%)** | **6/20** | **113/230** |

Specific failures: it wrongly flags “Rebuild libquartz.so for riscv64” and “Print 200 barcode labels for shipment RX-43”; it also flags **all four actionable non-English examples** (French, Spanish, Japanese and Chinese). Spelling-based features are not translation or language understanding. Do not use this English-only model to block multilingual work.

## Other situations

`other_contexts.json` defines **helpdesk** reports (a report with a reproducible symptom/error/device) and **meeting agendas** (a specific decision, document or subject), not Git commits. Each has 60 training examples (30 per label), 24 ordinary test rows and 16 harder test rows with matched unfamiliar identifiers/nouns on both sides. Every test row has a designated token absent from its *own domain's* training vocabulary. The v4 in-domain models from these training rows are now also shipped as separate experimental `helpdesk_model.json` and `agenda_model.json` CLI models (not browser presets). They still have no independent held-out accuracy measurement; see [new, leakage-checked domain diagnostics](DOMAIN_MODELS.md) for further probes and false positives. The archived task and shipped commit models are scored zero-shot (no retraining on those policies).

| Use case / model | Ordinary (24) | Hard (16) | Hard false positives | Hard unseen tokens |
| --- | ---: | ---: | ---: | ---: |
| Helpdesk: task v4 zero-shot | 18/24 | 9/16 | 6/8 | 87/154 |
| Helpdesk: commit v4 zero-shot | 21/24 | 12/16 | 1/8 | 97/154 |
| Helpdesk: **in-domain v4** | **24/24** | **14/16** | **1/8** | **87/154** |
| Helpdesk: in-domain v3 | 24/24 | 14/16 | 1/8 | 87/154 |
| Helpdesk: in-domain v2 | 24/24 | 14/16 | 1/8 | 87/154 |
| Helpdesk: in-domain legacy v1 | 24/24 | 12/16 | 1/8 | 87/154 |
| Agendas: task v4 zero-shot | 21/24 | 13/16 | 3/8 | 54/132 |
| Agendas: commit v4 zero-shot | 18/24 | 13/16 | 2/8 | 83/132 |
| Agendas: **in-domain v4** | **24/24** | **15/16** | **1/8** | **69/132** |
| Agendas: in-domain v3 | 24/24 | 16/16 | 0/8 | 69/132 |
| Agendas: in-domain v2 | 24/24 | 15/16 | 1/8 | 69/132 |
| Agendas: in-domain legacy v1 | 23/24 | 15/16 | 0/8 | 69/132 |

V4 helpdesk training still flags a concrete drone telemetry report and accepts a generic CX-88 ticket. V4 agenda training flags a specific question about contacting a spectrometer vendor. For these cases, the larger v4 does **not** consistently beat v3. The shipped in-domain v4 JSON files are about 165 KB and 139 KB. Handwritten style overlap makes ordinary examples easier than truly external user text. Train a separate model on real labels for each use case, collect a locked external test set, and do not equate an informative-sounding sentence with a truthful report or an event that occurred.
