# Helpdesk and meeting-agenda models (experimental)

Reproduce with `npm run train:contexts` and `npm run test:domains`. `node score.mjs --domain helpdesk|agendas` selects separate v4 models, rather than applying task or commit labels to a different policy. The browser page has no situation modes and loads only the task-trained model by default. These are **small English-oriented text-only demonstrations**, not truth or authorship detectors.

| Use case | `genuine` under this policy | `slop` under this policy |
| --- | --- | --- |
| Helpdesk | An observable symptom, error, affected device or reproducible condition | A request lacking a diagnostic clue |
| Meeting agendas | A concrete decision, document, question or subject for the meeting | A vague placeholder or generic catch-up |

The 60 examples (30 per label) in each `*_training.json` are **all used to train** its shipped model; there is no held-out accuracy field on either artifact. The models each have 524,289 trainable parameters; sparse JSON is 164,816 bytes (helpdesk, 5,402 nonzero weights) and 139,146 bytes (agendas, 4,555 nonzero weights). Each is trained deterministically for 120 epochs at learning rate 0.8, seed 7, then score-softened ×4. The recipe imports training files only; ordinary and hard rows are in `other_contexts.json`, and 24 newly authored OOV-annotated probes per use case are in `domain_probes.json`. `domain_benchmark.mjs` asserts no normalized probe duplicates in its own training/ordinary/hard sets, checks the annotated out-of-training-vocabulary tokens, checks the training file against the frozen fixtures, and confirms model metadata and trainable parameter counts. No evaluation rows are added to training. Threshold is 0.5.

The comparison below counts correct cases, **not independent accuracy estimates**. Existing ordinary/hard sets and new probes are manually written and inspected during development; their style and label policy could favour this model. All samples are deliberately small. Expanded task, historical task and commit models are scored without retraining on these policies, so their zero-shot numbers are not measures of how well they serve a real helpdesk or meeting. For a new setting, gather independently labelled real text, split by author/source and time, lock a fresh test set, tune thresholds using validation only, and measure false-positive cost.

| Situation / model | Helpdesk ordinary /24 | Helpdesk hard /16 | Helpdesk new /24 | Agendas ordinary /24 | Agendas hard /16 | Agendas new /24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expanded task v4 (zero-shot) | 17 | 9 | 17 | 23 | 13 | 22 |
| Historical task v4 (zero-shot) | 18 | 9 | 17 | 21 | 13 | 21 |
| Commit v4 (zero-shot) | 21 | 12 | 19 | 18 | 13 | 20 |
| Domain v4 (60-row trained) | **24** | **14** | **24** | **24** | **15** | **23** |

Mistakes matter more than the high small-set totals: on hard helpdesk reports the in-domain model falsely flags one specific report (a drone telemetry symptom) and accepts one generic CX-88 ticket. On hard agendas it falsely flags a concrete question about a spectrometer vendor. On the new agenda probes it also **falsely flags** “Determine if the Solaris rollout can proceed without SSO” (score 0.539). Domain-model false positives on the new helpdesk set are 0/12; on the new agenda set 1/12. The v3 agenda model scored 16/16 on the hard set versus v4's 15/16, so additional capacity is not consistently an improvement. Out-of-vocabulary technical names and multilingual text can still fail; subword features do not provide semantic comprehension.

The page is served over HTTP (`npm start`); `npm run test:app` uses the real static server to verify its single default model and dataset, scoring, retraining, evaluation and local save/reload. It has no situation selector. To try another model in the browser, you may explicitly import a dataset and load a model file; be sure their label policy matches. Browser retraining on any imported examples is a *new experiment on a subset*, not an evaluation of an already shipped 60-row domain model. A text-only classifier cannot confirm a report is true, an agenda was discussed, or a Git diff matches its commit subject.
