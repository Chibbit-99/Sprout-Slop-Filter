# Historical v3 parameter-budget audit

> This page records the **earlier unsupplemented v3 snapshot**, not the shipped model.
> The current default is v4 with training-only supplemental rows: see
> [ARCHITECTURE_V4.md](ARCHITECTURE_V4.md) and `npm run test:architecture`.


This was measured before the v4 upgrade. Reproduce the v3 capacity with `node train.mjs --encoder structured-subword-v3 --out /tmp/v3-task.json && node model_stats.mjs /tmp/v3-task.json`. The current browser and shipped files now report v4 counts.

## What is a parameter here?

A trainable parameter is a scalar coefficient updated by the optimiser. The v3 classifier has four **disjoint** feature banks of 32,768 real, reachable hash indices each: word unigrams, character 3–5-grams, adjacent ordered word pairs, and position/skip-pair context. The bank offset prevents two feature *types* from landing in the same slot. A weight within a bank may represent several colliding features. The logit is `sum(weights[index] * featureValue) + bias`, and all four banks can appear in normal text. SGD/AdaGrad updates each touched coefficient and the bias. Thus **4 × 32,768 + 1 = 131,073 trainable parameters**, including the bias. The v2 model had one shared 32,768-slot bank and a bias (32,769 trainable parameters). Versioned load/save preserves the correct feature extractor for v1/v2/v3 files.

Counting a potential coefficient does **not** mean it is trained or linguistically meaningful. In the shipped models, zero-initialized coefficients untouched by the small corpus remain zero and are omitted from JSON. `model_parameter_stats` distinguishes *capacity* (`trainable`) from *nonzero weights* (`nonZeroWeights`); the latter includes possible hash collisions and is only a proxy for the number of weights touched by examples. Vocabulary strings, optimizer accumulators, history, and metadata are not counted. For v1, random initial word-vector values make the nonzero count **not** a proxy for updates, so interpret it only as a numeric count. The 131k slots are reachable for different inputs, not artificially stored padding or 131k pretrained concepts.

| Model | Trainable | Nonzero weights after training | Word | Char | Adjacent | Position/skip | On disk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task v3 | 131,073 | 9,874 | 799 | 6,407 | 1,137 | 1,531 | 295,693 bytes |
| Commit v3 | 131,073 | 4,971 | 374 | 3,668 | 434 | 495 | 149,060 bytes |

Each group has 32,768 potential coefficient slots. The loader expands sparse JSON to the 131,072-element weight array, so numeric weight payload is ~1 MiB as double-precision values, plus runtime array overhead. Training has another ~1 MiB numeric accumulator payload (not shipped or counted as trainable). The model size and inference speed remain suitable for a normal consumer machine; timing depends on the machine and input length.

## Same-data comparison with v2

The archived comparison trained each architecture on the same seed-1 70/15/15 split (task: 478/102/104; commits: 139/29/31), 120 epochs, learning rate 0.8, temperature ×4, then compares held-out data and the *pre-existing* transfer/OOV and context challenge fixtures. It checks that v3 retraining reproduces the shipped model's weights and bias. Times below were from one run on the development sandbox, not consumer benchmarks. Rounded size is for the comparison's serialized model without CLI provenance metadata; shipped file sizes above include it.

| Domain | Encoder | Params (nonzero weights) | Held-out accuracy | TP / FP / TN / FN | Transfer | 40 unfamiliar tasks | 1,000 local scores |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| Tasks | v2 | 32,769 (8,137) | 97.1% | 48 / 0 / 53 / 3 | 20/20 | 33/40 | ~104 ms |
| Tasks | v3 | 131,073 (9,874) | 97.1% | 48 / 0 / 53 / 3 | 20/20 | 33/40 | ~100 ms |
| Commits | v2 | 32,769 (4,409) | 90.3% | 12 / 1 / 16 / 2 | 19/20 | n/a | ~90 ms |
| Commits | v3 | 131,073 (4,971) | 90.3% | 13 / 2 / 15 / 1 | 19/20 | n/a | ~80 ms |

The new capacity removes collisions between *types* of features and adds first/last/skip-pair context, but it **does not improve measured accuracy** on these tiny held-out splits. On commits it trades one fewer missed generic message for one more false positive. Reusing the shipped v3 task model zero-shot on hard helpdesk cases scores **9/16** (v2: 10/16), and on agendas **12/16** (v2: 12/16). Separate in-domain training performs better; see [CONTEXT_CHALLENGE.md](CONTEXT_CHALLENGE.md). Do not infer a guaranteed quality increase from a larger parameter count. Neither model is a language model or an authenticity detector. The fixtures were hand-written and inspected during development; new, independently labelled real-world data is needed for stronger claims.
