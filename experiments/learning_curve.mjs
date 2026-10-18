// Learning-curve experiment for Sprout Slop Filter.
//
// Question: roughly how many labelled examples does Sprout need before its
// slop scores become trustworthy?
//
// Method: train the real subsystem code at increasing dataset sizes and
// measure held-out accuracy, under two conditions:
//
//   AS-IS   — the real subsystem trainer in subsystems/. Since the
//             word-vector fix (FINDINGS.md section 1) this trains the word
//             vectors alongside the classifier.
//   PATCHED — the independent reference trainer (lib/trainer_v2.mjs), kept
//             as a cross-check that the subsystem fix reproduces it. The
//             two curves should now sit on top of each other.
//
// Two test regimes:
//   SEEN WORDS   — new combinations of words that appeared in training
//   UNSEEN WORDS — genuine/slop text built only from held-out word pools
//
// Run: node experiments/learning_curve.mjs

import { sprout } from "./lib/sprout.mjs";
import {
    VERBS, OBJECTS, VAGUE,
    HELD_OUT_VERBS, HELD_OUT_OBJECTS, HELD_OUT_VAGUE,
    makeRandom, makeExamples, balanced, buildVocabulary,
    metricsFor, average, standardDeviation, pct
} from "./lib/data.mjs";
import { trainWithWordVectors } from "./lib/trainer_v2.mjs";

const TRAIN_SIZES = [8, 16, 32, 50, 100, 200, 400, 800, 1600, 3200];
const SEEDS = [1, 2, 3, 4, 5];
const EPOCHS = 200;
const LEARNING_RATE = 0.2;

const buildTestSets = () => ({
    seen: balanced(300, makeRandom(999), { verbs: VERBS, objects: OBJECTS, vague: VAGUE }),
    unseen: balanced(300, makeRandom(1000), {
        verbs: HELD_OUT_VERBS, objects: HELD_OUT_OBJECTS, vague: HELD_OUT_VAGUE
    })
});

// ---------------------------------------------------------------------------
// Experiment 1: is the documented word-vector update actually happening?
// ---------------------------------------------------------------------------

function vectorAudit() {
    const trainSet = makeExamples(200, makeRandom(11), { verbs: VERBS, objects: OBJECTS, vague: VAGUE });

    const asIs = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed: 3 });
    const before = JSON.stringify(asIs.wordVectors);
    sprout.train_model(trainSet, [], asIs, { epochs: 50, learningRate: 0.1 });
    const asIsChanged = JSON.stringify(asIs.wordVectors) !== before;

    const patched = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed: 3 });
    trainWithWordVectors(trainSet, [], patched, sprout, { epochs: 50, learningRate: 0.1, seed: 3 });
    const patchedChanged = JSON.stringify(patched.wordVectors) !== before;

    return {
        vocabularySize: asIs.vocabulary.length,
        classifierParameters: asIs.weights.length + 1,
        asIsWordVectorValues: asIs.wordVectors.length * asIs.weights.length,
        asIsChanged,
        patchedChanged,
        patchedWordVectorValues: patched.wordVectors.length * patched.weights.length
    };
}

// ---------------------------------------------------------------------------
// Experiment 2: learning curves
// ---------------------------------------------------------------------------

function runCurve(mode) {
    const tests = buildTestSets();
    const rows = [];

    for (const size of TRAIN_SIZES) {
        const acc = { seen: [], unseen: [], train: [], recallSlop: [], recallGenuine: [], vocab: [], fpr: [] };

        for (const seed of SEEDS) {
            const random = makeRandom(seed * 7919 + size);
            const trainSet = makeExamples(size, random, { verbs: VERBS, objects: OBJECTS, vague: VAGUE });
            const model = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed });

            if (mode === "patched") {
                trainWithWordVectors(trainSet, [], model, sprout, {
                    epochs: EPOCHS, learningRate: LEARNING_RATE, seed
                });
            } else {
                sprout.train_model(trainSet, [], model, { epochs: EPOCHS, learningRate: LEARNING_RATE });
            }

            const seenMetrics = metricsFor(model, tests.seen);
            const unseenMetrics = metricsFor(model, tests.unseen);
            const trainMetrics = metricsFor(model, trainSet);

            acc.seen.push(seenMetrics.accuracy);
            acc.unseen.push(unseenMetrics.accuracy);
            acc.train.push(trainMetrics.accuracy);
            acc.recallSlop.push(seenMetrics.slopRecall);
            acc.recallGenuine.push(1 - seenMetrics.falsePositiveRate);
            acc.fpr.push(seenMetrics.falsePositiveRate);
            acc.vocab.push(model.vocabulary.length);
        }

        rows.push({
            size,
            vocabulary: Math.round(average(acc.vocab)),
            train: average(acc.train),
            seen: average(acc.seen),
            seenSd: standardDeviation(acc.seen),
            unseen: average(acc.unseen),
            slopRecall: average(acc.recallSlop),
            genuineRecall: average(acc.recallGenuine),
            fpr: average(acc.fpr)
        });
    }

    return rows;
}

function printTable(title, rows) {
    console.log(`=== ${title} ===`);
    console.log("train | vocab | trainAcc | SEEN acc (sd) | slopRecall | genuineRecall | falsePos | UNSEEN acc");
    console.log("-".repeat(104));
    for (const row of rows) {
        console.log(
            [
                String(row.size).padStart(5),
                String(row.vocabulary).padStart(5),
                pct(row.train).padStart(8),
                `${pct(row.seen)} (${(row.seenSd * 100).toFixed(1)})`.padStart(15),
                pct(row.slopRecall).padStart(10),
                pct(row.genuineRecall).padStart(13),
                pct(row.fpr).padStart(8),
                pct(row.unseen).padStart(10)
            ].join(" | ")
        );
    }
    console.log("");
}

// ---------------------------------------------------------------------------

console.log("=== Is the model actually learning word vectors? ===");
const audit = vectorAudit();
console.log(`vocabulary (200 training examples):   ${audit.vocabularySize} words`);
console.log(`word vector width:                    ${audit.classifierParameters - 1} dimensions`);
console.log(`trainable classifier parameters:      ${audit.classifierParameters} (weights + bias)`);
console.log(`word vectors changed during training:`);
console.log(`   AS-IS (subsystems/):               ${audit.asIsChanged ? "yes" : "NO — frozen at initial random values"}`);
console.log(`   PATCHED (trainer_v2.mjs):          ${audit.patchedChanged ? `yes — ${audit.patchedWordVectorValues} values now trained` : "no"}`);
console.log("");

printTable(`AS-IS learning curve (${SEEDS.length} seeds, ${EPOCHS} epochs)`, runCurve("as-is"));
printTable(`PATCHED learning curve (${SEEDS.length} seeds, ${EPOCHS} epochs, lr ${LEARNING_RATE})`, runCurve("patched"));
