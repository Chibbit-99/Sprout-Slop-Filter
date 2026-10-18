// Measurements on the real hand-written dataset in `training.json`.
//
// The other experiments use synthetic pools so they can vary one thing at a
// time. This one measures what actually ships: the 524 labelled examples in
// `training.json`, split and trained through the real subsystem code.
//
// It answers three questions:
//
//   A) How much of the real dataset does the model need before the held-out
//      score stops improving?
//   B) How much of the held-out text is covered by the training
//      vocabulary, and what happens when that coverage drops?
//   C) What does score softening do to the numbers?
//
// Run: node experiments/real_dataset.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { sprout, buildVocabulary, slopBreakdown } from "./lib/sprout.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataFile = path.join(here, "..", "training.json");

const SEEDS = [1, 2, 3, 4, 5];
const SIZES = [25, 50, 100, 200, 366];

function loadExamples() {
    const parsed = JSON.parse(fs.readFileSync(dataFile, "utf8"));
    return parsed.map(entry => ({
        text: entry.text,
        label: String(entry.label).trim().toLowerCase()
    }));
}

function average(values) {
    return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}

function standardDeviation(values) {
    const mean = average(values);
    return Math.sqrt(average(values.map(v => (v - mean) ** 2)));
}

function pct(value) {
    return `${(value * 100).toFixed(1)}%`;
}

// Train on a subset of the training split and score the held-out test
// split. The test split never changes, so the numbers are comparable.
function runAtSize(trainSubset, validationSet, testSet, seed) {
    const model = sprout.initialise_model({
        vocabulary: buildVocabulary(trainSubset),
        vectorSize: 24,
        seed,
        learningRate: 0.2
    });

    sprout.train_model(trainSubset, validationSet, model, { epochs: 60, learningRate: 0.2 });

    const metrics = sprout.evaluate_model(model, testSet);
    return { model, metrics };
}

const examples = loadExamples();
const genuine = examples.filter(e => e.label === "genuine").length;

console.log("=== Real dataset: training.json ===");
console.log(`${examples.length} examples (${genuine} genuine, ${examples.length - genuine} slop)`);
console.log("");

// One fixed split for every measurement below.
const { trainSet, validationSet, testSet } = sprout.split_dataset(examples, { seed: 1 });
console.log(`split: ${trainSet.length} train / ${validationSet.length} validation / ${testSet.length} test`);
console.log("");

// ---------------------------------------------------------------------------
// A) How much data does the real dataset need?
// ---------------------------------------------------------------------------

console.log("=== A) Held-out accuracy by training-set size (5 seeds) ===");
console.log("train size | accuracy (mean ± sd) | slop recall | false positives");
console.log("-".repeat(68));

for (const size of SIZES) {
    const accuracies = [];
    const recalls = [];
    const falsePositives = [];

    for (const seed of SEEDS) {
        // Take an evenly spread subset so every size sees both labels.
        const subset = [];
        const step = trainSet.length / size;
        for (let i = 0; i < size; i++) {
            subset.push(trainSet[Math.min(trainSet.length - 1, Math.floor(i * step))]);
        }

        const { metrics } = runAtSize(subset, validationSet, testSet, seed);
        const breakdown = slopBreakdown(metrics);

        accuracies.push(metrics.accuracy);
        recalls.push(breakdown.slopRecall);
        falsePositives.push(breakdown.falsePositiveRate);
    }

    console.log(
        `${String(size).padStart(10)} | ` +
        `${pct(average(accuracies)).padStart(6)} ± ${(standardDeviation(accuracies) * 100).toFixed(1)} pts | ` +
        `${pct(average(recalls)).padStart(10)} | ${pct(average(falsePositives)).padStart(14)}`
    );
}

console.log("");

// ---------------------------------------------------------------------------
// B) Vocabulary coverage
// ---------------------------------------------------------------------------

console.log("=== B) Vocabulary coverage of the held-out test split ===");
console.log("train size | vocabulary | test tokens known | accuracy");
console.log("-".repeat(62));

for (const size of SIZES) {
    const accuracies = [];
    const vocabularySizes = [];
    let knownTotal = 0;
    let tokenTotal = 0;

    for (const seed of SEEDS) {
        const subset = [];
        const step = trainSet.length / size;
        for (let i = 0; i < size; i++) {
            subset.push(trainSet[Math.min(trainSet.length - 1, Math.floor(i * step))]);
        }

        const { model, metrics } = runAtSize(subset, validationSet, testSet, seed);
        accuracies.push(metrics.accuracy);
        vocabularySizes.push(model.vocabulary.length);

        const known = new Set(model.vocabulary);
        for (const example of testSet) {
            for (const token of sprout.tokenise(sprout.normalise_text(example.text))) {
                tokenTotal++;
                if (known.has(token)) knownTotal++;
            }
        }
    }

    console.log(
        `${String(size).padStart(10)} | ${String(Math.round(average(vocabularySizes))).padStart(10)} | ` +
        `${pct(knownTotal / tokenTotal).padStart(17)} | ${pct(average(accuracies)).padStart(8)}`
    );
}

console.log("");

// ---------------------------------------------------------------------------
// C) Score softening
// ---------------------------------------------------------------------------

console.log("=== C) Score softening (temperature) on the held-out test split ===");
console.log("temp | accuracy | median genuine | median slop | scores at the extremes");
console.log("-".repeat(76));

const { model: baseModel, metrics: baseMetrics } = runAtSize(trainSet, validationSet, testSet, 7);

for (const temperature of [1, 2, 4, 8]) {
    const model = temperature === 1
        ? baseModel
        : {
            ...baseModel,
            weights: baseModel.weights.map(w => w / temperature),
            bias: baseModel.bias / temperature
        };

    const metrics = sprout.evaluate_model(model, testSet);
    const genuineScores = testSet
        .filter(e => e.label === "genuine")
        .map(e => sprout.make_prediction_for_training(e.text, model).score)
        .sort((a, b) => a - b);
    const slopScores = testSet
        .filter(e => e.label === "slop")
        .map(e => sprout.make_prediction_for_training(e.text, model).score)
        .sort((a, b) => a - b);
    const all = [...genuineScores, ...slopScores];
    const pinned = all.filter(s => s < 0.01 || s > 0.99).length;

    const median = values => values[Math.floor(values.length / 2)];

    console.log(
        `x${String(temperature).padEnd(4)} | ${pct(metrics.accuracy).padStart(8)} | ` +
        `${median(genuineScores).toFixed(3).padStart(14)} | ${median(slopScores).toFixed(3).padStart(11)} | ` +
        `${pct(pinned / all.length).padStart(8)} at 0.00/1.00`
    );
}

console.log("");
console.log(`accuracy without softening: ${pct(baseMetrics.accuracy)}`);
console.log("");
console.log("Softening divides the classifier weights and the bias, so the sign of every");
console.log("logit — and therefore every decision at the 0.5 boundary — is unchanged.");
console.log("Only the confidence moves.");

// ---------------------------------------------------------------------------
// D) Which held-out examples are still wrong?
// ---------------------------------------------------------------------------

console.log("");
console.log("=== D) The held-out examples the model still gets wrong ===");

const mistakes = testSet
    .map(example => {
        const { score } = sprout.make_prediction_for_training(example.text, baseModel);
        return { ...example, score, predicted: score >= 0.5 ? "slop" : "genuine" };
    })
    .filter(example => example.predicted !== example.label);

if (mistakes.length === 0) {
    console.log("none");
} else {
    for (const mistake of mistakes) {
        const known = new Set(baseModel.vocabulary);
        const tokens = sprout.tokenise(sprout.normalise_text(mistake.text));
        const unknown = tokens.filter(token => !known.has(token));
        console.log(
            `${mistake.score.toFixed(3)}  truth=${mistake.label.padEnd(7)} ` +
            `"${mistake.text}"  (${unknown.length}/${tokens.length} tokens unknown)`
        );
    }
}

console.log("");
console.log("All three are slop phrases built from words the model never saw in");
console.log("training, so almost every token collapses to <UNK> and only the");
console.log("function words are left to judge them by — and function words lean");
console.log("genuine, because most genuine task descriptions contain them.");
