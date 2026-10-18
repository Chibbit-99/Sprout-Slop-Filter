// Train the Sprout slop model from `training.json` and write a ready-to-use
// model file (`model.json` by default).
//
// The training code is the real subsystem code in `subsystems/`, loaded the
// same way `test.html` and the experiments load it, so the model produced
// here behaves exactly like the one the browser app trains in the page.
//
// Usage:
//   node train.mjs
//   node train.mjs --epochs 400 --lr 0.2 --vector-size 24 --seed 7
//   node train.mjs --data training.json --out model.json
//
// Options:
//   --data <file>       training data (JSON array of { text, label })
//   --out <file>        where to write the trained model
//   --epochs <n>        training rounds (default 60)
//   --lr <float>        learning rate (default 0.2)
//   --vector-size <n>   dimensions per word vector (default 24)
//   --seed <n>          PRNG seed, for reproducibility (default 7)
//   --split-seed <n>    seed for the train/validation/test split (default 1)
//   --temperature <n>   score softening, 1 = off (default 4)
//   --quiet             only print the final summary

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { sprout, buildVocabulary, slopBreakdown } from "./experiments/lib/sprout.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));

// ----- Command line --------------------------------------------------------

function parseArgs(argv) {
    const options = {
        data: "training.json",
        out: "model.json",
        epochs: 60,
        learningRate: 0.2,
        vectorSize: 24,
        seed: 7,
        splitSeed: 1,
        temperature: 4,
        quiet: false
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];

        switch (arg) {
            case "--data":       options.data = argv[++i]; break;
            case "--out":        options.out = argv[++i]; break;
            case "--epochs":     options.epochs = Number(argv[++i]); break;
            case "--lr":         options.learningRate = Number(argv[++i]); break;
            case "--vector-size": options.vectorSize = Number(argv[++i]); break;
            case "--seed":       options.seed = Number(argv[++i]); break;
            case "--split-seed": options.splitSeed = Number(argv[++i]); break;
            case "--temperature": options.temperature = Number(argv[++i]); break;
            case "--quiet":      options.quiet = true; break;
            case "--help":
                console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8")
                    .split("\n").slice(2, 24).join("\n"));
                process.exit(0);
                break;
            default:
                console.error(`Unknown option: ${arg}`);
                process.exit(1);
        }
    }

    return options;
}

// ----- Data ----------------------------------------------------------------

// Mirrors `load_training_data` in subsystems/: the same file shape, the same
// validation, the same "fail loudly on a bad label" behaviour. The browser
// uses the subsystem version; this script reads from disk directly because
// `fetch` cannot open a `file://` path in Node.
function loadExamples(file) {
    const fullPath = path.resolve(here, file);
    let parsed;

    try {
        parsed = JSON.parse(fs.readFileSync(fullPath, "utf8"));
    } catch (error) {
        throw new Error(`Could not read ${file}: ${error.message}`);
    }

    if (!Array.isArray(parsed)) {
        throw new TypeError(`${file}: expected an array of examples`);
    }

    return parsed.map((entry, index) => {
        if (!entry || typeof entry.text !== "string" || entry.text.length === 0) {
            throw new Error(`${file}: example ${index} is missing valid text`);
        }

        const label = typeof entry.label === "string" ? entry.label.trim().toLowerCase() : "";

        if (label !== "genuine" && label !== "slop") {
            throw new Error(
                `${file}: example ${index} has invalid label "${entry.label}" ` +
                `(expected "genuine" or "slop")`
            );
        }

        return { text: entry.text, label };
    });
}

// ----- Training ------------------------------------------------------------

const options = parseArgs(process.argv.slice(2));
const log = options.quiet ? () => {} : (...args) => console.log(...args);

const examples = loadExamples(options.data);
const genuineCount = examples.filter(e => e.label === "genuine").length;
const slopCount = examples.length - genuineCount;

log("=== Sprout Slop Filter — training ===");
log(`data      : ${options.data} (${examples.length} examples: ${genuineCount} genuine, ${slopCount} slop)`);
log(`settings  : epochs=${options.epochs} lr=${options.learningRate} vectorSize=${options.vectorSize} seed=${options.seed}`);
log("");

const { trainSet, validationSet, testSet } = sprout.split_dataset(examples, {
    seed: options.splitSeed
});

// The vocabulary is built from the training split only, so the held-out
// test split stays a fair measure of how the model generalises.
const vocabulary = buildVocabulary(trainSet);
const model = sprout.initialise_model({
    vocabulary,
    vectorSize: options.vectorSize,
    seed: options.seed,
    learningRate: options.learningRate,
    epochs: options.epochs
});

log(`split     : ${trainSet.length} train / ${validationSet.length} validation / ${testSet.length} test`);
log(`vocabulary: ${model.vocabulary.length} entries (including <UNK>)`);
log("");
log("training...");

const started = Date.now();
sprout.train_model(trainSet, validationSet, model, {
    epochs: options.epochs,
    learningRate: options.learningRate
});
const elapsed = ((Date.now() - started) / 1000).toFixed(1);

// Score softening.
//
// The training data is cleanly separable, so the logistic regression grows
// very confident and every score ends up pinned at 0.000 or 1.000. Dividing
// the classifier weights and the bias by a temperature spreads the scores
// out again. It cannot change a single decision: the sign of the logit is
// untouched, so the 0.5 boundary and every metric below are exactly the
// same — only the confidence changes.
if (options.temperature > 0 && options.temperature !== 1) {
    model.weights = model.weights.map(weight => weight / options.temperature);
    model.bias = model.bias / options.temperature;
}

const trainMetrics = sprout.evaluate_model(model, trainSet);
const validationMetrics = sprout.evaluate_model(model, validationSet);
const testMetrics = sprout.evaluate_model(model, testSet);

log(`done in ${elapsed}s`);
log("");
log("--- results -------------------------------------------------");
log(`train      accuracy ${pct(trainMetrics.accuracy)}  loss ${trainMetrics.averageLoss.toFixed(4)}`);
log(`validation accuracy ${pct(validationMetrics.accuracy)}  loss ${validationMetrics.averageLoss.toFixed(4)}`);
log(`test       accuracy ${pct(testMetrics.accuracy)}  loss ${testMetrics.averageLoss.toFixed(4)}`);
log(`test       slop recall ${pct(slopBreakdown(testMetrics).slopRecall)}  ` +
    `false positives ${pct(slopBreakdown(testMetrics).falsePositiveRate)}`);
log(`test       TP ${testMetrics.truePositive}  FP ${testMetrics.falsePositive}  ` +
    `TN ${testMetrics.trueNegative}  FN ${testMetrics.falseNegative}`);
log(`softening  x${options.temperature} (scores spread out; decisions unchanged)`);
log("-------------------------------------------------------------");

// ----- Save ----------------------------------------------------------------

// `save_model` is the subsystem's own serialiser. In Node it needs a
// `storage` hook to write the payload somewhere.
//
// The metadata below rides along inside `options`, so a browser app that
// loads the file can report where the model came from and how well it
// scored, without having to retrain it to find out.
model.options = {
    ...model.options,
    temperature: options.temperature,
    trainedFrom: {
        data: options.data,
        examples: examples.length,
        trainSet: trainSet.length,
        validationSet: validationSet.length,
        testSet: testSet.length,
        vectorSize: options.vectorSize,
        seed: options.seed,
        splitSeed: options.splitSeed,
        epochs: options.epochs,
        learningRate: options.learningRate,
        testAccuracy: testMetrics.accuracy,
        testSlopRecall: slopBreakdown(testMetrics).slopRecall,
        testFalsePositiveRate: slopBreakdown(testMetrics).falsePositiveRate,
        trainedAt: new Date().toISOString()
    }
};

const outPath = path.resolve(here, options.out);
const saveResult = sprout.save_model(model, options.out, {
    storage: payload => fs.writeFileSync(outPath, payload)
});

if (!saveResult.success) {
    console.error(`Failed to save the model: ${saveResult.error}`);
    process.exit(1);
}

log(`saved     : ${options.out} (${(fs.statSync(outPath).size / 1024).toFixed(1)} KB)`);

function pct(value) {
    return `${(value * 100).toFixed(1)}%`;
}
