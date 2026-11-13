// Train the Sprout slop model from `training.json` and write a ready-to-use
// model file (`model.json` by default).
//
// The training code is the real subsystem code in `subsystems/`, loaded the
// same way `test.html` and the experiments load it, so the model produced
// here behaves exactly like the one the browser app trains in the page.
//
// Usage:
//   npm run train                  # shipped task recipe (with supplemental training-only rows)
//   node train.mjs                # base training.json only
//   node train.mjs --epochs 400 --lr 0.2 --vector-size 24 --seed 7
//   node train.mjs --data training.json --out model.json
//
// Options:
//   --data <file>       training data (JSON array of { text, label })
//   --out <file>        where to write the trained model
//   --extra <file>      labelled examples added to train partition only
//   --extra-count <n>   prefix of supplemental examples (default: all)
//   --anchor-model <file> compatible v4 checkpoint with the same label policy
//   --anchor-mix <0..1> fraction of anchored weights after fitting/softening
//   --epochs <n>        training rounds (default 120 for v2/v3/v4)
//   --lr <float>        learning rate (default 0.8 for v2/v3/v4)
//   --vector-size <n>   dimensions per word vector (default 24)
//   --seed <n>          PRNG seed, for reproducibility (default 7)
//   --split-seed <n>    seed for the train/validation/test split (default 1)
//   --temperature <n>   score softening, 1 = off (default 4)
//   --encoder <name>   richer-subword-v4 (default), structured-subword-v3, hashed-subword-v2, legacy
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
        epochs: 120,
        learningRate: 0.8,
        vectorSize: 24,
        seed: 7,
        splitSeed: 1,
        temperature: 4,
        encoder: "richer-subword-v4",
        extra: null,
        extraCount: null,
        anchorModel: null,
        anchorMix: 0,
        quiet: false
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];

        switch (arg) {
            case "--data":       options.data = argv[++i]; break;
            case "--extra":      options.extra = argv[++i]; break;
            case "--extra-count": options.extraCount = Number(argv[++i]); break;
            case "--anchor-model": options.anchorModel = argv[++i]; break;
            case "--anchor-mix": options.anchorMix = Number(argv[++i]); break;
            case "--out":        options.out = argv[++i]; break;
            case "--epochs":     options.epochs = Number(argv[++i]); break;
            case "--lr":         options.learningRate = Number(argv[++i]); break;
            case "--vector-size": options.vectorSize = Number(argv[++i]); break;
            case "--seed":       options.seed = Number(argv[++i]); break;
            case "--split-seed": options.splitSeed = Number(argv[++i]); break;
            case "--temperature": options.temperature = Number(argv[++i]); break;
            case "--encoder":
                options.encoder = argv[++i];
                if (options.encoder === "legacy") {
                    // Keep historical defaults unless explicitly overridden.
                    if (!argv.includes("--epochs")) options.epochs = 60;
                    if (!argv.includes("--lr")) options.learningRate = 0.2;
                }
                break;
            case "--quiet":      options.quiet = true; break;
            case "--help":
                console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8")
                    .split("\nimport fs")[0]);
                process.exit(0);
                break;
            default:
                console.error(`Unknown option: ${arg}`);
                process.exit(1);
        }
    }

    if (!["richer-subword-v4", "structured-subword-v3", "hashed-subword-v2", "legacy"].includes(options.encoder)) {
        throw new Error("--encoder must be richer-subword-v4, structured-subword-v3, hashed-subword-v2 or legacy");
    }
    if (options.extraCount !== null && (!options.extra || !Number.isInteger(options.extraCount) || options.extraCount < 0)) {
        throw new Error("--extra-count requires --extra and a nonnegative integer");
    }
    if ((options.anchorModel && (!Number.isFinite(options.anchorMix) || options.anchorMix <= 0 || options.anchorMix > 1)) ||
        (!options.anchorModel && options.anchorMix !== 0) ||
        (options.anchorModel && (options.encoder !== "richer-subword-v4" || !Number.isFinite(options.temperature) || options.temperature <= 0))) {
        throw new Error("--anchor-model requires v4 and --anchor-mix in (0,1] with positive --temperature");
    }
    return options;
}

// ----- Data ----------------------------------------------------------------

// The validation rules live in `subsystems/load_training_data.js`, so the
// trainer and the browser app accept exactly the same files. This script
// reads from disk directly because `fetch` cannot open a `file://` path in
// Node.
function loadExamples(file) {
    const fullPath = path.resolve(here, file);
    let parsed;

    try {
        parsed = JSON.parse(fs.readFileSync(fullPath, "utf8"));
    } catch (error) {
        throw new Error(`Could not read ${file}: ${error.message}`);
    }

    return sprout.parse_training_data(parsed, file);
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

const { trainSet: baseTrainSet, validationSet, testSet } = sprout.split_dataset(examples, {
    seed: options.splitSeed
});
const supplement = options.extra ? loadExamples(options.extra) : [];
if (options.extraCount !== null && options.extraCount > supplement.length) {
    throw new Error("--extra-count exceeds the number of supplemental examples");
}
const additional = supplement.slice(0, options.extraCount ?? supplement.length);
const original = new Set(examples.map(example => sprout.normalise_text(example.text)));
const checked = new Set();
for (const example of additional) {
    const text = sprout.normalise_text(example.text);
    if (original.has(text) || checked.has(text)) throw new Error(`Supplemental duplicate / leakage: ${example.text}`);
    checked.add(text);
}
const trainSet = baseTrainSet.concat(additional);
log(`supplement: ${additional.length} training-only examples from ${options.extra || "none"}`);

// The vocabulary is built from the training split only, so the held-out
// test split stays a fair measure of how the model generalises.
const vocabulary = buildVocabulary(trainSet);
const model = sprout.initialise_model({
    vocabulary,
    encoder: options.encoder,
    vectorSize: options.vectorSize,
    seed: options.seed,
    learningRate: options.learningRate,
    epochs: options.epochs
});

log(`split     : ${trainSet.length} train / ${validationSet.length} validation / ${testSet.length} test`);
log(`vocabulary: ${model.vocabulary.length} observed tokens; encoder: ${options.encoder}`);
log(`parameters: ${sprout.model_parameter_stats(model).trainable.toLocaleString()} trainable scalars (incl. bias)`);
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

if (options.anchorModel) {
    const raw = fs.readFileSync(path.resolve(here, options.anchorModel), "utf8");
    const loaded = sprout.load_model(options.anchorModel, { storage: () => raw });
    if (!loaded.success) throw new Error(loaded.error);
    const anchor = loaded.model;
    if (anchor.options?.domain !== options.data || !(anchor.options.temperature > 0)) {
        throw new Error("Anchor must share the dataset label policy and have a positive temperature");
    }
    model.options.domain = options.data;
    sprout.blend_models(model, anchor, options.anchorMix, anchor.options.temperature / options.temperature);
    log(`anchor    : ${options.anchorMix} fraction from ${options.anchorModel} (one v4 classifier at inference)`);
}

const trainMetrics = sprout.evaluate_model(model, trainSet);
const validationMetrics = sprout.evaluate_model(model, validationSet);
const testMetrics = sprout.evaluate_model(model, testSet);

const parameterStats = sprout.model_parameter_stats(model);
log(`done in ${elapsed}s`);
log(`parameters: ${parameterStats.trainable.toLocaleString()} trainable; ` +
    `${parameterStats.nonZeroWeights.toLocaleString()} nonzero weights after training`);
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
    domain: options.data,
    temperature: options.temperature,
    trainedFrom: {
        data: options.data,
        examples: examples.length,
        trainSet: trainSet.length,
        baseTrainSet: baseTrainSet.length,
        extra: options.extra,
        extraRows: additional.length,
        ...(options.anchorModel ? { anchorModel: options.anchorModel, anchorMix: options.anchorMix } : {}),
        validationSet: validationSet.length,
        testSet: testSet.length,
        encoder: options.encoder,
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
