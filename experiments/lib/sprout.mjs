// Loads the real Sprout subsystem files the same way test.html does — as
// classic scripts sharing one global scope — so experiments measure the
// actual implementation, not a reimplementation of it.

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const subsystemDir = path.join(here, "..", "..", "subsystems");

const source = fs
    .readdirSync(subsystemDir)
    .filter(name => name.endsWith(".js"))
    .sort()
    .map(name => fs.readFileSync(path.join(subsystemDir, name), "utf8"))
    .join("\n");

const context = vm.createContext({ console, fetch, URL, Set, Map, Math, JSON, Promise, Number, Array, Object });

vm.runInContext(
    `${source}
     globalThis.__sprout = {
         normalise_text, tokenise, encode_text, predict_slop,
         calculate_loss, update_parameters, make_prediction_for_training,
         evaluate_model, train_model, initialise_model, get_word_vector,
         split_dataset, save_model, load_model, classify_text,
         map_synonyms, get_synonyms_cached, clear_synonym_cache,
         build_vocabulary, parse_training_data
     };`,
    context
);

export const sprout = context.__sprout;

// Shared helpers for the CLI tools (`train.mjs`) and the experiments, so
// every entry point agrees on how a vocabulary is derived and how a model
// is written to disk.

// Every token the examples use, in first-seen order. The vocabulary is
// built the same way at training time and at prediction time, so the
// model's word vectors line up with the tokens it will later receive.
// This is `build_vocabulary` from subsystems/, under the camelCase name the
// experiments have always used.
export const buildVocabulary = sprout.build_vocabulary;

// Percentage of a set that is predicted as slop, broken down by the true
// label. Used to report the false-positive rate separately from accuracy.
export function slopBreakdown(result) {
    const slopTotal = result.truePositive + result.falseNegative;
    const genuineTotal = result.trueNegative + result.falsePositive;
    return {
        accuracy: result.accuracy,
        precision: result.precision,
        recall: result.recall,
        slopRecall: slopTotal === 0 ? 0 : result.truePositive / slopTotal,
        falsePositiveRate: genuineTotal === 0 ? 0 : result.falsePositive / genuineTotal,
        averageLoss: result.averageLoss
    };
}
