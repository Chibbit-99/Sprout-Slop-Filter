// Validation-only selection for a SAME-SIZE v4 checkpoint blend.
// The held-out and challenge rows are read only AFTER the anchor is chosen.
// Checkpoints share a task label policy and an identical feature map; the
// expanded vocabulary stays untouched and all inference uses one model.
import assert from "node:assert/strict";
import fs from "node:fs";
import { sprout } from "./lib/sprout.mjs";

const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), "utf8"));
function load(file) {
    const loaded = sprout.load_model(file, { storage: () => fs.readFileSync(new URL(file, import.meta.url), "utf8") });
    assert.ok(loaded.success, loaded.error);
    return loaded.model;
}
const expanded = load("task_unanchored_model.json");
const baseline = load("task_baseline_model.json");
assert.equal(expanded.options.domain, baseline.options.domain);
assert.equal(expanded.options.temperature, baseline.options.temperature);
const full = sprout.split_dataset(read("../training.json"), { seed: 1 });
const frozen = sprout.split_dataset(read("task_baseline_training.json"), { seed: 1 });
const fractions = [0, 0.15, 0.30, 0.45, 0.60, 0.75];
const evaluate = (model, rows) => {
    const m = sprout.evaluate_model(model, rows);
    return `${m.truePositive + m.trueNegative}/${m.count} (FP ${m.falsePositive}, FN ${m.falseNegative})`;
};
let chosen = null;
console.log("Selection uses the frozen original VALIDATION rows only:");
for (const fraction of fractions) {
    const model = { ...expanded, weights: expanded.weights.slice() };
    sprout.blend_models(model, baseline, fraction);
    const result = sprout.evaluate_model(model, frozen.validationSet);
    console.log(`  reference fraction ${fraction.toFixed(2)}: ${evaluate(model, frozen.validationSet)}`);
    // Select accuracy first, then the lower false-positive count; on a tie,
    // the first candidate uses less of the historical anchor.
    if (!chosen || result.accuracy > chosen.result.accuracy ||
        (result.accuracy === chosen.result.accuracy && result.falsePositive < chosen.result.falsePositive)) {
        chosen = { fraction, result, model };
    }
}
assert.equal(chosen.fraction, 0.45, "the recorded recipe must track the validation selection");
console.log(`Chosen by validation only: ${chosen.fraction.toFixed(2)}; same ${sprout.model_parameter_stats(chosen.model).trainable} parameters`);
console.log(`Expanded validation: ${evaluate(chosen.model, full.validationSet)}`);
// Diagnostic slices below did NOT determine the selection. They were already
// inspected by developers and do not estimate independent deployment quality.
console.log(`Expanded held-out: ${evaluate(chosen.model, full.testSet)}`);
console.log(`Frozen old test: ${evaluate(chosen.model, frozen.testSet)}`);
for (const [name, rows] of Object.entries(read("unfamiliar_cases.json").groups)) {
    console.log(`  ${name}: ${evaluate(chosen.model, rows)}`);
}
console.log("Manually authored probes and historical validation have been inspected; these are not unbiased accuracy estimates.");
