// Fixed, manual diagnostic. The 60 training rows are separate from all
// ordinary, hard and new probe rows. No evaluation text is used in training.
import assert from "node:assert/strict";
import fs from "node:fs";
import { sprout, buildVocabulary } from "./lib/sprout.mjs";

const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), "utf8"));
const contexts = read("other_contexts.json");
const probes = read("domain_probes.json");
const load = file => {
    const raw = fs.readFileSync(new URL("../" + file, import.meta.url), "utf8");
    const result = sprout.load_model(file, { storage: () => raw });
    assert.equal(result.success, true, result.error);
    assert.equal(sprout.model_parameter_stats(result.model).trainable, 524289);
    return result.model;
};
const zeroShot = { tasks: load("model.json"), baseline: load("experiments/task_baseline_model.json"),
    commits: load("commit_model.json") };
function outcome(model, rows) {
    const m = sprout.evaluate_model(model, rows);
    return `${(m.accuracy * 100).toFixed(1)}% (TP ${m.truePositive} / FP ${m.falsePositive} / TN ${m.trueNegative} / FN ${m.falseNegative})`;
}
for (const [domain, file, modelFile] of [
    ["helpdesk", "helpdesk_training.json", "helpdesk_model.json"],
    ["agendas", "agenda_training.json", "agenda_model.json"]
]) {
    const key = domain === "agendas" ? "meeting_agendas" : "helpdesk";
    const data = contexts[key];
    const train = read("../" + file);
    const model = load(modelFile);
    assert.deepEqual(train, data.train, "tracked training data changed independently of frozen fixtures");
    assert.equal(model.options.domain, file);
    assert.equal(model.options.trainedFrom.trainSet, 60);
    assert.deepEqual([...model.vocabulary].slice(1), [...buildVocabulary(train)]);
    const known = new Set(model.vocabulary);
    const existing = new Set([...train, ...data.test, ...data.stress]
        .map(row => sprout.normalise_text(row.text)));
    const newTexts = new Set();
    for (const row of probes[domain]) {
        const normalized = sprout.normalise_text(row.text);
        assert.ok(!existing.has(normalized) && !newTexts.has(normalized), `probe leak: ${row.text}`);
        newTexts.add(normalized);
        assert.ok(["genuine", "slop"].includes(row.label));
        const tokens = sprout.tokenise(normalized);
        assert.ok(row.unseen?.length > 0);
        for (const word of row.unseen) assert.ok(tokens.includes(word) && !known.has(word),
            `annotated OOV token is known: ${word}`);
    }
    assert.equal(probes[domain].length, 24);
    assert.equal(probes[domain].filter(row => row.label === "slop").length, 12);
    console.log(`\n${domain}: 60 training-only rows, 24 ordinary / 16 hard / 24 newly written probes`);
    for (const [name, candidate] of [
        ["expanded task (zero-shot)", zeroShot.tasks],
        ["historical task (zero-shot)", zeroShot.baseline],
        ["shipped commit (zero-shot)", zeroShot.commits],
        ["domain-trained v4", model]
    ]) {
        console.log(`  ${name}: ordinary ${outcome(candidate, data.test)}; ` +
            `hard ${outcome(candidate, data.stress)}; new ${outcome(candidate, probes[domain])}`);
        if (name === "domain-trained v4") {
            for (const row of probes[domain]) {
                const score = sprout.make_prediction_for_training(row.text, model).score;
                if ((score >= 0.5) !== (row.label === "slop")) {
                    console.log(`    miss: ${score.toFixed(3)} ${row.label}: ${row.text}`);
                }
            }
        }
    }
}
console.log("\nManual, inspected developer probes are not a production accuracy estimate.");
