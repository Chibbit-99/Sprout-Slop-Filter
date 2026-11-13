// Freeze the original seed-1 validation/test split, then add *new labelled
// training-only* examples at balanced increments. Never leak a probe row into
// training. Measures coverage AND accuracy; bigger vocabulary is not a goal
// by itself. All figures are developer-authored diagnostics.
import assert from "node:assert/strict";
import fs from "node:fs";
import { sprout, buildVocabulary } from "./lib/sprout.mjs";

const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), "utf8"));
const transfer = read("./transfer_cases.json");
const unfamiliar = Object.values(read("./unfamiliar_cases.json").groups).flat();
const contexts = read("./other_contexts.json");
const normalize = text => sprout.normalise_text(text);
const score = (model, rows) => {
    const m = sprout.evaluate_model(model, rows);
    return `${(m.accuracy * 100).toFixed(1)}% (FP ${m.falsePositive}, FN ${m.falseNegative})`;
};
const coverage = (model, rows) => {
    const known = new Set(model.vocabulary);
    let total = 0, unseen = 0;
    for (const row of rows) for (const word of sprout.tokenise(normalize(row.text))) {
        total++;
        if (!known.has(word)) unseen++;
    }
    return `${unseen}/${total} OOV`;
};

for (const [domain, file, supplementFile, checkpoints] of [
    ["tasks", "task_baseline_training.json", "./task_supplement.json", [0, 70, 140]],
    ["commits", "../commit_training.json", "./commit_supplement.json", [0, 40, 80]]
]) {
    const examples = read(file), extra = read(supplementFile);
    const split = sprout.split_dataset(examples, { seed: 1 });
    const baseTexts = new Set(examples.map(row => normalize(row.text)));
    const newTexts = new Set();
    for (const row of extra) {
        assert.ok(row && typeof row.text === "string" && ["genuine", "slop"].includes(row.label));
        const text = normalize(row.text);
        assert.ok(!baseTexts.has(text) && !newTexts.has(text), `training leak or duplicate: ${text}`);
        newTexts.add(text);
    }
    const probeRows = Object.entries(transfer[domain])
        .flatMap(([label, texts]) => texts.map(text => ({ label, text })));
    for (const row of probeRows) assert.ok(!newTexts.has(normalize(row.text)));
    if (domain === "tasks") {
        const extraWords = new Set(buildVocabulary(extra));
        for (const row of unfamiliar) {
            assert.ok(!newTexts.has(normalize(row.text)));
            for (const token of row.unseen) assert.ok(!extraWords.has(token),
                `task challenge token leaked into supplement: ${token}`);
        }
        for (const group of Object.values(contexts)) {
            for (const row of [...group.test, ...group.stress]) {
                assert.ok(!newTexts.has(normalize(row.text)));
            }
        }
    }
    console.log(`\n${domain}: frozen ${split.trainSet.length}/${split.validationSet.length}/${split.testSet.length} original split, ` +
        `${extra.length} balanced new training-only examples`);
    for (const encoder of ["structured-subword-v3", "richer-subword-v4"]) {
        for (const added of checkpoints) {
            const train = split.trainSet.concat(extra.slice(0, added));
            assert.equal(train.filter(row => row.label === "genuine").length -
                split.trainSet.filter(row => row.label === "genuine").length, added / 2);
            const model = sprout.initialise_model({ encoder, vocabulary: buildVocabulary(train), seed: 7 });
            sprout.train_model(train, split.validationSet, model, { epochs: 120, learningRate: 0.8 });
            model.weights = model.weights.map(weight => weight / 4);
            model.bias /= 4;
            const stats = sprout.model_parameter_stats(model);
            console.log(`  v${model.version} +${String(added).padEnd(3)}: ${model.vocabulary.length - 1} observed train tokens, ` +
                `${stats.nonZeroWeights} nonzero / ${stats.trainable} trainable; ` +
                `val ${score(model, split.validationSet)}, test ${score(model, split.testSet)} (${coverage(model, split.testSet)}); ` +
                `transfer ${score(model, probeRows)}`);
            if (domain === "tasks") {
                console.log(`          unfamiliar ${score(model, unfamiliar)} (${coverage(model, unfamiliar)}); ` +
                    `helpdesk hard ${score(model, contexts.helpdesk.stress)}; ` +
                    `agendas hard ${score(model, contexts.meeting_agendas.stress)}`);
            }
        }
    }
}
console.log("\nAll probes remain excluded from supplement training. These authored results are not a production benchmark.");
