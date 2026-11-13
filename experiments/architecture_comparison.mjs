// Fixed-recipe v3/v4 comparison. Test and challenge fixtures are NEVER
// trained on. This audit reports capacity versus learned nonzero coefficients
// separately; a large hash table is not 524k learned facts about language.
import assert from "node:assert/strict";
import fs from "node:fs";
import { performance } from "node:perf_hooks";
import { sprout, buildVocabulary } from "./lib/sprout.mjs";

const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), "utf8"));
const transfer = read("transfer_cases.json");
const unfamiliar = Object.values(read("unfamiliar_cases.json").groups).flat();
const other = read("other_contexts.json");

function metrics(model, examples) {
    const m = sprout.evaluate_model(model, examples);
    return `${(m.accuracy * 100).toFixed(1)}% (TP ${m.truePositive}, FP ${m.falsePositive}, TN ${m.trueNegative}, FN ${m.falseNegative})`;
}
const transferRows = domain => Object.entries(transfer[domain])
    .flatMap(([label, texts]) => texts.map(text => ({ label, text })));

for (const [domain, dataFile, shippedFile] of [
    ["tasks", "task_baseline_training.json", "task_baseline_model.json"],
    ["commits", "../commit_training.json", "../commit_model.json"]
]) {
    const examples = read(dataFile);
    const split = sprout.split_dataset(examples, { seed: 1 });
    const extras = read(domain === "tasks" ? "task_supplement.json" : "commit_supplement.json")
        .slice(0, domain === "tasks" ? 70 : 80);
    const train = split.trainSet.concat(extras);
    const vocab = buildVocabulary(train);
    const shipped = read(shippedFile);
    console.log(`\n${domain}: same ${train.length} train (incl. ${extras.length} supplement) / ` +
        `${split.validationSet.length} validation / ${split.testSet.length} held-out test examples`);
    for (const [encoder, version] of [["structured-subword-v3", 3], ["richer-subword-v4", 4]]) {
        const model = sprout.initialise_model({ encoder, vocabulary: vocab, seed: 7 });
        const started = performance.now();
        sprout.train_model(train, split.validationSet, model, { epochs: 120, learningRate: 0.8 });
        const trainingMs = performance.now() - started;
        model.weights = model.weights.map(weight => weight / 4);
        model.bias /= 4;
        const stats = sprout.model_parameter_stats(model);
        const saved = sprout.save_model(model, "comparison.json", { storage: () => {} });
        assert.ok(saved.success);
        const loaded = sprout.load_model("comparison.json", { storage: () => saved.payload });
        assert.ok(loaded.success);
        assert.equal(stats.trainable, version === 4 ? 524289 : 131073);
        assert.ok(stats.nonZeroWeights < stats.trainable);
        assert.ok(Buffer.byteLength(saved.payload) < 650_000);
        for (const row of split.testSet) {
            assert.equal(await sprout.classify_text(row.text, loaded.model),
                await sprout.classify_text(row.text, model));
        }
        if (version === 4) {
            // Confirm the shipped model uses this exact training recipe.
            assert.equal(shipped.version, 4);
            const result = sprout.load_model(shippedFile, {
                storage: () => JSON.stringify(shipped)
            });
            assert.ok(result.success);
            assert.deepEqual(result.model.weights, model.weights);
            assert.equal(result.model.bias, model.bias);
        }
        console.log(`  v${version}: trainable ${stats.trainable}, nonzero ${stats.nonZeroWeights}; ` +
            `artifact ${Buffer.byteLength(saved.payload)} bytes; train ${trainingMs.toFixed(0)} ms`);
        console.log(`      held-out ${metrics(model, split.testSet)}; transfer ${metrics(model, transferRows(domain))}`);
        if (domain === "tasks") {
            console.log(`      new vocab ${metrics(model, unfamiliar)}; ` +
                `helpdesk hard ${metrics(model, other.helpdesk.stress)}; ` +
                `agendas hard ${metrics(model, other.meeting_agendas.stress)}`);
        }
        const sample = transferRows(domain);
        const t0 = performance.now();
        for (let i = 0; i < 1000; i++) await sprout.classify_text(sample[i % sample.length].text, model);
        console.log(`      1000 local scores in ${(performance.now() - t0).toFixed(0)} ms (machine-dependent)`);
    }
}
console.log("\nNumbers above are developer-authored probes, not a deployment benchmark.");
