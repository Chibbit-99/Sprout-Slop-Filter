// OOV/context stress tests with fixed, manually labelled fixtures. Never
// refit or tune on the challenge/test rows. These are small illustrative
// probes, not independent external validation or a measure of authenticity.
import assert from "node:assert/strict";
import fs from "node:fs";
import { sprout, buildVocabulary } from "./lib/sprout.mjs";

const read = relative => JSON.parse(fs.readFileSync(new URL(relative, import.meta.url), "utf8"));
const unfamiliar = read("./unfamiliar_cases.json");
const other = read("./other_contexts.json");
const taskExamples = read("task_baseline_training.json");
const commitExamples = read("../commit_training.json");
const taskExtra = read("./task_supplement.json").slice(0, 70);
const taskTrain = sprout.split_dataset(taskExamples, { seed: 1 }).trainSet.concat(taskExtra);

function loadShipped(file) {
    const raw = fs.readFileSync(new URL("../" + file, import.meta.url), "utf8");
    const loaded = sprout.load_model(file, { storage: () => raw });
    assert.equal(loaded.success, true, loaded.error);
    assert.equal(loaded.model.version, 4);
    return loaded.model;
}
const tasks = loadShipped("experiments/task_baseline_model.json");
const commits = loadShipped("commit_model.json");

function train(examples, encoder) {
    const model = sprout.initialise_model({ encoder, vocabulary: buildVocabulary(examples),
        vectorSize: 24, seed: 7 });
    sprout.train_model(examples, [], model, { epochs: encoder === "legacy" ? 60 : 120,
        learningRate: encoder === "legacy" ? 0.2 : 0.8 });
    // Apply the same monotonic ×4 score softening as the shipped models.
    model.weights = model.weights.map(weight => weight / 4);
    model.bias /= 4;
    return model;
}
const taskLegacy = train(taskTrain, "legacy");
const tokenise = text => sprout.tokenise(sprout.normalise_text(text));

function validate(trainRows, testRows, source, fullVocabulary) {
    const trainTexts = new Set(trainRows.map(row => sprout.normalise_text(row.text)));
    const known = new Set(buildVocabulary(trainRows));
    const all = new Set(trainTexts);
    assert.equal(trainTexts.size, trainRows.length, `${source}: duplicate train text`);
    assert.ok(testRows.length >= 20, `${source}: too few test cases`);
    for (const row of testRows) {
        assert.ok(row && typeof row.text === "string" && row.text.trim(), `${source}: missing text`);
        assert.ok(["genuine", "slop"].includes(row.label), `${source}: invalid label`);
        const normalized = sprout.normalise_text(row.text);
        assert.ok(!all.has(normalized), `${source}: duplicate or train/test leak: ${row.text}`);
        all.add(normalized);
        assert.ok(Array.isArray(row.unseen) && row.unseen.length,
            `${source}: row needs a pre-specified OOV token: ${row.text}`);
        const tokens = tokenise(row.text);
        for (const word of row.unseen) {
            assert.ok(tokens.includes(word), `${source}: ${word} is not a token of '${row.text}'`);
            assert.ok(!known.has(word), `${source}: ${word} occurs in training`);
            if (fullVocabulary) assert.ok(!fullVocabulary.has(word), `${source}: ${word} occurs in complete corpus`);
        }
    }
    assert.ok(testRows.some(row => row.label === "genuine") && testRows.some(row => row.label === "slop"),
        `${source}: need both labels`);
}

async function evaluate(rows, model, known, verbose = false) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    let unknown = 0, total = 0;
    const misses = [];
    for (const row of rows) {
        const score = await sprout.classify_text(row.text, model);
        assert.ok(Number.isFinite(score) && score >= 0 && score <= 1);
        assert.equal(score, sprout.make_prediction_for_training(row.text, model).score,
            "inference and evaluation must share the same features");
        const predicted = score >= 0.5 ? "slop" : "genuine";
        if (predicted === "slop" && row.label === "slop") tp++;
        else if (predicted === "slop") fp++;
        else if (row.label === "genuine") tn++;
        else fn++;
        if (predicted !== row.label) misses.push(`${score.toFixed(3)} ${row.label}: ${row.text}`);
        for (const word of tokenise(row.text)) {
            total++;
            if (!known.has(word)) unknown++;
        }
    }
    if (verbose) for (const miss of misses) console.log(`    miss: ${miss}`);
    return { accuracy: (tp + tn) / rows.length, tp, fp, tn, fn, unknown, total, misses: misses.length };
}
function fmt(metrics) {
    return `${(metrics.accuracy * 100).toFixed(1)}% (${metrics.tp} TP, ${metrics.fp} FP, ${metrics.tn} TN, ${metrics.fn} FN)`;
}

console.log("=== Unfamiliar-word challenge: task policy (40 rows, completely unseen target tokens) ===");
const groups = Object.entries(unfamiliar.groups);
const allRows = groups.flatMap(([, rows]) => rows);
const fullTaskWords = new Set(buildVocabulary(taskExamples.concat(taskExtra)));
validate(taskExamples.concat(taskExtra), allRows, "task OOV", fullTaskWords);
assert.equal(allRows.length, 40);
for (const [name, rows] of groups) {
    assert.equal(rows.length, 8, `${name}: expected 8 examples`);
    assert.equal(rows.filter(row => row.label === "slop").length, 4,
        `${name}: balance changed`);
    const v4 = await evaluate(rows, tasks, new Set(tasks.vocabulary));
    const v1 = await evaluate(rows, taskLegacy, new Set(taskLegacy.vocabulary));
    console.log(`  ${name.padEnd(30)} v4 ${fmt(v4)} | legacy ${fmt(v1)} | ${v4.unknown}/${v4.total} OOV tokens`);
}
const oovV4 = await evaluate(allRows, tasks, new Set(tasks.vocabulary), true);
const oovV1 = await evaluate(allRows, taskLegacy, new Set(taskLegacy.vocabulary));
console.log(`  total: v4 ${fmt(oovV4)}; legacy ${fmt(oovV1)}; OOV ${oovV4.unknown}/${oovV4.total}`);
assert.ok(oovV4.unknown > 0 && oovV4.unknown < oovV4.total);
// An unfamiliar subject by itself is not a decision: the same unseen word
// appears in both examples, with different surrounding information.
assert.ok(await sprout.classify_text("Inspect the cryostat for cracked seals", tasks) <
    await sprout.classify_text("Enhance the cryostat vibe in general", tasks));
// Regression floors for these authored cases, NOT deployment targets.
assert.ok(oovV4.accuracy >= 0.70, "task OOV challenge regressed");

console.log("\n=== Different situations (24 ordinary + 16 hard test examples per use case) ===");
for (const [domain, { policy, train: trainRows, test: testRows, stress: stressRows }] of Object.entries(other)) {
    validate(trainRows, [...testRows, ...stressRows], domain);
    assert.equal(trainRows.length, 60);
    assert.equal(testRows.length, 24);
    assert.equal(stressRows.length, 16);
    for (const rows of [trainRows, testRows, stressRows]) {
        assert.equal(rows.filter(row => row.label === "slop").length, rows.length / 2,
            `${domain}: label balance changed`);
    }
    console.log(`${domain}: ${policy}`);
    const trained = train(trainRows, "richer-subword-v4");
    const legacy = train(trainRows, "legacy");
    const known = new Set(trained.vocabulary);
    for (const [name, model] of [["task zero-shot", tasks], ["commit zero-shot", commits],
        ["domain-trained v4", trained], ["domain-trained v3", train(trainRows, "structured-subword-v3")],
        ["domain-trained v2", train(trainRows, "hashed-subword-v2")],
        ["domain-trained legacy", legacy]]) {
        const modelKnown = name.includes("zero-shot") ? new Set(model.vocabulary) : known;
        const m = await evaluate(testRows, model, modelKnown);
        const hard = await evaluate(stressRows, model, modelKnown, name === "domain-trained v4");
        console.log(`  ${name.padEnd(21)} ordinary ${fmt(m)} | hard ${fmt(hard)} | hard OOV ${hard.unknown}/${hard.total}`);
        if (name === "domain-trained v4") {
            assert.ok(m.accuracy >= 0.75 && hard.accuracy >= 0.625,
                `${domain}: contextual regression floor failed`);
            const saved = sprout.save_model(model, "memory.json", { storage: () => {} });
            assert.ok(saved.success);
            const roundTrip = sprout.load_model("memory.json", { storage: () => saved.payload });
            assert.ok(roundTrip.success);
            for (const row of [...testRows, ...stressRows]) assert.equal(await sprout.classify_text(row.text, model),
                await sprout.classify_text(row.text, roundTrip.model));
            console.log(`  transient v4 model: ${Buffer.byteLength(saved.payload)} bytes (not shipped)`);
        }
    }
}
console.log("Challenge completed; diagnostic scores are not deployment guarantees.");
