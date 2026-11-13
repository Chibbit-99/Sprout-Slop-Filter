// Reproducible checks for the shipped v4 models and older version loading. The 40
// transfer cases are independently written, not in either training file.
import assert from "node:assert/strict";
import fs from "node:fs";
import { performance } from "node:perf_hooks";
import { sprout, buildVocabulary, slopBreakdown } from "./lib/sprout.mjs";

const cases = JSON.parse(fs.readFileSync(new URL("./transfer_cases.json", import.meta.url)));
for (const [domain, dataFile, modelFile] of [
    ["tasks", "training.json", "model.json"],
    ["commits", "commit_training.json", "commit_model.json"]
]) {
    const examples = JSON.parse(fs.readFileSync(new URL("../" + dataFile, import.meta.url)));
    const raw = fs.readFileSync(new URL("../" + modelFile, import.meta.url), "utf8");
    const loaded = sprout.load_model(modelFile, { storage: () => raw });
    assert.equal(loaded.success, true, loaded.error);
    const model = loaded.model;
    assert.equal(model.version, 4);
    const stats = sprout.model_parameter_stats(model);
    assert.equal(stats.trainable, 524289);
    assert.equal(stats.classifierWeights, 524288);
    assert.equal(stats.embeddingWeights, 0);
    assert.equal(Object.keys(stats.families).length, 8);
    assert.ok(Object.values(stats.families).every(family => family.slots === 65536));
    assert.equal(stats.nonZeroWeights, Object.values(stats.families)
        .reduce((sum, family) => sum + family.nonZero, 0));
    assert.ok(stats.nonZeroWeights > 1000 && stats.nonZeroWeights < stats.trainable,
        "capacity is not the same as learned, nonzero weights");
    assert.equal(model.weights.length, model.featureSize);
    assert.ok(Buffer.byteLength(raw) < (domain === "tasks" ? 2_000_000 : 600_000), "model should remain small on disk");

    if (domain === "tasks") {
        const loadCheckpoint = file => {
            const payload = fs.readFileSync(new URL(file, import.meta.url), "utf8");
            const result = sprout.load_model(file, { storage: () => payload });
            assert.ok(result.success, result.error);
            return result.model;
        };
        const old = loadCheckpoint("task_baseline_model.json");
        const plain = loadCheckpoint("task_unanchored_model.json");
        const reconstructed = { ...plain, weights: plain.weights.slice() };
        assert.equal(model.options.trainedFrom.anchorMix, 0.45);
        assert.equal(model.options.trainedFrom.anchorModel, "experiments/task_baseline_model.json");
        sprout.blend_models(reconstructed, old, 0.45);
        assert.deepEqual(reconstructed.weights, model.weights, "one blended checkpoint reproduces shipped weights");
        assert.equal(reconstructed.bias, model.bias);
        assert.equal(sprout.model_parameter_stats(reconstructed).trainable, stats.trainable);
        const logit = p => Math.log(p / (1 - p));
        for (const text of ["Reupholster the armchair with velvet offcuts", "Fix the leaking tap", "Do something eventually"]) {
            const score = sprout.make_prediction_for_training(text, model).score;
            const original = sprout.make_prediction_for_training(text, plain).score;
            const prior = sprout.make_prediction_for_training(text, old).score;
            assert.ok(Math.abs(logit(score) - (0.55 * logit(original) + 0.45 * logit(prior))) < 1e-8,
                "one weight array exactly represents a convex blend of logits");
        }
        assert.throws(() => sprout.blend_models(reconstructed, { ...old, version: 3 }, 0.45), /compatible v4/);
        assert.throws(() => sprout.blend_models(reconstructed, old, -0.1), /compatible v4/);
        assert.throws(() => sprout.blend_models(reconstructed,
            { ...old, options: { ...old.options, domain: "commit_training.json" } }, 0.45),
        /shared label policy/);
        for (const word of ["glimmering", "fluxing"]) {
            assert.ok(!model.vocabulary.includes(word));
        }
        assert.notEqual(await sprout.classify_text("Service the glimmering capacitor", model),
            await sprout.classify_text("Service the fluxing capacitor", model),
            "unseen words must not share an <UNK> score");
        assert.notEqual(await sprout.classify_text("buy milk", model),
            await sprout.classify_text("milk buy", model), "local order should affect the score");
    }

    const split = sprout.split_dataset(examples, { seed: 1 });
    const extraFile = domain === "tasks" ? "task_supplement.json" : "commit_supplement.json";
    const extras = JSON.parse(fs.readFileSync(new URL(extraFile, import.meta.url), "utf8"));
    const usedExtras = extras.slice(0, domain === "tasks" ? extras.length : 80);
    const trainWords = buildVocabulary(split.trainSet.concat(usedExtras));
    if (domain === "tasks") {
        const old = JSON.parse(fs.readFileSync(new URL("task_baseline_training.json", import.meta.url)));
        const frozen = sprout.split_dataset(old, { seed: 1 });
        const trainTexts = new Set(split.trainSet.concat(usedExtras).map(row => sprout.normalise_text(row.text)));
        for (const row of [...frozen.validationSet, ...frozen.testSet]) {
            assert.ok(!trainTexts.has(sprout.normalise_text(row.text)), "original held-out row leaked into expanded training");
        }
        const newTest = new Set(split.testSet.map(row => sprout.normalise_text(row.text)));
        for (const row of frozen.testSet) {
            assert.ok(newTest.has(sprout.normalise_text(row.text)), "original test split must stay frozen");
        }
        const designated = Object.values(JSON.parse(fs.readFileSync(
            new URL("unfamiliar_cases.json", import.meta.url))).groups).flat();
        for (const row of designated) {
            for (const token of row.unseen) assert.ok(!trainWords.includes(token),
                `designated challenge word leaked: ${token}`);
        }
    }
    assert.equal((model.options?.trainedFrom || {}).extraRows, usedExtras.length);
    if (domain === "tasks") {
        assert.ok(examples.length >= 1000, "displayed corpus has at least 1000 labelled examples");
        assert.ok(split.trainSet.length + usedExtras.length >= 1000, "actual training partition has at least 1000 rows");
        assert.ok(trainWords.length >= 3000, "vocabulary comes from real training tokens, not padding");
        assert.ok(trainWords.filter(word => /^\p{L}+$/u.test(word)).length >= 3000,
            "at least 3000 alphabetic words, excluding punctuation, numbers and <UNK>");
    }
    assert.deepEqual([...model.vocabulary].slice(1), [...trainWords]);
    const metrics = sprout.evaluate_model(model, split.testSet);
    const breakdown = slopBreakdown(metrics);
    console.log(`${domain}: ${examples.length} examples, ${trainWords.length} training words, ${Buffer.byteLength(raw)} bytes`);
    console.log(`  ${stats.trainable} trainable parameters; ${stats.nonZeroWeights} nonzero weights`);
    console.log(`  held-out: ${(metrics.accuracy * 100).toFixed(1)}% accuracy, ${(breakdown.falsePositiveRate * 100).toFixed(1)}% false positives (${metrics.count} examples)`);
    assert.ok(metrics.accuracy >= (domain === "tasks" ? 0.95 : 0.85));

    // Regression against an unseen set. These are small hand-written probes,
    // not a representative production benchmark; print the errors too.
    let correct = 0;
    let unseen = 0;
    let total = 0;
    for (const [label, texts] of Object.entries(cases[domain])) {
        for (const text of texts) {
            assert.ok(!examples.some(example => example.text.toLowerCase() === text.toLowerCase()),
                "transfer case leaked into training: " + text);
            const score = await sprout.classify_text(text, model);
            assert.ok(Number.isFinite(score) && score >= 0 && score <= 1);
            const predicted = score >= 0.5 ? "slop" : "genuine";
            if (predicted === label) correct++;
            else console.log(`  miss: ${score.toFixed(3)} (${label}) ${text}`);
            for (const word of sprout.tokenise(sprout.normalise_text(text))) {
                total++;
                if (!model.vocabulary.includes(word)) unseen++;
            }
        }
    }
    console.log(`  transfer: ${correct}/20 correct; ${unseen}/${total} tokens unseen in training`);
    assert.ok(correct >= 17, "transfer probes regressed");
    assert.ok(unseen > 0, "transfer suite must include unknown words");

    // Same model and feature extractor on save/load and resumed training.
    const serialized = sprout.save_model(model, modelFile, { storage: () => {} });
    assert.ok(serialized.success);
    const roundtrip = sprout.load_model(modelFile, { storage: () => serialized.payload });
    assert.ok(roundtrip.success);
    for (const text of cases[domain].genuine) {
        assert.equal(await sprout.classify_text(text, model), await sprout.classify_text(text, roundtrip.model));
        assert.equal(sprout.make_prediction_for_training(text, model).score, await sprout.classify_text(text, model));
    }
    sprout.train_model(split.trainSet.slice(0, 2), [], roundtrip.model, { epochs: 1 });

    // v3, v2 and v1 models remain readable and score identically after roundtrip.
    const older = sprout.initialise_model({ encoder: "structured-subword-v3", vocabulary: trainWords.slice(0, 10) });
    sprout.train_model(split.trainSet.slice(0, 4), [], older, { epochs: 2, learningRate: 0.8 });
    const v3Saved = sprout.save_model(older, "v3.json", { storage: () => {} });
    const v3Loaded = sprout.load_model("v3.json", { storage: () => v3Saved.payload });
    assert.equal(v3Loaded.model.version, 3);
    assert.equal(await sprout.classify_text("repair the tap", older),
        await sprout.classify_text("repair the tap", v3Loaded.model));
    assert.equal(sprout.model_parameter_stats(v3Loaded.model).trainable, 131073);
    const previous = sprout.initialise_model({ encoder: "hashed-subword-v2", vocabulary: trainWords.slice(0, 10) });
    sprout.train_model(split.trainSet.slice(0, 4), [], previous, { epochs: 2, learningRate: 0.8 });
    const v2Saved = sprout.save_model(previous, "v2.json", { storage: () => {} });
    const v2Loaded = sprout.load_model("v2.json", { storage: () => v2Saved.payload });
    assert.equal(v2Loaded.model.version, 2);
    assert.equal(await sprout.classify_text("repair the tap", previous),
        await sprout.classify_text("repair the tap", v2Loaded.model));
    assert.equal(sprout.model_parameter_stats(v2Loaded.model).trainable, 32769);

    // Legacy models remain readable; legacy training is still available.
    const legacy = sprout.initialise_model({ vocabulary: trainWords.slice(0, 10), vectorSize: 8, seed: 7 });
    const old = sprout.save_model(legacy, "old.json", { storage: () => {} });
    assert.equal(sprout.load_model("old.json", { storage: () => old.payload }).success, true);
    assert.ok(Number.isFinite(await sprout.classify_text("buy milk", legacy)));

    const started = performance.now();
    for (let i = 0; i < 1000; i++) await sprout.classify_text(cases[domain].genuine[i % 10], model);
    console.log(`  1,000 local predictions: ${(performance.now() - started).toFixed(0)} ms`);
}
const invalid = sprout.load_model("x", { storage: () => JSON.stringify({
    version: 2, options: { encoder: "hashed-subword-v2" }, featureSize: 1024,
    vocabulary: [], weights: [[1024, 42]], bias: 0
}) });
assert.equal(invalid.success, false);
const invalidV3 = sprout.load_model("x", { storage: () => JSON.stringify({
    version: 3, options: { encoder: "structured-subword-v3" }, featureSize: 32768,
    vocabulary: [], weights: [], bias: 0
}) });
assert.equal(invalidV3.success, false);
const invalidV4 = sprout.load_model("x", { storage: () => JSON.stringify({
    version: 4, options: { encoder: "richer-subword-v4" }, featureSize: 131072,
    vocabulary: [], weights: [], bias: 0
}) });
assert.equal(invalidV4.success, false);
assert.throws(() => sprout.initialise_model({ encoder: "richer-subword-v4", featureSize: 131072 }),
    /524288/);
assert.throws(() => sprout.initialise_model({ encoder: "structured-subword-v3", featureSize: 42 }),
    /131072/);
assert.notDeepEqual(sprout.hashed_features("buy milk", 32768), sprout.hashed_features("milk buy", 32768));
assert.notDeepEqual(sprout.hashed_features("glimmers", 32768), sprout.hashed_features("fluxes", 32768));
const first = sprout.structured_features("buy milk", 131072);
const second = sprout.structured_features("milk buy", 131072);
assert.notDeepEqual(first, second);
assert.ok(first.some(([id]) => id < 32768));
assert.ok(first.some(([id]) => id >= 32768 && id < 65536));
assert.ok(first.some(([id]) => id >= 65536 && id < 98304));
assert.ok(first.some(([id]) => id >= 98304));
const v4 = sprout.richer_features("buy milk", 524288);
assert.ok([0, 1, 2, 3, 4, 6, 7].every(family =>
    v4.some(([id]) => id >= family * 65536 && id < (family + 1) * 65536)));
assert.notDeepEqual(v4, sprout.richer_features("milk buy", 524288));
console.log("v4, v3, v2 and legacy regressions passed");
