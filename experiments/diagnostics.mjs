// Diagnostics for the two ceilings that limit Sprout independently of how
// much training data it receives.
//
// A) Optimisation check — before the word-vector fix (FINDINGS.md section 1)
//    the subsystem code trained only 9 numbers (8 weights + bias) and held-out
//    accuracy stayed flat no matter how many epochs ran. With word vectors
//    trainable, this sweep shows what training length buys now: accuracy
//    should improve with epochs and then converge (or overfit), rather than
//    sit flat from the start.
//
// B) Vocabulary-coverage ceiling — every word the model has never seen
//    collapses to the single shared <UNK> vector, so unknown content words
//    carry no information. Measured by sweeping the out-of-vocabulary rate.
//
// Both sections now run the real subsystem trainer (`sprout.train_model`),
// not the reference implementation.
//
// Run: node experiments/diagnostics.mjs

import { sprout } from "./lib/sprout.mjs";
import {
    VERBS, OBJECTS, VAGUE,
    HELD_OUT_VERBS, HELD_OUT_OBJECTS, HELD_OUT_VAGUE,
    makeRandom, makeExamples, buildVocabulary, metricsFor, average, pct
} from "./lib/data.mjs";

const SEEDS = [1, 2, 3, 4, 5];

const FAMILIAR_POOLS = { verbs: VERBS, objects: OBJECTS, vague: VAGUE };
const HELD_OUT_POOLS = { verbs: HELD_OUT_VERBS, objects: HELD_OUT_OBJECTS, vague: HELD_OUT_VAGUE };

// A test set where each example's content words are drawn from the held-out
// pools with probability `oovRate`, and from the familiar pools otherwise.
function makeBlend(count, oovRate, random) {
    const examples = [];
    for (let i = 0; i < count; i++) {
        const pools = random() < oovRate ? HELD_OUT_POOLS : FAMILIAR_POOLS;
        examples.push(makeExamples(1, random, pools)[0]);
    }
    return examples;
}

// ---------------------------------------------------------------------------
// A) Does more optimisation rescue the current code?
// ---------------------------------------------------------------------------

console.log("=== A) Epoch sweep at 800 training examples (real subsystem trainer) ===");
console.log("Fixed 800 examples, 5 seeds. Before the word-vector fix this table was");
console.log("flat — the capacity ceiling. Now it shows how accuracy evolves with");
console.log("training length once every parameter can actually learn.");
console.log("");
console.log("epochs | seenAcc | unseenAcc | trainAcc");
console.log("-".repeat(44));

for (const epochs of [50, 200, 1000, 5000]) {
    const seen = [], unseen = [], train = [];

    for (const seed of SEEDS) {
        const trainSet = makeExamples(800, makeRandom(seed * 31 + 800), FAMILIAR_POOLS);
        const model = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed });

        sprout.train_model(trainSet, [], model, { epochs, learningRate: 0.2 });

        seen.push(metricsFor(model, makeExamples(600, makeRandom(999), FAMILIAR_POOLS)).accuracy);
        unseen.push(metricsFor(model, makeExamples(600, makeRandom(1000), HELD_OUT_POOLS)).accuracy);
        train.push(metricsFor(model, trainSet).accuracy);
    }

    console.log(
        [
            String(epochs).padStart(6),
            pct(average(seen)).padStart(7),
            pct(average(unseen)).padStart(9),
            pct(average(train)).padStart(8)
        ].join(" | ")
    );
}

// ---------------------------------------------------------------------------
// B) Accuracy versus vocabulary coverage
// ---------------------------------------------------------------------------

console.log("");
console.log("=== B) Held-out accuracy versus unknown-word rate ===");
console.log("Trained on 400 examples with the real subsystem trainer, 5 seeds.");
console.log("");
console.log("OOV rate | blendedAcc | (familiar=100% vs held-out=100% for reference)");
console.log("-".repeat(72));

for (const oovRate of [0, 0.25, 0.5, 0.75, 1.0]) {
    const blend = [];

    for (const seed of SEEDS) {
        const trainSet = makeExamples(400, makeRandom(seed * 977 + 400), FAMILIAR_POOLS);
        const model = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed });
        sprout.train_model(trainSet, [], model, { epochs: 200, learningRate: 0.2 });

        const mixed = makeBlend(600, oovRate, makeRandom(seed * 13 + 7));
        blend.push(metricsFor(model, mixed).accuracy);
    }

    const reference = oovRate === 0
        ? "100% familiar wording"
        : oovRate === 1
            ? "100% held-out wording"
            : `${pct(1 - oovRate)} familiar / ${pct(oovRate)} held-out`;

    console.log([pct(oovRate).padStart(8), pct(average(blend)).padStart(10), reference].join(" | "));
}
