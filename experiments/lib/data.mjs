// Synthetic task-description data with an explicit slop definition.
//
// Genuine = a concrete action on a concrete object ("buy milk", "renew the
//           passport").
// Slop    = keyboard mash ("asdfghjkl") or vague filler ("do something",
//           "fix stuff").
//
// Held-out word pools let us build test sets whose content words never
// appear in training, which is the hard case for a word-vector model.

import { sprout } from "./sprout.mjs";

export const VERBS = [
    "buy", "finish", "call", "email", "fix", "book", "replace", "water",
    "clean", "collect", "return", "renew", "pay", "submit", "print", "pack",
    "post", "mend", "wash", "iron", "charge", "update", "cancel", "confirm",
    "download", "install", "review", "file", "sort", "label", "deliver", "hire"
];

export const OBJECTS = [
    "milk", "homework", "dentist", "invoices", "boiler", "passport", "tyres",
    "curtains", "prescription", "taxes", "laundry", "dishes", "floors",
    "report", "slides", "form", "tickets", "appointment", "delivery", "keys",
    "glasses", "laptop", "phone", "bill", "garage", "bike", "shelf", "fence",
    "gutters", "carpet", "sofa", "windows", "bins", "garden", "printer"
];

export const VAGUE = [
    "do something", "stuff", "things", "misc", "tbd", "tba", "whatever",
    "blah", "idk", "not sure", "something", "anything", "handle it",
    "sort it out", "make it better", "improve things", "various", "etc",
    "todo", "fix stuff", "do the needful", "general tasks", "things and stuff",
    "look into it", "deal with it", "some stuff", "a few things", "misc tasks",
    "do the thing", "asdf", "qwerty", "zzz", "lorem ipsum", "placeholder",
    "to be decided", "who knows", "idk yet", "nothing", "dunno", "somehow"
];

export const HELD_OUT_VERBS = [
    "defrost", "haggle", "sand", "polish", "prune", "thaw", "wax",
    "sweep", "hoover", "unclog", "rehang", "descale", "reheel", "sharpen", "rinse"
];

export const HELD_OUT_OBJECTS = [
    "freezer", "lease", "wardrobe", "decking", "silverware", "hedge", "turkey",
    "surfboard", "chimney", "attic", "plughole", "kettle",
    "boots", "chisels", "birdfeeder"
];

export const HELD_OUT_VAGUE = [
    "assorted bits", "odds and ends", "the usual", "miscellany", "bric a brac",
    "sundries", "plain stuff", "general stuff", "bits and bobs", "thingamajig",
    "whatsit", "hoo hah", "jiggery pokery", "malarkey", "gubbins"
];

export function makeRandom(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function pick(array, random) {
    return array[Math.floor(random() * array.length)];
}

const MASH_LETTERS = "asdfghjklqwertyuiopzxcvbnm";

function gibberish(random) {
    let word = "";
    if (random() < 0.25) {
        const letter = MASH_LETTERS[Math.floor(random() * MASH_LETTERS.length)];
        return letter.repeat(2 + Math.floor(random() * 6));
    }
    const length = 3 + Math.floor(random() * 7);
    for (let i = 0; i < length; i++) {
        word += MASH_LETTERS[Math.floor(random() * MASH_LETTERS.length)];
    }
    return word;
}

function genuineTask(random, verbs, objects) {
    const verb = pick(verbs, random);
    const object = pick(objects, random);
    return random() < 0.5 ? `${verb} ${object}` : `${verb} the ${object}`;
}

function slopTask(random, vaguePool) {
    return random() < 0.4 ? gibberish(random) : pick(vaguePool, random);
}

export function makeExamples(count, random, pools) {
    const examples = [];
    for (let i = 0; i < count; i++) {
        if (i % 2 === 0) {
            examples.push({ text: genuineTask(random, pools.verbs, pools.objects), label: "genuine" });
        } else {
            examples.push({ text: slopTask(random, pools.vague), label: "slop" });
        }
    }
    return examples;
}

export function balanced(count, random, pools) {
    const all = makeExamples(count * 2, random, pools);
    const genuine = all.filter(e => e.label === "genuine").slice(0, count);
    const slop = all.filter(e => e.label === "slop").slice(0, count);
    return [...genuine, ...slop];
}

// Vocabulary = every token the training set actually used, plus <UNK>.
// The builder itself lives in subsystems/ so the browser app and the
// command-line trainer build vocabularies the same way.
export { buildVocabulary } from "./sprout.mjs";

export function metricsFor(model, testSet) {
    const result = sprout.evaluate_model(model, testSet);
    const slopTotal = result.truePositive + result.falseNegative;
    const genuineTotal = result.trueNegative + result.falsePositive;
    return {
        accuracy: result.accuracy,
        slopRecall: slopTotal === 0 ? 0 : result.truePositive / slopTotal,
        falsePositiveRate: genuineTotal === 0 ? 0 : result.falsePositive / genuineTotal
    };
}

export function average(values) {
    return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}

export function standardDeviation(values) {
    const mean = average(values);
    return Math.sqrt(average(values.map(v => (v - mean) ** 2)));
}

export function pct(value) {
    return `${(value * 100).toFixed(1)}%`;
}
