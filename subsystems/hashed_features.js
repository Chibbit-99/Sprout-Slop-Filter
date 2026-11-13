// v2: sparse word, word-pair and character n-gram features. Unlike a word
// lookup table, this has a fixed memory budget and produces features for
// words never seen in training. No network calls or downloaded vocabulary.
// All feature extraction is shared by training, evaluation and inference.
function hashed_feature_index(feature, size) {
    // FNV-1a on UTF-16 code units: deterministic in Node and browsers.
    let hash = 2166136261;
    for (let i = 0; i < feature.length; i++) {
        hash = Math.imul(hash ^ feature.charCodeAt(i), 16777619);
    }
    return (hash >>> 0) % size;
}

function hashed_features(text, size) {
    const words = tokenise(normalise_text(text)).slice(0, 128);
    const features = new Map();
    const add = (name, value) => {
        const id = hashed_feature_index(name, size);
        features.set(id, (features.get(id) || 0) + value);
    };
    const normalizer = Math.sqrt(Math.max(1, words.length));

    for (let i = 0; i < words.length; i++) {
        const word = words[i].slice(0, 64);
        add("w:" + word, 1.5 / normalizer);
        if (i === 0) add("first:" + word, 0.8);
        if (i > 0) add("pair:" + words[i - 1].slice(0, 64) + " " + word, 1 / normalizer);
        // Boundary markers distinguish prefixes/suffixes. Only word tokens
        // get n-grams; punctuation already has word/pair features.
        if (!/[\p{L}\p{N}]/u.test(word)) continue;
        const padded = "^" + word + "$";
        const grams = [];
        for (let n = 3; n <= 5; n++) {
            for (let j = 0; j + n <= padded.length; j++) {
                grams.push("c:" + padded.slice(j, j + n));
            }
        }
        const gramWeight = 0.8 / (normalizer * Math.sqrt(Math.max(1, grams.length)));
        for (const gram of grams) add(gram, gramWeight);
    }
    return [...features]; // [hash index, feature value] pairs
}

function hashed_predict(text, model) {
    const representation = model.version === 4 ? richer_features(text, model.featureSize)
        : model.version === 3 ? structured_features(text, model.featureSize)
        : hashed_features(text, model.featureSize);
    let logit = model.bias;
    for (const [id, value] of representation) logit += model.weights[id] * value;
    return { score: sigmoid(logit), representation, tokenIndices: [] };
}

function hashed_update(model, errorInformation, learningRate, representation) {
    const gradient = errorInformation.gradient;
    if (!model.accumulators) model.accumulators = new Array(model.featureSize).fill(0);
    for (const [id, value] of representation) {
        // Per-coordinate AdaGrad, with a mild L2 penalty. The accumulator
        // belongs to the trainer, not the saved inference model.
        const g = gradient * value;
        model.accumulators[id] += g * g;
        model.weights[id] -= learningRate * (g + 0.0001 * model.weights[id]) /
            Math.sqrt(1 + model.accumulators[id]);
    }
    model.bias -= learningRate * gradient * 0.25;
}
