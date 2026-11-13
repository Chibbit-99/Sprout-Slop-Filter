// v3: four disjoint 32,768-coordinate namespaces (131,072 trainable weights).
// Word, subword, adjacent and position/skip features cannot overwrite each
// other's weight through hash collisions. Each coordinate is a trainable
// classifier coefficient, not a dummy padding entry; only coordinates touched
// by the training corpus acquire nonzero values. Unknown words still get
// character features, and local context contributes ordered pairs.
//
// Keep this a classic script: the browser and Node's vm load the same code.
function structured_features(text, size) {
    if (size !== 131072) throw new RangeError("structured_features: expected 131072 features");
    const words = tokenise(normalise_text(text)).slice(0, 128).map(word => word.slice(0, 64));
    const features = new Map();
    const bucketSize = size / 4;
    const add = (namespace, key, value) => {
        const id = namespace * bucketSize + hashed_feature_index(key, bucketSize);
        features.set(id, (features.get(id) || 0) + value);
    };
    const scale = 1 / Math.sqrt(Math.max(1, words.length));

    for (let i = 0; i < words.length; i++) {
        const word = words[i];
        add(0, word, 1.5 * scale); // word identity
        if (i > 0) add(2, words[i - 1] + " " + word, scale); // adjacency/order
        if (i === 0) add(3, "first:" + word, 0.8);
        if (i === words.length - 1) add(3, "last:" + word, 0.35 * scale);
        if (i > 1) add(3, "skip:" + words[i - 2] + " " + word, 0.6 * scale);

        if (!/[\p{L}\p{N}]/u.test(word)) continue;
        const padded = "^" + word + "$";
        const grams = [];
        for (let n = 3; n <= 5; n++) {
            for (let j = 0; j + n <= padded.length; j++) grams.push(padded.slice(j, j + n));
        }
        const gramScale = 0.8 * scale / Math.sqrt(Math.max(1, grams.length));
        for (const gram of grams) add(1, gram, gramScale); // unseen spellings
    }
    return [...features];
}
