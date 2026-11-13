// v4: eight disjoint 65,536-slot families, 524,288 potential classifier
// weights + bias. No padding: each bank is reachable by ordinary input.
// More capacity reduces cross-feature collisions; extra affix and skip
// features let the classifier share evidence across inflected/new words.
// Does not import a language model or assume that spelling implies meaning.
function richer_features(text, size) {
    if (size !== 524288) throw new RangeError("richer_features: expected 524288 features");
    const words = tokenise(normalise_text(text)).slice(0, 128).map(word => word.slice(0, 64));
    const features = new Map();
    const bank = size / 8;
    const add = (family, key, value) => {
        const id = family * bank + hashed_feature_index(key, bank);
        features.set(id, (features.get(id) || 0) + value);
    };
    const scale = 1 / Math.sqrt(Math.max(1, words.length));
    for (let i = 0; i < words.length; i++) {
        const word = words[i];
        add(0, word, 1.5 * scale);
        if (i > 0) add(4, words[i - 1] + " " + word, scale);
        if (i > 1) add(5, words[i - 2] + " " + word, 0.6 * scale);
        if (i === 0) add(6, "first:" + word, 0.8);
        if (i === words.length - 1) add(6, "last:" + word, 0.35 * scale);
        if (!/[\p{L}\p{N}]/u.test(word)) continue;
        const padded = "^" + word + "$";
        let gramCount = 0;
        for (let n = 3; n <= 5; n++) gramCount += Math.max(0, padded.length - n + 1);
        const gramScale = 0.8 * scale / Math.sqrt(Math.max(1, gramCount));
        for (let n = 3; n <= 5; n++) {
            for (let j = 0; j + n <= padded.length; j++) {
                add(n - 2, padded.slice(j, j + n), gramScale);
            }
        }
        const affixes = [];
        for (let n = 2; n <= 5 && n <= word.length; n++) {
            affixes.push("prefix:" + word.slice(0, n), "suffix:" + word.slice(-n));
        }
        const affixScale = 0.35 * scale / Math.sqrt(Math.max(1, affixes.length));
        for (const affix of affixes) add(7, affix, affixScale);
    }
    return [...features];
}
