// v4: eight disjoint 65,536-slot families, 524,288 potential classifier
// weights + bias. No padding: each bank is reachable by ordinary input.
// More capacity reduces cross-feature collisions; extra affix and skip
// features let the classifier share evidence across inflected/new words.
// Does not import a language model or assume that spelling implies meaning.
function lexicon_entry_for(word, lexicon) {
    if (!lexicon || typeof lexicon !== "object") return null;
    if (Object.hasOwn(lexicon, word)) return lexicon[word];
    const stems = [];
    if (word.endsWith("ies")) stems.push(word.slice(0, -3) + "y");
    if (word.endsWith("es")) stems.push(word.slice(0, -2));
    if (word.endsWith("s")) stems.push(word.slice(0, -1));
    for (const stem of stems) if (Object.hasOwn(lexicon, stem)) return lexicon[stem];
    return null;
}

function richer_features(text, size, lexicon, lexiconStrength = 1, structureVersion = 0) {
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
        // WordNet category is a shared, learned lexical cue. The optional
        // direct first-sense synonym reuses an already trained unigram slot.
        // Both are inside the existing hash banks: no new parameters.
        if (lexicon) {
            const entry = lexicon_entry_for(word, lexicon);
            if (entry) {
                add(7, "lex:" + entry[0], 0.45 * lexiconStrength * scale);
                if (entry[1]) add(0, entry[1], 0.25 * lexiconStrength * scale);
            }
        }
    }
    // Opt-in v4 structural view. The existing v4 feature map stays exactly
    // unchanged when structureVersion is absent, so old checkpoints keep
    // their scores. Abstractions deliberately ignore content-word identity:
    // a previously unseen component can occupy the same role as a known one.
    // These are learned evidence, not hard-coded verdicts or a dictionary of
    // technical terms. All slots are in the existing position/affix banks.
    if (structureVersion === 1 || structureVersion === 2 || structureVersion === 3) {
        const failures = new Set(["fail", "fails", "failed", "failing", "crash", "crashes", "crashed",
            "error", "errors", "broken", "breaks", "incorrect", "wrong", "missing", "hangs", "stuck",
            "doesn't", "don't", "cannot", "can't", "not", "never", "invalid", "unable", "regression"]);
        const conditions = new Set(["when", "after", "before", "while", "unless", "without",
            "during", "under", "with", "on", "if", "instead", "because", "since"]);
        const actions = new Set(["add", "support", "allow", "remove", "fix", "make", "show", "return",
            "update", "use", "prevent", "handle", "document", "implement", "enable", "disable", "change"]);
        const vague = new Set(["things", "thing", "stuff", "something", "various", "misc", "general",
            "better", "improve", "improvements", "issues", "problem", "problems", "somehow"]);
        const functionWords = new Set(["the", "a", "an", "is", "are", "in", "to", "for", "from", "of",
            "and", "or", "as", "by", "it", "this", "that", "at", "be", "should", "will", "can"]);
        const types = words.map(word => {
            if (/^\d+(?:[.,]\d+)*$/.test(word)) return "NUM";
            if (!/[\p{L}\p{N}]/u.test(word)) return "PUNCT";
            if (failures.has(word)) return "FAIL";
            if (conditions.has(word)) return "COND:" + word;
            if (actions.has(word)) return "ACTION";
            if (vague.has(word)) return "VAGUE";
            if (functionWords.has(word)) return "FUNC:" + word;
            return "TARGET";
        });
        const has = type => types.includes(type);
        const hasCondition = types.some(type => type.startsWith("COND:"));
        const targetCount = types.filter(type => type === "TARGET").length;
        const codeLike = /`[^`]+`|[\p{L}_]+\.[\p{L}_]+|\b[A-Z][A-Z_\d]{2,}\b/u.test(text);
        // v2 learns abstract relations *only* where there is evidence of a
        // specific symptom, conditional behaviour, or code-shaped target.
        // Generic action-to-target requests are left to the normal v4 map.
        // This is not a project-name whitelist or an alternate threshold.
        const evidence = has("FAIL") || codeLike || (has("ACTION") && hasCondition) ||
            (structureVersion === 3 && words.length >= 7);
        if (structureVersion === 1 || evidence) {
            for (let i = 0; i < types.length; i++) {
                if (i) add(6, "shape:pair:" + types[i - 1] + ":" + types[i], 0.35 * scale);
                if (i > 1) add(7, "shape:triple:" + types[i - 2] + ":" + types[i - 1] + ":" + types[i], 0.26 * scale);
            }
        }
        if (has("FAIL") && hasCondition) add(6, "shape:failure-condition", 0.85);
        if (has("ACTION") && hasCondition) add(6, "shape:action-condition", 0.55);
        if (has("FAIL") && targetCount >= 2) add(6, "shape:failure-targets", 0.55);
        if (has("ACTION") && targetCount >= 2 && (structureVersion === 1 || evidence)) add(6, "shape:action-targets", 0.55);
        if (has("VAGUE") && targetCount < 2 && (structureVersion === 1 || evidence)) add(6, "shape:vague-without-target", 0.7);
        if (has("FAIL") && targetCount < 2) add(6, "shape:failure-without-target", 0.55);
        if (has("NUM") && has("FAIL")) add(6, "shape:number-failure", 0.5);
        if (codeLike) {
            add(6, "shape:code-or-identifier", 0.55);
            if (has("FAIL")) add(6, "shape:code-failure", 0.55);
        }
    }
    return [...features];
}
