// Restore a previously saved model.
//
// In a browser, `load_model` fetches JSON from `path`. A `storage` option
// can be supplied to override the source (for example, to read from
// localStorage in tests).
//
// The returned object has the same shape as the model produced by
// `initialise_model`, ready to be passed to `classify_text` or
// `evaluate_model`.
//
// Returns: { success: true, model } on success, or
//          { success: false, error: string } on failure.

function load_model(path, options) {
    const settings = options || {};

    if (typeof settings.storage === "function") {
        try {
            const raw = settings.storage(path);
            return parsePayload(raw);
        } catch (error) {
            return { success: false, error: "load_model: storage hook threw: " + error.message };
        }
    }

    if (typeof fetch !== "function") {
        return { success: false, error: "load_model: fetch is not available in this environment" };
    }

    return fetch(path).then(response => {
        if (!response.ok) {
            return {
                success: false,
                error: `load_model: failed to load ${path} (status ${response.status})`
            };
        }

        return response.json().then(json => parsePayload(JSON.stringify(json)));
    });
}

function parsePayload(raw) {
    if (typeof raw !== "string" || raw.length === 0) {
        return { success: false, error: "load_model: empty payload" };
    }

    let parsed;

    try {
        parsed = JSON.parse(raw);
    } catch (error) {
        return { success: false, error: "load_model: payload is not valid JSON" };
    }

    if (parsed && (parsed.version === 2 || parsed.version === 3 || parsed.version === 4)) {
        const expectedEncoder = parsed.version === 4 ? "richer-subword-v4" :
            parsed.version === 3 ? "structured-subword-v3" : "hashed-subword-v2";
        if (parsed.options?.encoder !== expectedEncoder ||
            (parsed.version === 3 && parsed.featureSize !== 131072) ||
            (parsed.version === 4 && parsed.featureSize !== 524288) ||
            !Number.isInteger(parsed.featureSize) || parsed.featureSize < 1024 || parsed.featureSize > 524288 ||
            !Array.isArray(parsed.weights) || !Array.isArray(parsed.vocabulary) ||
            !parsed.vocabulary.every(word => typeof word === "string") || !Number.isFinite(parsed.bias)) {
            return { success: false, error: "load_model: invalid hashed model" };
        }
        const weights = new Array(parsed.featureSize).fill(0);
        const used = new Set();
        for (const pair of parsed.weights) {
            if (!Array.isArray(pair) || pair.length !== 2 ||
                !Number.isInteger(pair[0]) || pair[0] < 0 || pair[0] >= weights.length ||
                !Number.isFinite(pair[1]) || used.has(pair[0])) {
                return { success: false, error: "load_model: invalid hashed weights" };
            }
            used.add(pair[0]);
            weights[pair[0]] = pair[1];
        }
        return { success: true, model: { version: parsed.version, featureSize: parsed.featureSize,
            vocabulary: parsed.vocabulary, weights, bias: parsed.bias, options: parsed.options } };
    }
    if (parsed && parsed.version !== undefined && parsed.version !== 1) {
        return { success: false, error: "load_model: unsupported version " + parsed.version };
    }
    if (!parsed || !Array.isArray(parsed.vocabulary) || !Array.isArray(parsed.wordVectors)) {
        return { success: false, error: "load_model: payload is missing vocabulary or wordVectors" };
    }

    if (!Array.isArray(parsed.weights)) {
        return { success: false, error: "load_model: payload is missing weights" };
    }

    if (parsed.wordVectors.length !== parsed.vocabulary.length) {
        return { success: false, error: "load_model: vocabulary and wordVectors length mismatch" };
    }

    return {
        success: true,
        model: {
            vocabulary: parsed.vocabulary,
            wordVectors: parsed.wordVectors,
            weights: parsed.weights,
            bias: typeof parsed.bias === "number" ? parsed.bias : 0,
            options: parsed.options || {}
        }
    };
}