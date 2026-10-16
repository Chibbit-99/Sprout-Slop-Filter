// Serialise a model so it can be restored later without retraining.
//
// The vocabulary, word vectors, classifier weights, bias, and options are
// all written to the destination. In a browser, `path` is treated as a
// download trigger (anchor + object URL); in Node-style environments, the
// caller can pass a custom `storage` hook to redirect the JSON payload
// elsewhere.
//
// Returns: { success: true, payload } on success, or
//          { success: false, error: string } on failure.

function save_model(model, path, options) {
    if (!model || !Array.isArray(model.vocabulary) || !Array.isArray(model.wordVectors)) {
        return { success: false, error: "save_model: model is missing vocabulary or wordVectors" };
    }

    if (!Array.isArray(model.weights)) {
        return { success: false, error: "save_model: model is missing weights" };
    }

    const payload = JSON.stringify({
        version: 1,
        vocabulary: model.vocabulary,
        wordVectors: model.wordVectors,
        weights: model.weights,
        bias: typeof model.bias === "number" ? model.bias : 0,
        options: model.options || {}
    });

    const settings = options || {};
    if (typeof settings.storage === "function") {
        try {
            settings.storage(payload, path);
            return { success: true, payload };
        } catch (error) {
            return { success: false, error: "save_model: storage hook threw: " + error.message };
        }
    }

    if (typeof document === "undefined" || typeof path !== "string" || path.length === 0) {
        return { success: false, error: "save_model: no storage mechanism available" };
    }

    try {
        const blob = new Blob([payload], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = path;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
        return { success: true, payload };
    } catch (error) {
        return { success: false, error: "save_model: failed to save: " + error.message };
    }
}