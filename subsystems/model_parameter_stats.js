// Count actual trainable numeric scalar slots in the model, excluding
// vocabulary strings, optimiser accumulators, history and JSON metadata.
// For zero-initialised hashed models, nonZeroWeights shows how much of the
// allocated capacity training has populated. It is NOT the parameter count.
function model_parameter_stats(model) {
    if (!model || !Array.isArray(model.weights) || !Number.isFinite(model.bias)) {
        throw new TypeError("model_parameter_stats: invalid model");
    }
    const hashed = model.version === 2 || model.version === 3 || model.version === 4;
    if (hashed && (model.weights.length !== model.featureSize ||
        (model.version === 3 && model.featureSize !== 131072) ||
        (model.version === 4 && model.featureSize !== 524288))) {
        throw new RangeError("model_parameter_stats: feature count mismatch");
    }
    let embeddings = 0;
    let nonZeroEmbeddings = 0;
    if (!hashed && Array.isArray(model.wordVectors)) {
        for (const vector of model.wordVectors) {
            if (!Array.isArray(vector)) throw new TypeError("model_parameter_stats: invalid word vector");
            embeddings += vector.length;
            nonZeroEmbeddings += vector.filter(value => value !== 0).length;
        }
    }
    const nonZeroClassifier = model.weights.filter(value => value !== 0).length;
    const families = {};
    if (model.version === 3 || model.version === 4) {
        const bank = model.version === 4 ? 65536 : 32768;
        const names = model.version === 4
            ? ["word", "char3", "char4", "char5", "adjacentPair", "skipPair", "position", "affix"]
            : ["word", "character", "adjacentPair", "positionAndSkip"];
        for (const [i, name] of names.entries()) {
            const slice = model.weights.slice(i * bank, (i + 1) * bank);
            families[name] = { slots: slice.length, nonZero: slice.filter(value => value !== 0).length };
        }
    }
    return {
        version: model.version || 1,
        trainable: model.weights.length + embeddings + 1, // bias
        classifierWeights: model.weights.length,
        embeddingWeights: embeddings,
        biasParameters: 1,
        nonZeroWeights: nonZeroClassifier + nonZeroEmbeddings,
        nonZeroIncludingBias: nonZeroClassifier + nonZeroEmbeddings + Number(model.bias !== 0),
        families
    };
}
