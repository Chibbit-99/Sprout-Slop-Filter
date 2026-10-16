// Produce a slop score between 0 and 1 from a text representation.
//
// The first version uses logistic regression over the averaged word
// vector: the score is a sigmoid applied to (weights · representation +
// bias). A score closer to 0 means the model leans genuine; closer to 1
// means it leans slop. This is a model score, not a guarantee.
//
// Expected inputs:
//   textRepresentation — array of numbers (the encoded text)
//   modelParameters    — { weights: number[], bias: number }

function sigmoid(score) {
    // Numerically stable sigmoid. Clamp the input so we never overflow
    // when computing exp on huge magnitudes.
    if (score >= 0) {
        const z = Math.exp(-score);
        return 1 / (1 + z);
    }

    const z = Math.exp(score);
    return z / (1 + z);
}

function predict_slop(textRepresentation, modelParameters) {
    if (!Array.isArray(textRepresentation)) {
        throw new TypeError("predict_slop: textRepresentation must be an array");
    }

    if (!modelParameters || !Array.isArray(modelParameters.weights)) {
        throw new TypeError("predict_slop: modelParameters.weights must be an array");
    }

    const { weights, bias } = modelParameters;
    const safeBias = typeof bias === "number" ? bias : 0;

    if (weights.length !== textRepresentation.length) {
        throw new Error(
            "predict_slop: weights length (" + weights.length +
            ") does not match textRepresentation length (" + textRepresentation.length + ")"
        );
    }

    let score = safeBias;

    for (let i = 0; i < weights.length; i++) {
        const value = textRepresentation[i];
        if (typeof value !== "number") continue;

        score += weights[i] * value;
    }

    return sigmoid(score);
}