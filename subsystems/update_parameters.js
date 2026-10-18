// Apply one gradient descent step to the model's parameters.
//
// Given the gradient information from `calculate_loss`, the encoded
// representation that produced the prediction, and the vocabulary indices
// of the tokens that were averaged into that representation, adjust every
// parameter a little in the direction that reduces the error.
//
// The model is a logistic regression on the averaged word vector:
//
//   r = (1/n) * sum(v_t)        averaged word vectors of the n tokens
//   p = sigmoid(w . r + b)
//
// The gradients (with respect to the logit) are:
//
//   dL/dw_i = gradient * r_i
//   dL/db   = gradient
//   dL/dr_i = gradient * w_i
//
// Word vectors get a small share of the error too, weighted by how often
// they participated in the representation. The contribution per token is
// `gradient * w / n` where `n` is the number of tokens that contributed
// to the average (non-zero vectors).
//
// `tokenIndices` carries that participation information: one vocabulary
// index per token that contributed to the representation, exactly as
// returned by `make_prediction_for_training`. The word-vector step is
// computed from the weights of the forward pass, so it is derived before
// the classifier weights are modified.
//
// `tokenIndices` is optional for backwards compatibility: when it is
// absent (or the model carries no `wordVectors`), only the classifier
// weights and the bias are updated and the word vectors stay frozen at
// their initial values — the pre-fix behaviour documented in
// experiments/FINDINGS.md section 1.

function update_parameters(modelParameters, errorInformation, learningRate, representation, tokenIndices) {
    if (!modelParameters || !Array.isArray(modelParameters.weights)) {
        throw new TypeError("update_parameters: modelParameters is invalid");
    }

    if (!errorInformation || typeof errorInformation.gradient !== "number") {
        throw new TypeError("update_parameters: errorInformation must contain a numeric gradient");
    }

    const rate = typeof learningRate === "number" ? learningRate : 0.1;

    if (!Array.isArray(representation) || representation.length !== modelParameters.weights.length) {
        throw new Error("update_parameters: representation length must match weights length");
    }

    const { gradient } = errorInformation;

    // Work out the word-vector step first: it uses the weights as they were
    // during the forward pass, and those are about to change.
    const trainsWordVectors = Array.isArray(tokenIndices) && tokenIndices.length > 0
        && Array.isArray(modelParameters.wordVectors);

    let vectorStep = null;
    if (trainsWordVectors) {
        const shared = rate * gradient / tokenIndices.length;
        vectorStep = modelParameters.weights.map(weight => shared * weight);
    }

    // Update classifier weights and bias.
    for (let i = 0; i < modelParameters.weights.length; i++) {
        modelParameters.weights[i] -= rate * gradient * representation[i];
    }

    modelParameters.bias -= rate * gradient;

    // Update every word vector that participated in the representation.
    // A token appearing twice contributes twice, matching the forward pass
    // where it was averaged twice.
    if (vectorStep) {
        for (const index of tokenIndices) {
            const vector = modelParameters.wordVectors[index];
            if (!Array.isArray(vector)) continue;

            for (let i = 0; i < vectorStep.length && i < vector.length; i++) {
                vector[i] -= vectorStep[i];
            }
        }
    }

    return modelParameters;
}
