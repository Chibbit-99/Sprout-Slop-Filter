// Apply one gradient descent step to the model's parameters.
//
// Given the gradient information from `calculate_loss` and the encoded
// representation that produced the prediction, adjust every parameter a
// little in the direction that reduces the error.
//
// The model is a logistic regression on the averaged word vector:
//
//   p = sigmoid(w · r + b)
//
// The gradients (with respect to the logit) are:
//
//   dL/dw_i = gradient · r_i
//   dL/db   = gradient
//   dL/dr_i = gradient · w_i
//
// Word vectors get a small share of the error too, weighted by how often
// they participated in the representation. The contribution per token is
// `gradient · w / n` where `n` is the number of tokens that contributed
// to the average (non-zero vectors).

function update_parameters(modelParameters, errorInformation, learningRate, representation) {
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

    // Update classifier weights and bias.
    for (let i = 0; i < modelParameters.weights.length; i++) {
        modelParameters.weights[i] -= rate * gradient * representation[i];
    }

    modelParameters.bias -= rate * gradient;

    return modelParameters;
}