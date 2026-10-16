// Measure how wrong a prediction was, returning both the loss and the
// gradient information the optimiser needs.
//
// The first version uses binary cross-entropy, the standard loss for
// binary classification:
//   loss = -[ y · log(p) + (1 - y) · log(1 - p) ]
// The label `y` is 1 for slop, 0 for genuine.
//
// For logistic regression the gradient with respect to the pre-sigmoid
// logit is simply (p - y). Returning the gradient alongside the loss
// keeps `update_parameters` simple.

function clampProbability(p) {
    // Avoid log(0) which would produce -Infinity.
    if (p < 1e-12) return 1e-12;
    if (p > 1 - 1e-12) return 1 - 1e-12;
    return p;
}

function calculate_loss(score, correctLabel) {
    if (typeof score !== "number" || Number.isNaN(score)) {
        throw new TypeError("calculate_loss: score must be a number");
    }

    const label = correctLabel === "slop" ? 1 : 0;
    const probability = clampProbability(score);

    const loss = -(label * Math.log(probability) + (1 - label) * Math.log(1 - probability));
    const gradient = score - label;

    return { loss, gradient };
}