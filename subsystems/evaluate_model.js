// Measure how well a trained model performs on a held-out test set.
//
// Returns counts and rates for true positives, false positives, true
// negatives, and false negatives. Together they describe how often the
// model correctly identifies slop and, importantly, how often it wrongly
// flags genuine text.
//
// `threshold` defaults to 0.5 — anything at or above is read as "slop".

function evaluate_model(model, testSet, options) {
    if (!model || !Array.isArray(model.weights)) {
        throw new TypeError("evaluate_model: model is missing weights");
    }

    if (!Array.isArray(testSet)) {
        throw new TypeError("evaluate_model: testSet must be an array");
    }

    const settings = options || {};
    const threshold = typeof settings.threshold === "number" ? settings.threshold : 0.5;

    let truePositive = 0;
    let falsePositive = 0;
    let trueNegative = 0;
    let falseNegative = 0;
    let totalLoss = 0;

    for (const example of testSet) {
        const { score } = make_prediction_for_training(example.text, model);
        const predictedLabel = score >= threshold ? "slop" : "genuine";

        if (predictedLabel === "slop" && example.label === "slop") truePositive++;
        else if (predictedLabel === "slop" && example.label === "genuine") falsePositive++;
        else if (predictedLabel === "genuine" && example.label === "genuine") trueNegative++;
        else if (predictedLabel === "genuine" && example.label === "slop") falseNegative++;

        const { loss } = calculate_loss(score, example.label);
        totalLoss += loss;
    }

    const total = testSet.length;
    const safeDivisor = (a, b) => (b === 0 ? 0 : a / b);

    return {
        count: total,
        truePositive,
        falsePositive,
        trueNegative,
        falseNegative,
        accuracy: safeDivisor(truePositive + trueNegative, total),
        precision: safeDivisor(truePositive, truePositive + falsePositive),
        recall: safeDivisor(truePositive, truePositive + falseNegative),
        averageLoss: total > 0 ? totalLoss / total : 0
    };
}