// Train the model by repeating prediction, loss calculation, and parameter
// updates over the training set for a number of epochs.
//
// Validation performance is reported each epoch (without influencing the
// model) so the caller can decide when to stop. The test set is never
// touched here.
//
// Expected inputs:
//   trainSet      — array of { text, label } examples
//   validationSet — array of { text, label } examples (may be empty)
//   model         — model object from `initialise_model` (will be mutated)
//   options       — { epochs?: number, learningRate?: number,
//                     onEpoch?: function(epoch, metrics) }
//
// Returns: the trained model, plus a `history` record describing how loss
// and accuracy evolved across epochs.

function defaultLoss(model, example) {
    const { score, representation, tokenIndices } = make_prediction_for_training(example.text, model);
    const errorInfo = calculate_loss(score, example.label);
    return { errorInfo, representation, tokenIndices };
}

function evaluateOnSet(model, set) {
    if (set.length === 0) {
        return { accuracy: 0, averageLoss: 0, count: 0 };
    }

    let correct = 0;
    let totalLoss = 0;

    for (const example of set) {
        const { score } = make_prediction_for_training(example.text, model);
        const predictedLabel = score >= 0.5 ? "slop" : "genuine";

        if (predictedLabel === example.label) {
            correct++;
        }

        const { loss } = calculate_loss(score, example.label);
        totalLoss += loss;
    }

    return {
        accuracy: correct / set.length,
        averageLoss: totalLoss / set.length,
        count: set.length
    };
}

function train_model(trainSet, validationSet, model, options) {
    if (!Array.isArray(trainSet)) {
        throw new TypeError("train_model: trainSet must be an array");
    }

    if (!Array.isArray(validationSet)) {
        throw new TypeError("train_model: validationSet must be an array");
    }

    if (!model || !Array.isArray(model.weights)) {
        throw new TypeError("train_model: model is missing weights");
    }

    const settings = options || {};
    const epochs = typeof settings.epochs === "number" ? settings.epochs : (model.options && model.options.epochs) || 50;
    const learningRate = typeof settings.learningRate === "number"
        ? settings.learningRate
        : (model.options && model.options.learningRate) || 0.1;
    const onEpoch = typeof settings.onEpoch === "function" ? settings.onEpoch : null;

    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
        let epochLoss = 0;

        for (const example of trainSet) {
            const { errorInfo, representation, tokenIndices } = defaultLoss(model, example);
            // Passing `tokenIndices` lets `update_parameters` train the word
            // vectors of every token that contributed to this prediction,
            // not just the classifier weights (see FINDINGS.md section 1).
            update_parameters(model, errorInfo, learningRate, representation, tokenIndices);
            epochLoss += errorInfo.loss;
        }

        const trainMetrics = {
            accuracy: 0,
            averageLoss: trainSet.length > 0 ? epochLoss / trainSet.length : 0,
            count: trainSet.length
        };

        const validationMetrics = evaluateOnSet(model, validationSet);
        const record = {
            epoch,
            train: trainMetrics,
            validation: validationMetrics
        };

        history.push(record);

        if (onEpoch) {
            onEpoch(epoch, record);
        }
    }

    model.history = history;
    return model;
}