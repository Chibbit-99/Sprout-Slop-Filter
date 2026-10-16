// PROPOSED PATCH — reference implementation only, does not touch subsystems/.
//
// `subsystems/update_parameters.js` documents that "word vectors get a small
// share of the error too, weighted by how often they participated in the
// representation", but the function never receives `wordVectors` and never
// writes to them. `train_model.js` passes only the averaged `representation`,
// so the only parameters that ever change are the 8 classifier weights and
// the bias.
//
// Consequence: every word vector stays at its initial random value forever.
// The model can only learn a single linear boundary in a fixed random 8-D
// space, so held-out accuracy sits near chance-plus-a-bit no matter how much
// training data you supply.
//
// This module implements the update described in those existing comments:
//
//   r   = (1/n) * sum(v_t)             averaged word vectors
//   p   = sigmoid(w . r + b)
//   g   = p - y                        gradient of binary cross-entropy
//                                      w.r.t. the logit
//   w   -= lr * g * r
//   b   -= lr * g
//   v_t -= lr * g * w / n              for every token t in the example

export function trainWithWordVectors(trainSet, validationSet, model, sprout, options) {
    const settings = options || {};
    const epochs = typeof settings.epochs === "number" ? settings.epochs : 50;
    const learningRate = typeof settings.learningRate === "number" ? settings.learningRate : 0.1;
    const seed = typeof settings.seed === "number" ? settings.seed : 1;

    const indexOf = new Map();
    for (let i = 0; i < model.vocabulary.length; i++) indexOf.set(model.vocabulary[i], i);

    let a = (seed * 2654435761) >>> 0;
    const random = () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    // Pre-tokenise once; tokenisation does not change during training.
    const unknownIndex = indexOf.get("<UNK>") ?? -1;
    const prepared = trainSet.map(example => {
        const tokens = sprout.tokenise(sprout.normalise_text(example.text));
        const indices = [];
        for (const token of tokens) {
            const index = indexOf.has(token) ? indexOf.get(token) : unknownIndex;
            if (index !== -1) indices.push(index);
        }
        return { indices, label: example.label };
    });

    const history = [];
    const order = prepared.map((_, i) => i);

    for (let epoch = 0; epoch < epochs; epoch++) {
        // Shuffle each epoch — plain SGD on ordered data otherwise oscillates.
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
        }

        let epochLoss = 0;

        for (const index of order) {
            const { indices, label } = prepared[index];
            if (indices.length === 0) continue;
            epochLoss += trainStep(indices, label, model, sprout, learningRate);
        }

        history.push({
            epoch,
            train: { averageLoss: trainSet.length > 0 ? epochLoss / trainSet.length : 0 },
            validation: evaluateModel(model, sprout, validationSet)
        });
    }

    model.history = history;
    return model;
}

// One forward pass + one SGD step over pre-tokenised vocabulary indices.
// Returns the loss for this example.
function trainStep(indices, label, model, sprout, learningRate) {
    const dimension = model.weights.length;
    const count = indices.length;

    const representation = new Array(dimension).fill(0);
    for (const index of indices) {
        const vector = model.wordVectors[index];
        for (let i = 0; i < dimension; i++) representation[i] += vector[i];
    }
    for (let i = 0; i < dimension; i++) representation[i] /= count;

    const score = sprout.predict_slop(representation, { weights: model.weights, bias: model.bias });
    const { loss, gradient } = sprout.calculate_loss(score, label);

    // Classifier weights and bias.
    for (let i = 0; i < dimension; i++) {
        model.weights[i] -= learningRate * gradient * representation[i];
    }
    model.bias -= learningRate * gradient;

    // Word vectors: the shared error, divided across the tokens that
    // contributed to this representation.
    const shared = learningRate * gradient / count;
    for (const index of indices) {
        const vector = model.wordVectors[index];
        for (let i = 0; i < dimension; i++) {
            vector[i] -= shared * model.weights[i];
        }
    }

    return loss;
}

function evaluateModel(model, sprout, set) {
    if (set.length === 0) return { accuracy: 0, averageLoss: 0, count: 0 };
    let correct = 0;
    let totalLoss = 0;
    for (const example of set) {
        const { score } = sprout.make_prediction_for_training(example.text, model);
        if ((score >= 0.5 ? "slop" : "genuine") === example.label) correct++;
        totalLoss += sprout.calculate_loss(score, example.label).loss;
    }
    return { accuracy: correct / set.length, averageLoss: totalLoss / set.length, count: set.length };
}
