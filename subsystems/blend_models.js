// Blend two compatible linear classifiers in weight space. The inference
// architecture and parameter count stay unchanged: for a fixed feature map,
// this is exactly a convex combination of their logits, in ONE weight array.
// `referenceScale` converts a score-softened checkpoint back to the units
// of the target model (or from its temperature to another temperature).
// Never combine models trained for different label definitions.
function blend_models(target, reference, fraction, referenceScale = 1) {
    if (!target || !reference || target.version !== 4 || reference.version !== 4 ||
        target.featureSize !== reference.featureSize ||
        target.options?.encoder !== reference.options?.encoder ||
        !Array.isArray(target.weights) || !Array.isArray(reference.weights) ||
        target.weights.length !== target.featureSize || reference.weights.length !== reference.featureSize ||
        !Number.isFinite(target.bias) || !Number.isFinite(reference.bias) ||
        !Number.isFinite(fraction) || fraction < 0 || fraction > 1 ||
        !Number.isFinite(referenceScale) || referenceScale <= 0 ||
        target.options?.domain !== reference.options?.domain) {
        throw new TypeError("blend_models: requires compatible v4 models, a shared label policy and a fraction in [0,1]");
    }
    const keep = 1 - fraction;
    for (let i = 0; i < target.weights.length; i++) {
        target.weights[i] = keep * target.weights[i] + fraction * referenceScale * reference.weights[i];
    }
    target.bias = keep * target.bias + fraction * referenceScale * reference.bias;
    return target;
}
