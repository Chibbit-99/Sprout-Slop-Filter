# Sprout Slop Filter — Subsystems

This document lists the systems Sprout may need and describes each one as a function: what it receives, what it returns, and what it does. These are design sketches, not implemented code. The first version should be kept small; some training and testing systems are only needed while building the model.

## Main prediction pipeline

These functions work together when Sprout receives a string and needs to produce a slop score.

### 1. `normalise_text(text) -> clean_text`

**Input:** A text string.  
**Output:** A consistently formatted string.

Cleans up harmless differences, such as surrounding whitespace and inconsistent casing, so the model does not treat `"Buy Milk"` and `"buy milk"` as completely different just because of capitalisation. It should not delete meaningful punctuation or words without a reason.

### 2. `tokenise(clean_text) -> tokens`

**Input:** The cleaned string.  
**Output:** A list of tokens (usually words, and possibly punctuation).

Splits the text into manageable pieces. For example, `"Purchase some milk."` might become `["purchase", "some", "milk"]`. The exact rules should be consistent during both training and prediction.

### 3. `map_synonyms(token, vocabulary) -> word_links`

**Input:** Token and the model's known vocabulary.  
**Output:** Optional links between tokens and known related words.

For words that are not in the vocabulary, this function can query the [Datamuse API](https://www.datamuse.com/api/) for synonyms. For example, a lookup for `purchase` may return `buy`, which could already be known to Sprout. It records that relationship as a clue; it should **not automatically replace every word with a synonym**.

Synonym results can include another sense of a word. For example, `buy` may return `steal` or `bribe`, so a returned word is not guaranteed to be interchangeable in a sentence. The first version should use one round of lookups only, cache results, handle API failures, and allow words to remain unknown if no useful match is found.

### 4. `get_word_vector(token, vocabulary, word_links) -> vector`

**Input:** A token, the vocabulary, and any synonym links.  
**Output:** A numerical vector representing the token for the model.

For a known word, retrieves its learned vector. For an unknown word, uses a suitable known-word link if one is available, or a consistent unknown-word representation otherwise. Synonym links should be used cautiously so an unreliable match does not force two words to mean exactly the same thing.

### 5. `encode_text(tokens, vectors) -> text_representation`

**Input:** The tokens and their vectors.  
**Output:** One numerical representation of the whole string.

Combines the word information into a single representation the classifier can use. The first version could use a simple method, such as averaging the word vectors. More complicated ways of combining words can be explored later.

### 6. `predict_slop(text_representation, model_parameters) -> score`

**Input:** The text representation and the model's learned parameters.  
**Output:** A score between `0` and `1`.

Uses the combined representation to estimate how likely the text is to be slop. A score closer to `0` means the model leans towards genuine text; a score closer to `1` means it leans towards slop. This is a model score, not a guarantee that the prediction is correct.

### 7. `classify_text(text, model) -> score`

**Input:** A raw string and a loaded model.  
**Output:** The slop score.

Coordinates the prediction pipeline: normalise the text, tokenise it, find any useful synonym links, get vectors, encode the text, and ask the classifier for a score. Keeping this as one public function makes the rest of the project easier to use.

## Training systems

These functions are used to teach the model from examples labelled as genuine or slop.

### 8. `load_training_data(file) -> examples`

**Input:** A data file containing text and labels.  
**Output:** A list of examples, each containing text and its label.

Loads labelled examples, such as `("Buy milk", genuine)` and `("asdfghjkl", slop)`. It should check that labels are valid and that examples are not accidentally missing their text or label.

### 9. `split_dataset(examples) -> train_set, validation_set, test_set`

**Input:** All labelled examples.  
**Output:** Separate training, validation, and test sets.

Keeps some examples aside so the model can be checked on text it did not learn from. The test set should not be used to adjust the model during training.

### 10. `initialise_model(vocabulary, settings) -> model_parameters`

**Input:** The vocabulary and model settings, including vector size.  
**Output:** The model's starting numerical values.

Creates the word vectors and classifier parameters. The initial values should be set in a controlled way so training can adjust them.

### 11. `make_prediction_for_training(text, model_parameters) -> score`

**Input:** A training example's text and the current model parameters.  
**Output:** The predicted slop score.

Runs the same text-processing and prediction steps used during normal use. Training must use consistent tokenisation and vocabulary rules so the model learns from the representation it will later receive.

### 12. `calculate_loss(score, correct_label) -> error`

**Input:** The predicted score and the example's correct label.  
**Output:** A numerical measure of how wrong the prediction was.

Produces a larger error when the prediction disagrees strongly with the label and a smaller error when it agrees. A suitable loss function for binary classification can be chosen for the first implementation.

### 13. `update_parameters(model_parameters, error_information, learning_rate) -> updated_parameters`

**Input:** The model parameters and information about how they contributed to the error.  
**Output:** Adjusted parameters.

Changes the word vectors and classifier values a little at a time to make similar mistakes less likely. This requires calculating how each parameter contributed to the error; the exact method depends on the chosen model.

### 14. `train_model(train_set, validation_set, settings) -> trained_model`

**Input:** Training examples, validation examples, and settings such as number of training rounds.  
**Output:** A trained model.

Repeats prediction, error calculation, and parameter updates over the training examples. Checks validation performance along the way to see whether learning is improving. It should not train directly on the final test set.

## Testing and model storage

### 15. `evaluate_model(model, test_set) -> metrics`

**Input:** A trained model and held-out test examples.  
**Output:** Measurements of how well it performs.

Checks performance on examples the model did not train on. Useful measurements include how often it correctly identifies slop and, importantly, how often it wrongly flags genuine text. Tests should cover varied wording, including words that were not in the training vocabulary.

### 16. `save_model(model, path) -> result`

**Input:** The trained model and a destination path.  
**Output:** A success or failure result.

Saves the learned vectors, classifier parameters, and necessary vocabulary/settings so the model can be used again without retraining.

### 17. `load_model(path) -> model`

**Input:** A saved model path.  
**Output:** A ready-to-use model, or a clear error if loading fails.

Restores the parameters and vocabulary needed by `classify_text`.

### 18. `get_synonyms_cached(word) -> synonyms`

**Input:** A word.  
**Output:** A list of synonym results, or an empty list if lookup fails or returns nothing useful.

Supports `map_synonyms` by checking a local cache before requesting Datamuse. This avoids repeating the same requests. API errors, rate limits, or unavailable network access should not crash the entire classifier; Sprout should be able to continue without synonym information.

## Overall flow

**When predicting:**  
`classify_text` → `normalise_text` → `tokenise` → `map_synonyms` → `get_word_vector` → `encode_text` → `predict_slop`

**When training:**  
`load_training_data` → `split_dataset` → `initialise_model` → repeat prediction, `calculate_loss`, and `update_parameters` → `evaluate_model` → `save_model`

The names are illustrative. Functions may be combined or renamed in the actual code, and the synonym lookup is optional: Sprout should still be able to make a prediction when the API is unavailable.

## Suggested first implementation

Start with the smallest useful version:

1. `normalise_text` and `tokenise`
2. `load_training_data` and `split_dataset`
3. Word vectors, a simple `encode_text`, and `predict_slop`
4. `calculate_loss`, `update_parameters`, and `train_model`
5. `evaluate_model`, `save_model`, and `load_model`
6. Add synonym lookup and caching as a separate experiment, then compare performance with and without it.

This order gives Sprout a working baseline before adding the extra complexity of synonym mapping.
