# Sprout Slop Filter

A small machine-learning project exploring how to detect meaningless or low-effort text ("slop"), starting with task descriptions in a task tracker.

## Goal

Given a string, predict how likely it is to be slop, returning a score from **0 to 1**:

- `0` = the model thinks the text is genuine.
- `1` = the model thinks the text is slop.

"Slop" means meaningless, nonsensical, or empty filler—not something unethical or bad. For example, `Steal someone's wallet` describes a genuine intention even though the action is unethical.

## Basic architecture plan

1. **Training examples** — collect text samples and label each one as either genuine or slop.
2. **Split text into words** — turn the input string into words the model can work with.
3. **Represent words with numbers** — give words numerical representations (vectors). These can be learned during training.
4. **Handle unfamiliar words** — explore looking up unknown words in a thesaurus and connecting them to known related words. Related words are clues, not guaranteed replacements.
5. **Combine the words** — use the word representations together to make a representation of the full input.
6. **Make a prediction** — use the combined representation to produce a slop score between 0 and 1.
7. **Learn from mistakes** — compare predictions with the training labels and adjust the model's numbers to improve future predictions.
8. **Test it** — check how well it handles examples it was not trained on, including genuine text it should not incorrectly flag.

This is an initial plan, not a claim that all these components have been implemented. The first version should stay small and understandable.

## Example labels

| Text | Label |
| --- | --- |
| `Buy milk` | Genuine |
| `Finish maths homework` | Genuine |
| `Do something` | Depends on the intended definition; possibly slop |
| `asdfghjkl` | Slop |
| `Steal someone's wallet` | Genuine text, despite describing an unethical action |

The labels used for training will determine what the model learns, so examples should cover varied wording and avoid treating vagueness or ethics as automatically meaning slop.

## Possible future use

The same general training code could later be used for other text-classification tasks, such as checking whether GitHub commit messages are informative. That would need its own examples and labels: a message that is acceptable as a task description may still be a poor commit message.

## Current status

Architecture planning. The model, training data, and thesaurus-based vocabulary expansion are not yet implemented.
