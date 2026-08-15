# GRPO reward contract

For each generated completion, require a strict JSON object with exactly
`1m`, `1y`, and `5y`. Each value is a probability in `[0, 1]` or `null`, and at
least one value must be non-null. Valid format receives weight 1. For observed
binary endpoints, the Brier-complement term is `2 * (1 - mean((p-y)^2))`; null
labels and null predictions are excluded. Invalid output receives a penalty of
1 and does not receive the Brier-complement term.
