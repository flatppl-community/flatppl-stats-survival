# flatppl-stats-survival

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-survival%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Survival models, censoring likelihoods, and regression transforms for FlatPPL. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
s = load_module("main.flatppl")
baseline = s.weibull([1.0, 2.0, 4.0], 1.5, 3.0)
probabilities = s.survival(baseline)
scores = s.right_censored_loglikelihood(baseline, [true, false, true])
```

Five baseline families provide hazard summaries. Helpers cover exact, right-, left-, and interval-censored observations, delayed entry, proportional hazards, and accelerated failure time models. Censoring and entry are explicit.

The [lifetime regression example](examples/playground.flatppl) infers a treatment effect from exact and right-censored synthetic observations with proper priors.

See the reference for definitions, domains, numerical limits, and validation commands.
