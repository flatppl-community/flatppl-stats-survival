# flatppl-stats-survival

[Function reference](docs/reference.md)

Survival models, censoring likelihoods and regression transforms in FlatPPL.
Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
s = load_module("main.flatppl")
baseline = s.weibull([1.0, 2.0, 4.0], 1.5, 3.0)
probabilities = s.survival(baseline)
scores = s.right_censored_loglikelihood(baseline, [true, false, true])
```

The [lifetime regression example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-survival%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)
infers a treatment effect from exact and right-censored synthetic lifetimes.
It uses proper priors and the ordinary prior–kernel–likelihood–posterior flow.

Five baseline families return numeric hazard summaries. Helpers provide exact,
right-, left- and interval-censored likelihoods, delayed entry, proportional
hazards and accelerated failure time models. Censoring and entry remain explicit.
The [reference](docs/reference.md) gives every formula and its domain.

Run the mathematical checks with Python's standard library and Node 24 or newer:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check_math.py
```

The checks cover density normalization, event probabilities, origin limits,
piecewise boundaries, lognormal tails and the example's posterior density.
[The runtime fixture](tests/runtime.flatppl) and the independent `runtime_values`
function in [the checker](tests/check_math.py) support compiled value, gradient
and batching checks.
