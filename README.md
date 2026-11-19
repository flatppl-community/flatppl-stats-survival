# flatppl-stats-survival

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-survival%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Survival likelihoods and regression transforms for FlatPPL. Library version **0.1**.

Baselines return `record(log_hazard, cumulative_hazard)` with scalar or matching vector fields. For hazard $h$ and cumulative hazard $H$,

$$S(t)=e^{-H(t)},\qquad f(t)=h(t)e^{-H(t)},\qquad \ell_{\rm right}=d\log h(t)-H(t).$$

```flatppl
flatppl_compat = "0.1"
s = load_module("main.flatppl")
baseline = s.weibull([1.0, 2.0, 4.0], 1.5, 3.0)
probabilities = s.survival(baseline)
scores = s.right_censored_loglikelihood(baseline, [true, false, true])
```

Times are finite and nonnegative; shape, scale, rate and log-space standard deviation are positive. Exponential, Weibull, lognormal and log-logistic baselines accept scalars or matching vectors; piecewise exponential uses scalar time and matching knot/rate vectors. Helpers cover left and interval censoring, delayed entry, proportional hazards and accelerated failure time.

The [lifetime example](examples/playground.flatppl) infers a treatment effect. See the reference for domains, tail formulas and validation.
