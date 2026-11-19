# Survival reference

Library version **0.1** · [Package overview](../README.md)

Load the community library with `load_module("main.flatppl")`.
All functions use existing FlatPPL primitives.

| Function | Arguments | Result |
|---|---|---|
| [`exponential`](#exponential) | `time, rate` | constant-hazard summary |
| [`weibull`](#weibull) | `time, shape, scale` | Weibull summary |
| [`lognormal`](#lognormal) | `time, mu, sigma` | lognormal summary |
| [`loglogistic`](#loglogistic) | `time, shape, scale` | log-logistic summary |
| [`piecewise_exponential`](#piecewise_exponential) | `time, starts, rates` | piecewise constant-hazard summary |
| [`log_survival`](#log_survival) | `summary` | log survival probability |
| [`survival`](#survival) | `summary` | survival probability |
| [`hazard`](#hazard) | `summary` | instantaneous event rate |
| [`log_density`](#log_density) | `summary` | log event-time density |
| [`right_censored_loglikelihood`](#right_censored_loglikelihood) | `summary, event` | exact or right-censored score |
| [`left_censored_loglikelihood`](#left_censored_loglikelihood) | `summary` | left-censored score |
| [`interval_loglikelihood`](#interval_loglikelihood) | `lower, upper` | interval-censored score |
| [`delayed_entry_loglikelihood`](#delayed_entry_loglikelihood) | `loglikelihood, entry` | score conditional on surviving entry |
| [`proportional_hazards`](#proportional_hazards) | `baseline, eta` | summary with hazard ratio $e^\eta$ |
| [`accelerated_failure_time`](#accelerated_failure_time) | `baseline, time, eta, parameters` | summary with time ratio $e^\eta$ |

## Conventions

For a positive continuous lifetime $T$, write

$$S(t)=\Pr(T>t),\qquad H(t)=-\log S(t),\qquad h(t)=f(t)/S(t).$$

A **summary** is `record(log_hazard = log(h), cumulative_hazard = H)`
evaluated at a supplied time. It is a numeric record, not a measure or a sampler.
Keeping $\log h$ and $H$ avoids recovering a hazard by subtracting two large
negative log probabilities. Hazard has inverse-time units; cumulative hazard
is dimensionless.

Times are finite and nonnegative. Rates, shapes, scales and standard deviations
are finite and strictly positive; locations and regression predictors are finite.
The four smooth families accept scalars or matching numeric vectors, with scalar
broadcasting. A vector call returns one record whose fields are vectors.
`piecewise_exponential` takes a scalar time and two parameter vectors.
Observation helpers operate elementwise and do not sum scores.

At zero, every $H$ is zero. Weibull and log-logistic hazards are $+\infty$,
$1/\text{scale}$, or $0$ for shapes below, equal to, or above one. The lognormal
hazard is zero. These are boundary limits; derivatives at singular boundaries
are not promised. Negative times and infinite endpoints are outside the API.
Use a right-censoring score for an interval extending to infinity.

No prior, censoring mechanism, truncation, or parameter transformation is implicit.
Scores omit only factors from an independent, parameter-free observation process.
Informative censoring needs a joint model for that process.

## Baseline families

### `exponential`

**`exponential(time, rate)`** returns

$$\log h(t)=\log r,\qquad H(t)=rt.$$

`rate` is an inverse-time rate, matching `Exponential(rate)`.

### `weibull`

**`weibull(time, shape, scale)`** returns, for $t>0$,

$$\log h(t)=\log k-\log a+(k-1)\log(t/a),\qquad H(t)=(t/a)^k.$$

`shape` is $k$ and `scale` is $a$ in time units, matching `Weibull(shape, scale)`.
Shape one gives `exponential(time, 1/scale)`.

### `lognormal`

**`lognormal(time, mu, sigma)`** describes $\log T\sim N(\mu,\sigma^2)$.
For $z=(\log t-\mu)/\sigma$,

$$S(t)=\Phi(-z),\qquad f(t)=\frac{e^{-z^2/2}}{t\sigma\sqrt{2\pi}}.$$

The parameters match `LogNormal(mu, sigma)`; `sigma` is a log-space standard
deviation. The implementation uses `special-functions.erfcx` in the positive
normal tail, so a finite log survival remains available when the probability
itself underflows. It uses `erfc` and `log1p` in the opposite tail. Very small
cumulative hazards can still round to zero in the chosen floating-point type.

### `loglogistic`

**`loglogistic(time, shape, scale)`** returns

$$S(t)=\frac{1}{1+(t/a)^k},\qquad
h(t)=\frac{(k/a)(t/a)^{k-1}}{1+(t/a)^k}.$$

Here $a$ is the median lifetime and $k$ is the shape. Equivalently,
$\log T\sim\mathrm{Logistic}(\log a,1/k)$.
Log-space evaluation avoids forming a large power explicitly.
The corresponding generative law is `pushfwd(exp, Logistic(log(scale), 1/shape))`.

### `piecewise_exponential`

**`piecewise_exponential(time, starts, rates)`** takes nonempty vectors of equal
length. `starts[1]` is zero; starts are finite and strictly increasing.
Every rate is positive. With $s_{m+1}=\infty$,

$$h(t)=r_j\quad(s_j\le t<s_{j+1}),\qquad
H(t)=\sum_{j=1}^m r_j\max(0,\min(t,s_{j+1})-s_j).$$

The hazard is right-continuous at starts. The final rate continues forever;
the cumulative hazard is continuous. Start times and vector lengths are caller
inputs, not inferred binning. Derivatives at hazard jumps are undefined.

## Observation likelihoods

### `log_survival`

**`log_survival(summary)`** returns $-H(t)=\log\Pr(T>t)$.
This remains useful when `survival(summary)` underflows.

### `survival`

**`survival(summary)`** returns $e^{-H(t)}$.
It can parameterize a `Bernoulli` observation for the event $T>t$.
Use log-space helpers for scores when that probability is too small to represent.

### `hazard`

**`hazard(summary)`** returns $e^{\log h(t)}$, an instantaneous rate, not a probability.

### `log_density`

**`log_density(summary)`** returns $\log h(t)-H(t)=\log f(t)$.
The reference measure is Lebesgue measure in lifetime units.

### `right_censored_loglikelihood`

**`right_censored_loglikelihood(summary, event)`** returns

$$\ell(t,d)=d\log h(t)-H(t),$$

where boolean `event` is true for an exact event and false for right censoring.
The implementation selects the hazard term rather than multiplying an infinite
log hazard by zero. A censored time of zero therefore contributes zero.
Sum the resulting scores for conditionally independent subjects.

### `left_censored_loglikelihood`

**`left_censored_loglikelihood(summary)`** returns

$$\log\Pr(T\le u)=\log(1-e^{-H(u)}).$$

It uses `expm1` when $H$ is small. A limit of zero gives $-\infty$.

### `interval_loglikelihood`

**`interval_loglikelihood(lower, upper)`** takes summaries from the same lifetime
law at finite endpoints $0\le l\le u$, and returns

$$\log\Pr(l<T\le u)=-H(l)+\log(1-e^{-[H(u)-H(l)]}).$$

Equal endpoints give $-\infty$. `expm1` avoids subtracting nearly equal survival
probabilities, but the difference $H(u)-H(l)$ must itself remain representable.
An exact event uses `log_density`, not a zero-width interval.

### `delayed_entry_loglikelihood`

**`delayed_entry_loglikelihood(loglikelihood, entry)`** returns

$$\ell_{\rm conditional}=\ell+H(e),$$

conditional on entry through $T>e$. The numerator event **must lie entirely above
the entry time**. Exact events and right censoring require their times to exceed
entry. For interval censoring, replace the lower endpoint by $\max(l,e)$ first.
For a left-censored event after entry, use the interval $(e,u]$ as the numerator;
passing an unconditional left-censoring score is incorrect.

This is left truncation (selection into the cohort), not left censoring.
The helper cannot verify these time relations from summaries alone.

## Regression transforms

### `proportional_hazards`

**`proportional_hazards(baseline, eta)`** returns

$$\log h(t)=\log h_0(t)+\eta,\qquad H(t)=e^\eta H_0(t).$$

Evaluate `baseline` at the target time first. `eta` is a scalar or matching vector,
often a linear predictor. A positive value raises the hazard. This is a fully
specified baseline model, not a Cox partial-likelihood implementation.

### `accelerated_failure_time`

**`accelerated_failure_time(baseline, time, eta, parameters)`** calls
`baseline(time .* exp.(-eta), parameters)`, then subtracts `eta` from its
log hazard. Thus

$$S(t)=S_0(te^{-\eta}),\qquad h(t)=e^{-\eta}h_0(te^{-\eta}).$$

`baseline` is a pure callable accepting a time and an explicit parameter object.
No stochastic input is captured. A positive `eta` lengthens lifetimes.

```flatppl
flatppl_compat = "0.1"
s = load_module("main.flatppl")
baseline(time, p) = s.weibull(time, p.shape, p.scale)
result = s.accelerated_failure_time(
    baseline, 2.0, 0.4, record(shape = 1.5, scale = 3.0)
)
```

For a Weibull baseline this equals `weibull(time, shape, scale * exp(eta))`.
Its proportional-hazards predictor is $`-\text{shape}\,\eta`$.

## Inference example

The [playground model](../examples/playground.flatppl) uses
$\log a_i=\alpha+\beta x_i$ with known Weibull shape $1.5$.
Exact events contribute Weibull densities. Independently right-censored subjects
contribute Bernoulli probabilities for the observed event $T>c_i$.
The likelihood is

$$L(\alpha,\beta)=\prod_{i\in E}f(t_i\mid a_i)
\prod_{i\in C}S(c_i\mid a_i).$$

There is no normalization by the probability of having an observed event:
the complete event/censoring pattern supplies that information. Fixed independent
censoring schedules contribute no parameter-dependent factor. The example leaves
the result unnormalized for the sampler, as `bayesupdate` normally does.
The numeric helpers preserve log likelihoods in more extreme tails than the
example's probability-valued Bernoulli representation.

The [Stan survival guide](https://mc-stan.org/docs/stan-users-guide/survival.html)
gives the same event-density and censored-survival construction.

## Validation

Run the mathematical checks with Python's standard library and Node 24 or newer:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check_math.py
```

The checks cover density normalization, event probabilities, origin limits,
piecewise boundaries, lognormal tails and the example's posterior density.
[The runtime fixture](../tests/runtime.flatppl) and the independent `runtime_values`
function in [the checker](../tests/check_math.py) support compiled value, gradient
and batching checks.
