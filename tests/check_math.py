"""Check exported functions against distribution densities and event probabilities."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_POINTS = [
    [math.log(0.4), math.log(1.5), math.log(3.0), 0.2, math.log(0.8)],
    [math.log(0.7), 0.0, math.log(2.0), math.log(2.0), 0.0],
    [math.log(0.2), math.log(0.7), math.log(1.4), -0.3, math.log(1.2)],
]


def simpson(function, lower, upper, count=4000):
    step = (upper - lower) / count
    total = function(lower) + function(upper)
    total += 4 * sum(function(lower + i * step) for i in range(1, count, 2))
    total += 2 * sum(function(lower + i * step) for i in range(2, count, 2))
    return total * step / 3


def normal_logsurvival(z):
    if z < 8:
        return math.log(math.erfc(z / math.sqrt(2)) / 2)
    # Integrate the normal tail after u=z+v/z. This never evaluates exp(-z*z/2).
    integral = simpson(lambda v: math.exp(-v - v * v / (2 * z * z)), 0, 40)
    return -z * z / 2 - math.log(2 * math.pi) / 2 + math.log(integral / z)


def family(name, time, *parameters):
    """Return log density and log survival from independent lifetime laws."""
    if name == "exponential":
        (rate,) = parameters
        return math.log(rate) - rate * time, -rate * time
    if name == "weibull":
        shape, scale = parameters
        power = (time / scale) ** shape
        return math.log(shape / scale) + (shape - 1) * math.log(
            time / scale
        ) - power, -power
    if name == "lognormal":
        mu, sigma = parameters
        z = (math.log(time) - mu) / sigma
        return -z * z / 2 - math.log(
            time * sigma * math.sqrt(2 * math.pi)
        ), normal_logsurvival(z)
    shape, scale = parameters
    ratio = time / scale
    density = shape / scale * ratio ** (shape - 1) / (1 + ratio**shape) ** 2
    return math.log(density), -math.log1p(ratio**shape)


def runtime_values(x):
    rate, shape, scale, sigma = (math.exp(x[i]) for i in [0, 1, 2, 4])
    mu = x[3]
    ex = family("exponential", 2, rate)
    we = family("weibull", 2, shape, scale)
    ln = family("lognormal", 2, mu, sigma)
    ll = family("loglogistic", 2, shape, scale)
    # Explicit exposure in three intervals, with the final rate continuing forever.
    pw = (math.log(1.5 * rate) - 3.7 * rate, -3.7 * rate)
    multiplier = math.exp(mu)
    # PH is a powered survivor. Its density is q*S^(q-1)*f.
    ph = (math.log(multiplier) + (multiplier - 1) * we[1] + we[0], multiplier * we[1])
    aft = family("weibull", 2, shape, scale * math.exp(mu))
    lower = family("weibull", 1.9, shape, scale)[1]
    entry = family("weibull", 0.5, shape, scale)[1]
    return [
        *ex,
        *we,
        *ln,
        *ll,
        *pw,
        *ph,
        *aft,
        we[0],
        we[1],
        math.log(-math.expm1(we[1])),
        lower + math.log(-math.expm1(we[1] - lower)),
        we[0] - entry,
        normal_logsurvival((40 - mu) / sigma),
        family("loglogistic", scale, shape, scale)[0],
        math.log(0.5),
    ]


def posterior(point):
    alpha, beta = point
    events = [(1.8, 0), (3.2, 0), (4.1, 0), (3.0, 1), (4.9, 1), (5.7, 1)]
    censored = [(5.0, 0), (6.0, 0), (7.0, 1), (8.0, 1)]
    prior = -0.5 * ((alpha - math.log(5)) / 0.6) ** 2 - math.log(
        0.6 * math.sqrt(2 * math.pi)
    )
    prior += -0.5 * (beta / 0.8) ** 2 - math.log(0.8 * math.sqrt(2 * math.pi))
    return (
        prior
        + sum(
            family("weibull", t, 1.5, math.exp(alpha + beta * x))[0] for t, x in events
        )
        + sum(
            family("weibull", t, 1.5, math.exp(alpha + beta * x))[1]
            for t, x in censored
        )
    )


def decode(value) -> Any:
    if isinstance(value, dict):
        if "data" in value:
            data = [decode(x) for x in value["data"].values()]
            return data[0] if value.get("shape") == [] else data
        return {key: decode(x) for key, x in value.items()}
    if isinstance(value, list):
        return [decode(x) for x in value]
    return float(value) if isinstance(value, str) else value


def evaluate(request):
    result = subprocess.run(
        ["node", str(ROOT / "tests/evaluate.cjs")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    return decode(json.loads(result.stdout))


def close(actual, expected, tolerance=2e-10):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            close(a, e, tolerance)
    else:
        assert actual == expected or math.isclose(
            actual, expected, rel_tol=tolerance, abs_tol=tolerance
        ), (actual, expected)


def run():
    probes, expected = [], []
    families = [
        ("exponential", [0.4]),
        ("weibull", [1.7, 3.0]),
        ("lognormal", [0.2, 0.9]),
        ("loglogistic", [1.8, 2.0]),
    ]
    for name, parameters in families:
        for time in [0.01, 0.5, 2.0, 20.0]:
            summary = f"a.{name}({time},{','.join(map(str, parameters))})"
            probes += [f"a.log_density({summary})", f"a.log_survival({summary})"]
            expected += list(family(name, time, *parameters))
    close(evaluate(probes), expected)
    # Recover normalized densities from the exported values on log-time quadrature nodes.
    nodes = [-25 + i * 0.025 for i in range(2001)]
    for name, parameters in families:
        times = [math.exp(u) for u in nodes]
        vector = evaluate(
            [
                f"a.log_density(a.{name}({json.dumps(times)},{','.join(map(str, parameters))}))"
            ]
        )[0]
        transformed = [math.exp(v + u) for v, u in zip(vector, nodes)]
        mass = (
            0.025
            / 3
            * (
                transformed[0]
                + transformed[-1]
                + 4 * sum(transformed[1:-1:2])
                + 2 * sum(transformed[2:-1:2])
            )
        )
        close(mass, 1, 1e-9)
    # Origin limits, one interval, breakpoint convention and continuation.
    close(
        evaluate(
            [
                "a.hazard(a.weibull(0.0,0.7,3.0))",
                "a.hazard(a.weibull(0.0,1.0,3.0))",
                "a.hazard(a.weibull(0.0,1.5,3.0))",
                "a.hazard(a.lognormal(0.0,0.2,0.9))",
                "a.log_survival(a.lognormal(0.0,0.2,0.9))",
                "a.log_survival(a.piecewise_exponential(4.0,[0.0],[0.3]))",
                "a.hazard(a.piecewise_exponential(3.0,[0.0,1.0,3.0],[0.3,0.7,0.2]))",
                "a.log_survival(a.piecewise_exponential(4.0,[0.0,1.0,3.0],[0.3,0.7,0.2]))",
                "a.interval_loglikelihood(a.exponential(0.0,0.3),a.exponential(0.000000000001,0.3))",
                "a.interval_loglikelihood(a.exponential(2.0,0.3),a.exponential(2.0,0.3))",
                "a.left_censored_loglikelihood(a.exponential(0.0,0.3))",
                "a.log_survival(a.lognormal(exp(40.0),0.0,1.0))",
            ]
        ),
        [
            math.inf,
            1 / 3,
            0,
            0,
            0,
            -1.2,
            0.2,
            -1.9,
            math.log(-math.expm1(-3e-13)),
            -math.inf,
            -math.inf,
            normal_logsurvival(40),
        ],
    )
    # AFT Jacobian and PH scaling use distinct identities of the baseline law.
    close(
        evaluate(
            [
                "a.log_density(a.accelerated_failure_time((t,p)->a.weibull(t,p.shape,p.scale),2.0,0.4,record(shape=1.5,scale=3.0)))",
                "a.log_survival(a.proportional_hazards(a.weibull(2.0,1.5,3.0),0.4))",
                "a.delayed_entry_loglikelihood(a.interval_loglikelihood(a.weibull(1.0,1.5,3.0),a.weibull(2.0,1.5,3.0)),a.weibull(1.0,1.5,3.0))",
            ]
        ),
        [
            family("weibull", 2, 1.5, 3 * math.exp(0.4))[0],
            math.exp(0.4) * family("weibull", 2, 1.5, 3)[1],
            math.log1p(
                -math.exp(
                    family("weibull", 2, 1.5, 3)[1] - family("weibull", 1, 1.5, 3)[1]
                )
            ),
        ],
    )
    source = (ROOT / "tests/runtime.flatppl").read_text()
    for x in RUNTIME_POINTS:
        close(
            evaluate(
                {
                    "source": source.replace(
                        "elementof(cartpow(reals, 5))", json.dumps(x)
                    ),
                    "bindings": ["outputs"],
                }
            )[0],
            runtime_values(x),
        )
    points = [[1.3, -0.2], [1.6, 0.2], [1.9, 0.5]]
    query = 'flatppl_compat="0.1"\np=load_module("../examples/playground.flatppl")\n'
    query += "\n".join(
        f"score{i}=logdensityof(p.Pi_post,record(log_scale={x[0]},treatment_effect={x[1]}))"
        for i, x in enumerate(points)
    )
    close(
        evaluate({"source": query, "bindings": [f"score{i}" for i in range(3)]}),
        [posterior(x) for x in points],
    )
    print(
        json.dumps(
            {
                "family_points": 16,
                "normalization_checks": 4,
                "runtime_points": 3,
                "runtime_outputs": 22,
                "posterior_points": 3,
            }
        )
    )


if __name__ == "__main__":
    run()
