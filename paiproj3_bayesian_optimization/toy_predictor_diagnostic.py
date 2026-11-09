"""Reproduce an overconfident missed objective peak in the current SafeBOAgent.

Run with the project's Python environment:
    python toy_predictor_diagnostic.py

The fixed training design deliberately leaves x in (4.8, 7.4) unobserved.
Both functions are smooth, f stays in [0, 1], and the entire domain is safe.
Observation noise has the task's standard deviations: 0.15 and 0.0001.
This isolates objective prediction from safety and acquisition decisions.

The script imports the real solution, feeds observations through observe(),
and calls _fit_models(). Ground truth is used only for evaluation and plotting.
A second, independently fitted agent gets three extra observations near the
hidden peak. This demonstrates the effect of better data coverage; it does
not change or repair the predictor. These observations are selected using
knowledge of this synthetic function, not by the acquisition function.

In addition to the before/after error comparison, diagnostics_before.png and
diagnostics_after.png show the objective, constraint, acquisition, and the
next query chosen by the fitted agent. The next query is not observed.
"""

import argparse
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from solution import MAX_SAFE_SA, SEARCH_BOUNDS, SafeBOAgent


TRAIN_X = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 7.4, 8.2, 9.0, 10.0])
EXTRA_X = np.array([5.65, 6.1, 6.55])
UNOBSERVED_INTERVAL = (4.8, 7.4)
NOISE_F_STD = 0.15
NOISE_V_STD = 0.0001


def true_objective(x):
    """A broad, modest peak and a higher smooth peak inside the sampling gap."""
    x = np.asarray(x, dtype=float)
    return (
        0.1
        + 0.3 * np.exp(-0.5 * ((x - 2.2) / 0.9) ** 2)
        + 0.85 * np.exp(-0.5 * ((x - 6.1) / 0.45) ** 2)
    )


def true_safety(x):
    """A smooth positive constraint comfortably below 4 everywhere on [0, 10]."""
    x = np.asarray(x, dtype=float)
    return 3.15 + 0.04 * (x - 5.0) + 0.18 * np.sin(1.2 * x)


def observations(x, rng):
    return np.column_stack((
        x,
        true_objective(x) + rng.normal(0.0, NOISE_F_STD, len(x)),
        true_safety(x) + rng.normal(0.0, NOISE_V_STD, len(x)),
    ))


def write_csv(path, values, header):
    np.savetxt(path, values, delimiter=",", header=header, comments="", fmt="%.17g")


def save_figure(fig, path):
    """Replace completed plots atomically, including on Windows bind mounts."""
    rendered_path = path.with_name(f"{path.stem}.render{path.suffix}")
    fig.savefig(rendered_path, dpi=150)
    rendered_path.replace(path)
    plt.close(fig)


def evaluate(training, grid):
    """Evaluate latent-function accuracy on inputs absent from the training set."""
    agent = SafeBOAgent()
    for x, yf, yv in training:
        agent.observe(np.array([[x]]), yf, yv)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent._fit_models()
        f_mean, f_std = agent.gpr_estim_f.predict(grid[:, None], return_std=True)
        v_mean, v_std = agent._predict_safety(grid[:, None])
        # Use the fitted agent's actual selector without refitting or observing.
        next_point = agent.optimize_acquisition()
        next_f_mean, next_f_std = agent.gpr_estim_f.predict(next_point, return_std=True)
        next_v_mean, next_v_std = agent._predict_safety(next_point)
        next_acquisition = agent.acquisition_function(next_point).item()

    f_true = true_objective(grid)
    v_true = true_safety(grid)
    held_out = ~np.any(
        np.isclose(grid[:, None], training[None, :, 0], atol=1e-12, rtol=0.0),
        axis=1,
    )
    gap = (grid > UNOBSERVED_INTERVAL[0]) & (grid < UNOBSERVED_INTERVAL[1])
    gap_held_out = gap & held_out
    f_error = np.abs(f_true - f_mean)
    peak_idx = int(np.argmax(f_true))
    peak_x = float(grid[peak_idx])
    peak_mean, peak_std = agent.gpr_estim_f.predict([[peak_x]], return_std=True)
    certified = (
        v_mean + agent.safety_confidence_scale * v_std
        < MAX_SAFE_SA - agent.safety_margin
    )
    metrics = {
        "n_observations": len(training),
        "n_held_out_grid_points": int(held_out.sum()),
        "objective_held_out_rmse": float(np.sqrt(np.mean(f_error[held_out] ** 2))),
        "objective_gap_held_out_rmse": float(np.sqrt(np.mean(f_error[gap_held_out] ** 2))),
        "objective_held_out_3sigma_coverage": float(np.mean(
            f_error[held_out] <= 3.0 * f_std[held_out]
        )),
        "objective_gap_held_out_3sigma_coverage": float(np.mean(
            f_error[gap_held_out] <= 3.0 * f_std[gap_held_out]
        )),
        "safety_held_out_rmse": float(np.sqrt(np.mean(
            (v_true[held_out] - v_mean[held_out]) ** 2
        ))),
        "pointwise_certified_safe_grid_fraction": float(np.mean(certified)),
        "pointwise_false_safe_grid_points": int(np.sum(certified & (v_true >= MAX_SAFE_SA))),
        "true_peak_x": peak_x,
        "true_peak_value": float(f_true[peak_idx]),
        "predicted_mean_at_true_peak": float(peak_mean.item()),
        "predicted_std_at_true_peak": float(peak_std.item()),
        "absolute_error_at_true_peak": float(f_error[peak_idx]),
        "peak_error_in_predicted_std_units": float(
            f_error[peak_idx] / max(peak_std.item(), np.finfo(float).eps)
        ),
        "objective_kernel": str(agent.gpr_estim_f.kernel_),
        "safety_kernel": str(agent.gpr_estim_v.kernel_),
        "objective_confidence_scale": agent.confidence_width_scale,
        "safety_confidence_scale": agent.safety_confidence_scale,
        "safety_margin": agent.safety_margin,
        "next_query_x": float(next_point.item()),
        "next_query_objective_mean": float(next_f_mean.item()),
        "next_query_objective_ucb": float(
            next_f_mean.item() + agent.confidence_width_scale * next_f_std.item()
        ),
        "next_query_safety_mean": float(next_v_mean.item()),
        "next_query_safety_ucb": float(
            next_v_mean.item() + agent.safety_confidence_scale * next_v_std.item()
        ),
        "next_query_passes_pointwise_filter": bool(np.isfinite(next_acquisition)),
        "fit_warnings": list(dict.fromkeys(str(w.message) for w in caught)),
    }
    predictions = np.column_stack((grid, f_true, f_mean, f_std, v_true, v_mean, v_std))
    return metrics, predictions, agent


def plot_diagnostics(path, training, grid, agent, metrics, title):
    """Plot the fitted models and acquisition, like SafeBOAgent.plot_diagnostics.

    The acquisition curve applies pointwise safety. The marked next query
    comes from the real optimizer, which also enforces seed connectivity and
    handles the trusted-seed fallback.
    """
    x = np.unique(np.concatenate((grid, agent.candidate_grid, training[:, 0])))
    candidates = x[:, None]
    f_mean, f_std = agent.gpr_estim_f.predict(candidates, return_std=True)
    v_mean, v_std = agent._predict_safety(candidates)
    f_width = agent.confidence_width_scale * f_std
    v_width = agent.safety_confidence_scale * v_std
    objective_ucb = f_mean + f_width
    acquisition = agent.acquisition_function(candidates)
    certified = np.isfinite(acquisition)
    query_x = metrics["next_query_x"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax in axes:
        ax.axvspan(*UNOBSERVED_INTERVAL, color="#f0b44d", alpha=0.12)
        ax.axvline(query_x, color="#9c2462", linestyle=":", alpha=0.8)
        ax.set_xlim(*SEARCH_BOUNDS[0])
        ax.set_xlabel("Structural parameter x")
        ax.grid(alpha=0.2)

    ax = axes[0]
    ax.set_title("Objective posterior")
    ax.plot(x, true_objective(x), color="#168268", label="True objective")
    ax.plot(x, f_mean, color="#2864be", label="GP mean")
    ax.fill_between(x, f_mean - f_width, f_mean + f_width,
                    color="#2864be", alpha=0.16,
                    label=f"Mean +/- {agent.confidence_width_scale:g} std")
    ax.scatter(training[:, 0], training[:, 1], color="#b85120", s=30,
               zorder=4, label="Observations")
    ax.scatter([query_x], [metrics["next_query_objective_mean"]], marker="*",
               s=170, color="#9c2462", zorder=5, label="Next query (predicted)")
    ax.set_ylabel("Objective f(x)")
    ax.text(0.5, -0.23, str(agent.gpr_estim_f.kernel_), transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_title("Constraint posterior")
    ax.plot(x, true_safety(x), color="#168268", label="True SA")
    ax.plot(x, v_mean, color="#2864be", label="GP mean (SA scale)")
    ax.fill_between(x, v_mean - v_width, v_mean + v_width,
                    color="#2864be", alpha=0.16,
                    label=f"Mean +/- {agent.safety_confidence_scale:g} std")
    ax.plot(x, v_mean + v_width, color="#2864be", linestyle="--", label="Safety UCB")
    ax.scatter(training[:, 0], training[:, 2], color="#b85120", s=30, zorder=4)
    ax.scatter([query_x], [metrics["next_query_safety_mean"]], marker="*",
               s=170, color="#9c2462", zorder=5, label="Next query (predicted)")
    ax.axhline(MAX_SAFE_SA, color="#b92f3c", linestyle="--", label="Safety threshold")
    ax.set_ylabel("Constraint v(x)")
    ax.text(0.5, -0.23, str(agent.gpr_estim_v.kernel_), transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.set_title("Acquisition and next query")
    ax.plot(x, objective_ucb, color="#777777", linestyle="--", label="Objective UCB")
    # Rejected values are -inf; plot them as gaps rather than finite scores.
    ax.plot(x, np.where(certified, acquisition, np.nan), color="#2864be",
            linewidth=2, label="Pointwise safe acquisition")
    ax.fill_between(x, 0, 1, where=~certified, transform=ax.get_xaxis_transform(),
                    color="#b92f3c", alpha=0.1, label="Rejected by safety filter")
    ax.scatter([query_x], [metrics["next_query_objective_ucb"]], marker="*",
               s=170, color="#9c2462", zorder=5, label=f"Actual next query: {query_x:.3f}")
    ax.set_ylabel("Acquisition score")
    if not metrics["next_query_passes_pointwise_filter"]:
        ax.text(0.5, 0.02, "Trusted initial safe point used as fallback",
                transform=ax.transAxes, ha="center", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle(f"{title} ({len(training)} observations)", fontsize=14)
    fig.text(0.5, 0.025,
             "Stars mark predictions at the next query; no new observation is added. "
             "Selection also applies the seed-connectivity filter.\n"
             f"Orange shading: original sampling gap. Safety margin: {agent.safety_margin:g}.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    save_figure(fig, path)


def plot_comparison(path, training, augmented, before, after):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    error_limit = 1.05 * max(
        max(np.max(np.abs(p[:, 1] - p[:, 2])), np.max(3 * p[:, 3]))
        for p in (before, after)
    )
    for column, (data, predictions, title) in enumerate((
        (training, before, "Before: 11 observations, no samples near the high peak"),
        (augmented, after, "After: three extra observations near the high peak"),
    )):
        x, f_true, f_mean, f_std, v_true, v_mean, v_std = predictions.T
        for row in range(3):
            axes[row, column].axvspan(*UNOBSERVED_INTERVAL, color="#f0b44d", alpha=0.12)
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xlim(*SEARCH_BOUNDS[0])

        ax = axes[0, column]
        ax.set_title(title, fontsize=11)
        ax.plot(x, f_true, color="#168268", linewidth=2, label="True objective")
        ax.plot(x, f_mean, color="#2864be", linewidth=2, label="GP mean")
        ax.fill_between(x, f_mean - 3 * f_std, f_mean + 3 * f_std,
                        color="#2864be", alpha=0.16, label="GP mean +/- 3 std")
        ax.scatter(data[:, 0], data[:, 1], color="#b85120", s=30, zorder=4,
                   label="Noisy training observations")
        ax.axvline(6.1, color="#168268", linestyle=":", alpha=0.8)
        ax.set_ylabel("Objective f(x)")
        ax.set_ylim(-0.3, 1.3)
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[1, column]
        ax.plot(x, v_true, color="#168268", linewidth=2, label="True SA")
        ax.plot(x, v_mean, color="#2864be", linewidth=1.5, label="GP mean")
        ax.fill_between(x, v_mean - 3 * v_std, v_mean + 3 * v_std,
                        color="#2864be", alpha=0.16, label="GP mean +/- 3 std")
        ax.scatter(data[:, 0], data[:, 2], color="#b85120", s=25, zorder=4)
        ax.axhline(MAX_SAFE_SA, color="#b92f3c", linestyle="--", label="Safety threshold")
        ax.set_ylabel("Constraint v(x)")
        ax.set_ylim(2.7, 4.15)
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[2, column]
        ax.plot(x, np.abs(f_true - f_mean), color="#b85120", linewidth=2,
                label="Absolute objective error")
        ax.plot(x, 3 * f_std, color="#2864be", linestyle="--", label="3 x predicted std")
        ax.set_ylabel("Error / uncertainty")
        ax.set_xlabel("Structural parameter x")
        ax.set_ylim(0, error_limit)
        ax.legend(fontsize=8)

    fig.suptitle("A smooth hidden peak exposes overconfident objective predictions", fontsize=15)
    fig.text(0.5, 0.015,
             "Orange shading: original sampling gap. Ground truth is never used to fit either GP.\n"
             "The additional observations are a controlled coverage experiment, not a BO trajectory.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "toy_data" / "hidden_peak")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    training = observations(TRAIN_X, rng)
    extra = observations(EXTRA_X, rng)
    augmented = np.vstack((training, extra))
    grid = np.linspace(*SEARCH_BOUNDS[0], 2001)
    reference = np.column_stack((grid, true_objective(grid), true_safety(grid)))
    write_csv(output / "train.csv", training, "x,y_objective,y_safety")
    write_csv(output / "extra_observations.csv", extra, "x,y_objective,y_safety")
    write_csv(output / "reference.csv", reference, "x,true_objective,true_safety")

    before_metrics, before, before_agent = evaluate(training, grid)
    after_metrics, after, after_agent = evaluate(augmented, grid)
    prediction_columns = "x,true_objective,predicted_objective,std_objective,true_safety,predicted_safety,std_safety"
    write_csv(output / "predictions_before.csv", before, prediction_columns)
    write_csv(output / "predictions_after.csv", after, prediction_columns)
    metrics = {
        "seed": args.seed,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "noise_std_objective": NOISE_F_STD,
        "noise_std_safety": NOISE_V_STD,
        "original_unobserved_interval": list(UNOBSERVED_INTERVAL),
        "design": "Fixed-design synthetic stress case; not sampled by the BO acquisition.",
        "before": before_metrics,
        "after_extra_observations": after_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_comparison(output / "diagnostic.png", training, augmented, before, after)
    plot_diagnostics(output / "diagnostics_before.png", training, grid,
                     before_agent, before_metrics, "Before extra peak observations")
    plot_diagnostics(output / "diagnostics_after.png", augmented, grid,
                     after_agent, after_metrics, "After extra peak observations")

    for name, result in (("BEFORE", before_metrics), ("AFTER +3 observations", after_metrics)):
        print(f"{name}: {result['n_observations']} observations")
        print(f"  Objective held-out RMSE: {result['objective_held_out_rmse']:.6f}")
        print(f"  At true peak x={result['true_peak_x']:.3f}: "
              f"true={result['true_peak_value']:.6f}, "
              f"mean={result['predicted_mean_at_true_peak']:.6f}, "
              f"std={result['predicted_std_at_true_peak']:.6f}")
        print(f"  Peak error: {result['peak_error_in_predicted_std_units']:.2f} predicted std")
        print(f"  Safety held-out RMSE: {result['safety_held_out_rmse']:.6f}")
        print(f"  Objective kernel: {result['objective_kernel']}")
        print(f"  Next query: x={result['next_query_x']:.3f}, "
              f"predicted SA UCB={result['next_query_safety_ucb']:.6f}")
        if result["fit_warnings"]:
            print(f"  Fitting warnings: {len(result['fit_warnings'])}; see metrics.json")
    print(f"Saved dataset, predictions, metrics, diagnostic.png, "
          f"and diagnostics_before/after.png to {output}")


if __name__ == "__main__":
    main()
