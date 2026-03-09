import argparse
import math
import random
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_printoptions(precision=10)


def format_scalar(value):
    return f"{value:.10f}"


def loss_and_grad_1d(params):
    w = params[0]
    loss = 0.5 * (w - 4.0) ** 2
    grad = torch.tensor([w.item() - 4.0], dtype=torch.float64)
    return float(loss.item()), grad


def loss_and_grad_2d(params):
    x, y = params.tolist()
    loss = 0.5 * (100.0 * x * x + y * y)
    grad = torch.tensor([100.0 * x, y], dtype=torch.float64)
    return loss, grad


def make_noise_sequence(num_steps, dim, noise_std=0.0, seed=0):
    rng = random.Random(seed)
    return [
        torch.tensor([rng.gauss(0.0, noise_std) for _ in range(dim)], dtype=torch.float64)
        for _ in range(num_steps)
    ]


def make_spike_sequence(num_steps, spike_step, spike_value):
    sequence = [torch.zeros(1, dtype=torch.float64) for _ in range(num_steps)]
    sequence[spike_step - 1] = torch.tensor([spike_value], dtype=torch.float64)
    return sequence


def simulate_path(
    optimizer_name,
    initial_params,
    loss_grad_fn,
    steps,
    lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    noise_sequence=None,
):
    params = torch.tensor(initial_params, dtype=torch.float64)
    noise_sequence = noise_sequence or [torch.zeros_like(params) for _ in range(steps)]
    history = []

    if optimizer_name == "adam":
        beta1, beta2 = betas
        m = torch.zeros_like(params)
        v = torch.zeros_like(params)

    for step in range(1, steps + 1):
        params_before = params.clone()
        loss, true_grad = loss_grad_fn(params_before)
        observed_grad = true_grad + noise_sequence[step - 1]

        if optimizer_name == "adam":
            m = beta1 * m + (1 - beta1) * observed_grad
            v = beta2 * v + (1 - beta2) * observed_grad.pow(2)
            m_hat = m / (1 - beta1**step)
            v_hat = v / (1 - beta2**step)
            effective_lr = lr / (torch.sqrt(v_hat) + eps)
            delta = -effective_lr * m_hat
        elif optimizer_name == "sgd":
            effective_lr = torch.full_like(params_before, lr)
            delta = -lr * observed_grad
            m_hat = None
            v_hat = None
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        params = params_before + delta
        history.append(
            {
                "step": step,
                "loss": loss,
                "params_before": params_before.tolist(),
                "params_after": params.tolist(),
                "true_grad": true_grad.tolist(),
                "observed_grad": observed_grad.tolist(),
                "effective_lr": effective_lr.tolist(),
                "delta": delta.tolist(),
                "m_hat": None if m_hat is None else m_hat.tolist(),
                "v_hat": None if v_hat is None else v_hat.tolist(),
            }
        )

    return history


def summarize_final_loss(name, history):
    final_loss = history[-1]["loss"]
    print(f"{name} final loss = {final_loss:.6f}")


def run_one_dim_demo(num_steps):
    w = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [w],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )

    beta1, beta2 = optimizer.param_groups[0]["betas"]
    lr = optimizer.param_groups[0]["lr"]
    eps = optimizer.param_groups[0]["eps"]

    manual_m = 0.0
    manual_v = 0.0
    history = []
    all_match = True

    print(f"torch version: {torch.__version__}")
    print("1D demo: L(w) = 0.5 * (w - 4)^2, so grad = (w - 4)")
    print(f"initial w = {format_scalar(w.item())}\n")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = 0.5 * (w - 4.0) ** 2
        loss.backward()
        grad = w.grad.item()

        manual_m = beta1 * manual_m + (1 - beta1) * grad
        manual_v = beta2 * manual_v + (1 - beta2) * (grad**2)
        m_hat = manual_m / (1 - beta1**step)
        v_hat = manual_v / (1 - beta2**step)
        effective_lr = lr / (math.sqrt(v_hat) + eps)
        delta = -effective_lr * m_hat
        manual_next_w = w.item() + delta

        w_before = w.item()
        optimizer.step()
        state = optimizer.state[w]
        exp_avg = float(state["exp_avg"].item())
        exp_avg_sq = float(state["exp_avg_sq"].item())
        w_after = w.item()
        match = abs(w_after - manual_next_w) < 1e-12
        all_match = all_match and match

        record = {
            "step": step,
            "loss": float(loss.item()),
            "base_lr": lr,
            "w_before": w_before,
            "grad": grad,
            "grad_sq": grad**2,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "m_hat": m_hat,
            "v_hat": v_hat,
            "effective_lr": effective_lr,
            "delta": delta,
            "w_after": w_after,
            "manual_next_w": manual_next_w,
            "match": match,
        }
        history.append(record)

        if step <= min(5, num_steps):
            print(f"step {step}")
            print(f"  w_before         = {format_scalar(w_before)}")
            print(f"  grad             = {format_scalar(grad)}")
            print(f"  exp_avg (m_t)    = {format_scalar(exp_avg)}")
            print(f"  exp_avg_sq (v_t) = {format_scalar(exp_avg_sq)}")
            print(f"  m_hat            = {format_scalar(m_hat)}")
            print(f"  v_hat            = {format_scalar(v_hat)}")
            print(f"  effective_lr     = {format_scalar(effective_lr)}")
            print(f"  update delta     = {format_scalar(delta)}")
            print(f"  w_after          = {format_scalar(w_after)}")
            print(f"  manual_next_w    = {format_scalar(manual_next_w)}")
            print(f"  match?           = {match}\n")

    if num_steps > 5:
        print(f"... skipped detailed logs for steps 6-{num_steps}")
    print(f"all 1D manual updates matched torch? {all_match}\n")
    return history


def run_two_dim_demo(num_steps):
    params = torch.nn.Parameter(torch.tensor([2.0, 2.0], dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [params],
        lr=0.25,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )

    beta1, beta2 = optimizer.param_groups[0]["betas"]
    lr = optimizer.param_groups[0]["lr"]
    eps = optimizer.param_groups[0]["eps"]

    manual_m = torch.zeros(2, dtype=torch.float64)
    manual_v = torch.zeros(2, dtype=torch.float64)
    history = []
    initial_point = params.detach().clone()
    all_match = True

    print("2D demo: L(w1, w2) = 0.5 * (100 * w1^2 + w2^2)")
    print(f"initial point = ({initial_point[0].item():.4f}, {initial_point[1].item():.4f})")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = 0.5 * (100.0 * params[0] ** 2 + params[1] ** 2)
        loss.backward()
        grad = params.grad.detach().clone()

        manual_m = beta1 * manual_m + (1 - beta1) * grad
        manual_v = beta2 * manual_v + (1 - beta2) * grad.pow(2)
        m_hat = manual_m / (1 - beta1**step)
        v_hat = manual_v / (1 - beta2**step)
        effective_lr = lr / (torch.sqrt(v_hat) + eps)
        delta = -effective_lr * m_hat

        params_before = params.detach().clone()
        manual_next_params = params_before + delta
        optimizer.step()

        state = optimizer.state[params]
        params_after = params.detach().clone()
        exp_avg = state["exp_avg"].detach().clone()
        exp_avg_sq = state["exp_avg_sq"].detach().clone()
        match = torch.allclose(params_after, manual_next_params, atol=1e-12, rtol=0.0)
        all_match = all_match and bool(match)

        history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "params_before": params_before.tolist(),
                "params_after": params_after.tolist(),
                "grad": grad.tolist(),
                "exp_avg": exp_avg.tolist(),
                "exp_avg_sq": exp_avg_sq.tolist(),
                "m_hat": m_hat.tolist(),
                "v_hat": v_hat.tolist(),
                "effective_lr": effective_lr.tolist(),
                "delta": delta.tolist(),
                "match": bool(match),
            }
        )

    final_point = history[-1]["params_after"]
    print(f"final point   = ({final_point[0]:.4f}, {final_point[1]:.4f})")
    print(f"all 2D manual updates matched torch? {all_match}\n")
    return history


def save_one_dim_plot(history, output_dir):
    steps = [record["step"] for record in history]
    base_lr = history[0]["base_lr"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)

    axes[0].plot(steps, [record["grad"] for record in history], marker="o", label="grad")
    axes[0].plot(steps, [record["exp_avg"] for record in history], marker="o", label="m_t")
    axes[0].plot(steps, [record["m_hat"] for record in history], marker="o", label="m_hat")
    axes[0].set_title("1D Adam: gradient and first-moment smoothing")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, [record["grad_sq"] for record in history], marker="o", label="grad^2")
    axes[1].plot(steps, [record["exp_avg_sq"] for record in history], marker="o", label="v_t")
    axes[1].plot(steps, [record["v_hat"] for record in history], marker="o", label="v_hat")
    axes[1].set_title("1D Adam: second moment tracks gradient scale")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        steps,
        [record["effective_lr"] for record in history],
        marker="o",
        label="effective step size",
    )
    axes[2].axhline(
        y=base_lr,
        color="tab:gray",
        linestyle="--",
        linewidth=2,
        label=f"base learning rate = {base_lr}",
    )
    axes[2].plot(
        steps,
        [abs(record["delta"]) for record in history],
        marker="o",
        label="|update delta|",
    )
    axes[2].text(
        0.02,
        0.08,
        "effective step size = lr / (sqrt(v_hat) + eps)\nupdate delta = - effective step size * m_hat",
        transform=axes[2].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axes[2].set_title("1D Adam: base lr, adaptive step size, and actual parameter update")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("value")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    output_path = output_dir / "adam_1d_dynamics.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_two_dim_plot(history, sgd_reference, output_dir):
    initial_point = history[0]["params_before"]
    xs = [initial_point[0]] + [record["params_after"][0] for record in history]
    ys = [initial_point[1]] + [record["params_after"][1] for record in history]
    steps = [record["step"] for record in history]
    effective_lr_x = [record["effective_lr"][0] for record in history]
    effective_lr_y = [record["effective_lr"][1] for record in history]
    sgd_xs = [sgd_reference[0]["params_before"][0]] + [record["params_after"][0] for record in sgd_reference]
    sgd_ys = [sgd_reference[0]["params_before"][1]] + [record["params_after"][1] for record in sgd_reference]

    grid_x = torch.linspace(-2.2, 2.2, steps=250, dtype=torch.float64)
    grid_y = torch.linspace(-2.2, 2.2, steps=250, dtype=torch.float64)
    mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
    mesh_z = 0.5 * (100.0 * mesh_x**2 + mesh_y**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    contour_levels = [0.1, 0.5, 2, 8, 32, 128, 256]
    axes[0].contour(mesh_x.numpy(), mesh_y.numpy(), mesh_z.numpy(), levels=contour_levels, cmap="viridis")
    axes[0].plot(xs, ys, marker="o", markersize=3, linewidth=2, color="tab:orange", label="Adam path")
    axes[0].plot(
        sgd_xs,
        sgd_ys,
        marker="x",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        color="tab:blue",
        label="SGD path",
    )
    axes[0].scatter(xs[0], ys[0], color="tab:green", s=50, label="start")
    axes[0].scatter(xs[-1], ys[-1], color="tab:red", s=50, label="end")
    axes[0].set_title("2D bowl: Adam vs SGD on loss contours")
    axes[0].set_xlabel("w1 (high-curvature axis)")
    axes[0].set_ylabel("w2 (low-curvature axis)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, effective_lr_x, marker="o", label="effective step size for w1")
    axes[1].plot(steps, effective_lr_y, marker="o", label="effective step size for w2")
    axes[1].axhline(
        y=0.015,
        color="tab:gray",
        linestyle="--",
        linewidth=2,
        label="SGD fixed lr = 0.015",
    )
    axes[1].set_title("2D bowl: Adam adapts per dimension, SGD uses one lr")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = output_dir / "adam_2d_trajectory.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_loss_vs_step_plot(adam_history, sgd_history, output_dir):
    steps = [record["step"] for record in adam_history]
    adam_losses = [record["loss"] for record in adam_history]
    sgd_losses = [record["loss"] for record in sgd_history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].plot(steps, adam_losses, linewidth=2, label="Adam")
    axes[0].plot(steps, sgd_losses, linewidth=2, linestyle="--", label="SGD")
    axes[0].set_title("Deterministic 2D bowl: loss vs step")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(steps, adam_losses, linewidth=2, label="Adam")
    axes[1].semilogy(steps, sgd_losses, linewidth=2, linestyle="--", label="SGD")
    axes[1].set_title("Same curve on log scale")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss (log scale)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = output_dir / "adam_loss_vs_step.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_beta_sensitivity_plot(beta1_runs, beta2_runs, spike_step, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for beta1, history in beta1_runs.items():
        axes[0].semilogy(
            [record["step"] for record in history],
            [record["loss"] for record in history],
            linewidth=2,
            label=f"beta1 = {beta1}",
        )
    axes[0].set_title("beta1 sweep: momentum memory changes the loss curve")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for beta2, history in beta2_runs.items():
        axes[1].plot(
            [record["step"] for record in history],
            [record["effective_lr"][0] for record in history],
            linewidth=2,
            label=f"beta2 = {beta2}",
        )
    axes[1].axvline(spike_step, color="tab:red", linestyle="--", linewidth=2, label="gradient spike")
    axes[1].set_title("beta2 sweep: step size memory after a sudden gradient spike")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("effective step size")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = output_dir / "adam_beta_sensitivity.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_noisy_comparison_plot(noisy_paths, noisy_stats, output_dir):
    adam_path = noisy_paths["adam"]
    sgd_path = noisy_paths["sgd"]
    adam_mean = noisy_stats["adam"]["mean"]
    adam_q20 = noisy_stats["adam"]["q20"]
    adam_q80 = noisy_stats["adam"]["q80"]
    sgd_mean = noisy_stats["sgd"]["mean"]
    sgd_q20 = noisy_stats["sgd"]["q20"]
    sgd_q80 = noisy_stats["sgd"]["q80"]
    steps = list(range(1, len(adam_mean) + 1))

    grid_x = torch.linspace(-2.5, 2.5, steps=250, dtype=torch.float64)
    grid_y = torch.linspace(-2.5, 2.5, steps=250, dtype=torch.float64)
    mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing="xy")
    mesh_z = 0.5 * (100.0 * mesh_x**2 + mesh_y**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].contour(
        mesh_x.numpy(),
        mesh_y.numpy(),
        mesh_z.numpy(),
        levels=[0.1, 0.5, 2, 8, 32, 128, 256],
        cmap="viridis",
    )
    axes[0].plot(
        [adam_path[0]["params_before"][0]] + [record["params_after"][0] for record in adam_path],
        [adam_path[0]["params_before"][1]] + [record["params_after"][1] for record in adam_path],
        linewidth=2,
        marker="o",
        markersize=3,
        color="tab:orange",
        label="Adam path",
    )
    axes[0].plot(
        [sgd_path[0]["params_before"][0]] + [record["params_after"][0] for record in sgd_path],
        [sgd_path[0]["params_before"][1]] + [record["params_after"][1] for record in sgd_path],
        linewidth=1.5,
        linestyle="--",
        marker="x",
        markersize=4,
        color="tab:blue",
        label="SGD path",
    )
    axes[0].set_title("Shared noisy gradients: Adam path is steadier on the bowl")
    axes[0].set_xlabel("w1")
    axes[0].set_ylabel("w2")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(steps, adam_mean, linewidth=2, color="tab:orange", label="Adam mean loss")
    axes[1].fill_between(steps, adam_q20, adam_q80, color="tab:orange", alpha=0.2, label="Adam 20-80% band")
    axes[1].semilogy(steps, sgd_mean, linewidth=2, linestyle="--", color="tab:blue", label="SGD mean loss")
    axes[1].fill_between(steps, sgd_q20, sgd_q80, color="tab:blue", alpha=0.15, label="SGD 20-80% band")
    axes[1].set_title("Across 40 noisy runs: Adam keeps loss lower and more stable")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("true loss (log scale)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = output_dir / "adam_noisy_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_noisy_statistics(num_runs, num_steps, noise_std):
    adam_losses = []
    sgd_losses = []

    for seed in range(num_runs):
        noise_sequence = make_noise_sequence(num_steps, dim=2, noise_std=noise_std, seed=seed)
        adam_history = simulate_path(
            "adam",
            initial_params=[2.0, 2.0],
            loss_grad_fn=loss_and_grad_2d,
            steps=num_steps,
            lr=0.15,
            noise_sequence=noise_sequence,
        )
        sgd_history = simulate_path(
            "sgd",
            initial_params=[2.0, 2.0],
            loss_grad_fn=loss_and_grad_2d,
            steps=num_steps,
            lr=0.015,
            noise_sequence=noise_sequence,
        )
        adam_losses.append([record["loss"] for record in adam_history])
        sgd_losses.append([record["loss"] for record in sgd_history])

    adam_tensor = torch.tensor(adam_losses, dtype=torch.float64)
    sgd_tensor = torch.tensor(sgd_losses, dtype=torch.float64)
    return {
        "adam": {
            "mean": adam_tensor.mean(dim=0).tolist(),
            "q20": torch.quantile(adam_tensor, 0.2, dim=0).tolist(),
            "q80": torch.quantile(adam_tensor, 0.8, dim=0).tolist(),
        },
        "sgd": {
            "mean": sgd_tensor.mean(dim=0).tolist(),
            "q20": torch.quantile(sgd_tensor, 0.2, dim=0).tolist(),
            "q80": torch.quantile(sgd_tensor, 0.8, dim=0).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Adam learning demo with plots.")
    parser.add_argument("--steps-1d", type=int, default=20, help="Number of steps for the 1D demo.")
    parser.add_argument("--steps-2d", type=int, default=30, help="Number of steps for the 2D demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory used to save generated plots.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    one_dim_history = run_one_dim_demo(args.steps_1d)
    two_dim_history = run_two_dim_demo(args.steps_2d)

    deterministic_adam = simulate_path(
        "adam",
        initial_params=[2.0, 2.0],
        loss_grad_fn=loss_and_grad_2d,
        steps=40,
        lr=0.15,
    )
    deterministic_sgd = simulate_path(
        "sgd",
        initial_params=[2.0, 2.0],
        loss_grad_fn=loss_and_grad_2d,
        steps=40,
        lr=0.015,
    )

    beta1_runs = {
        beta1: simulate_path(
            "adam",
            initial_params=[1.0],
            loss_grad_fn=loss_and_grad_1d,
            steps=40,
            lr=0.1,
            betas=(beta1, 0.999),
        )
        for beta1 in (0.5, 0.9, 0.99)
    }
    spike_step = 10
    beta2_runs = {
        beta2: simulate_path(
            "adam",
            initial_params=[1.0],
            loss_grad_fn=loss_and_grad_1d,
            steps=30,
            lr=0.1,
            betas=(0.9, beta2),
            noise_sequence=make_spike_sequence(30, spike_step=spike_step, spike_value=40.0),
        )
        for beta2 in (0.5, 0.9, 0.99, 0.999)
    }

    noisy_sequence = make_noise_sequence(35, dim=2, noise_std=5.0, seed=7)
    noisy_paths = {
        "adam": simulate_path(
            "adam",
            initial_params=[2.0, 2.0],
            loss_grad_fn=loss_and_grad_2d,
            steps=35,
            lr=0.15,
            noise_sequence=noisy_sequence,
        ),
        "sgd": simulate_path(
            "sgd",
            initial_params=[2.0, 2.0],
            loss_grad_fn=loss_and_grad_2d,
            steps=35,
            lr=0.015,
            noise_sequence=noisy_sequence,
        ),
    }
    noisy_stats = build_noisy_statistics(num_runs=40, num_steps=60, noise_std=5.0)

    one_dim_plot = save_one_dim_plot(one_dim_history, args.output_dir)
    two_dim_plot = save_two_dim_plot(two_dim_history, deterministic_sgd[: args.steps_2d], args.output_dir)
    loss_plot = save_loss_vs_step_plot(deterministic_adam, deterministic_sgd, args.output_dir)
    beta_plot = save_beta_sensitivity_plot(beta1_runs, beta2_runs, spike_step, args.output_dir)
    noisy_plot = save_noisy_comparison_plot(noisy_paths, noisy_stats, args.output_dir)

    summarize_final_loss("deterministic Adam", deterministic_adam)
    summarize_final_loss("deterministic SGD", deterministic_sgd)
    print(f"noisy Adam mean final loss = {noisy_stats['adam']['mean'][-1]:.6f}")
    print(f"noisy SGD mean final loss = {noisy_stats['sgd']['mean'][-1]:.6f}\n")

    print(f"saved 1D plot to {one_dim_plot}")
    print(f"saved 2D plot to {two_dim_plot}")
    print(f"saved loss plot to {loss_plot}")
    print(f"saved beta sweep plot to {beta_plot}")
    print(f"saved noisy comparison plot to {noisy_plot}")


if __name__ == "__main__":
    main()
