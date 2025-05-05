import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def is_controllable(A, B):
    n = A.shape[0]
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack((ctrb, np.linalg.matrix_power(A, i) @ B))
    return np.linalg.matrix_rank(ctrb) == n


def simulate_dynamics_discrete(A, B, x0, u_traj, T):
    """Simulate ẋ = A·x + B·u using forward Euler method."""
    steps = u_traj.shape[1]
    dt = T / steps
    x = np.zeros((A.shape[0], steps))
    x[:, 0] = x0
    for i in range(steps - 1):
        x[:, i + 1] = x[:, i] + dt * (A @ x[:, i] + B @ u_traj[:, i])
    return x


def cost_function(u_flat, A, B, x0, T, steps, rho=0.0):
    """Compute cost functional with optional control penalty and optional control effort penalty."""
    _, m = B.shape
    u_traj = u_flat.reshape(m, steps)
    x = simulate_dynamics_discrete(A, B, x0, u_traj, T)
    dt = T / steps
    x0_vals = x[0, :]
    x1_vals = x[1, :] if x.shape[0] > 1 else np.zeros_like(x0_vals)
    state_cost = np.sum((x0_vals - 6) ** 2 - x1_vals) * dt
    control_cost = rho * np.sum(u_traj ** 2) * dt
    return state_cost + control_cost


def solve_optimal_control_direct(A, B, x0, T, lb=-10, ub=10, rho=0.0, coarse_factor=1, steps=200):
    """Solve the optimal control problem with optional coarse warm-start."""
    n, m = B.shape
    bounds = [(lb, ub)] * (m * steps)

    # Coarse optimization
    if coarse_factor != 1:
        coarse_steps = steps // coarse_factor
        u_coarse0 = np.zeros((m, coarse_steps))

        res_coarse = minimize(
            cost_function,
            u_coarse0.flatten(),
            args=(A, B, x0, T, coarse_steps, rho),
            bounds=[(lb, ub)] * (m * coarse_steps),
            method='L-BFGS-B',
            options={'maxiter': 200}
        )

        if res_coarse.success:
            u_coarse_opt = res_coarse.x.reshape(m, coarse_steps)
            u_fine0 = np.repeat(u_coarse_opt, coarse_factor, axis=1)[:, :steps]
        else:
            print("Warning: Coarse optimization failed. Falling back to zero init.")
            u_fine0 = np.zeros((m, steps))
    else:
        u_fine0 = np.zeros((m, steps))

    # Fine optimization
    res = minimize(
        cost_function,
        u_fine0.flatten(),
        args=(A, B, x0, T, steps, rho),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 200}
    )

    u_opt = res.x.reshape(m, steps)
    x = simulate_dynamics_discrete(A, B, x0, u_opt, T)
    t_grid = np.linspace(0, T, steps)
    return t_grid, x, u_opt, res


def plot_results(t, x, u):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i in range(x.shape[0]):
        axs[0].plot(t, x[i], label=f"x{i}")
    axs[0].set_ylabel("State x(t)")
    axs[0].legend()
    axs[0].grid(True)

    for j in range(u.shape[0]):
        axs[1].plot(t, u[j], label=f"u{j}")
    axs[1].set_ylabel("Control u(t)")
    axs[1].set_xlabel("Time t")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle("Optimal State and Control Trajectories")
    plt.tight_layout()
    plt.show()


def main():
    config = {
        "T": 10.0,
        "n": 6,
        "m": 5,
        "steps": 100,
        "lb": -5,
        "ub": 5,
        "rho": 0.0,
        "coarse_factor": 5,
        "seed": 0
    }

    np.random.seed(config["seed"])
    A = np.random.randn(config["n"], config["n"]) * 0.3
    B = np.random.randn(config["n"], config["m"]) * 0.5
    x0 = np.random.randn(config["n"]) * 5

    if not is_controllable(A, B):
        raise ValueError("System (A, B) is not controllable.")

    t, x, u, res = solve_optimal_control_direct(
        A, B, x0,
        T=config["T"],
        lb=config["lb"],
        ub=config["ub"],
        rho=config["rho"],
        steps=config["steps"],
        coarse_factor=config["coarse_factor"],
    )

    print(f"Optimal cost: {res.fun:.4f}")
    plot_results(t, x, u)


if __name__ == "__main__":
    main()


# n = 4
# m = 4
# A = np.array([[0.3, 0.0, 0.0, 0.3],
#               [0.0, -0.1, 0.15, 0.0],
#               [0.0, 0.2, 0.05, 0.02],
#               [0.4, 0.0, 0.01, -0.3]])
# B = np.eye(4)
# x0 = np.array([2.0, -1.0, 0.5, -0.3])