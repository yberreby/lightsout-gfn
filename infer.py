import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Literal

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

import config
from core import (
    TrainableStateContainer,
    toggle_tile,
    is_solved,
    generate_all_boards,
    greedy_solve_board,
)
from models import PolicyNet

N = config.N
FLAT_DIM = config.FLAT_DIM
QUANTITATIVE_EVAL_MAX_STEPS = config.EVAL_MAX_STEPS
VISUALIZATION_MAX_STEPS = config.EVAL_MAX_STEPS
SimMode = Literal['greedy', 'stochastic', 'retry']


def int_to_board(i: int, n: int = N) -> np.ndarray:
    dim = n * n
    binary_repr = bin(i)[2:].zfill(dim)
    board = np.array([int(bit) for bit in reversed(binary_repr)], dtype=np.int8)
    return board


def generate_truly_random_board(
    key: jax.random.PRNGKey, n: int = N
) -> Tuple[jax.random.PRNGKey, np.ndarray]:
    num_states = 2**(n * n)
    current_key = key
    attempts = 0
    while attempts < 50:
        new_key, sample_key = jax.random.split(current_key)
        random_int = jax.random.randint(
            sample_key, (), minval=1, maxval=num_states
        )
        board = int_to_board(random_int.item(), n)
        if not np.all(board == 0):
            return new_key, board
        current_key = new_key
        attempts += 1
    print("[Warning] Failed to generate non-zero board, using all-ones.")
    return new_key, np.ones((n * n,), dtype=np.int8)


def simulate_greedy_solve_for_vis(
    pf_model: PolicyNet, start_board_np: np.ndarray, n: int, max_steps: int
) -> Tuple[List[np.ndarray], List[int], bool, int]:
    history = [start_board_np]
    actions = []
    current_jax = jnp.asarray(start_board_np)
    steps = 0
    solved = is_solved(current_jax).item()

    for i in range(max_steps):
        if solved:
            steps = i
            break
        logits = pf_model(current_jax.astype(jnp.float32))
        action = jnp.argmax(logits).astype(jnp.int32)
        action_item = int(action.item())
        next_jax = toggle_tile(current_jax, action)
        history.append(np.array(next_jax))
        actions.append(action_item)
        current_jax = next_jax
        solved = is_solved(current_jax).item()
        steps = i + 1
    else:
        steps = max_steps
        solved = is_solved(current_jax).item()

    if solved and (not actions or actions[-1] != -1):
        actions.append(-1)
    return history, actions, solved, steps


def simulate_stochastic_solve_for_vis(
    key: jax.random.PRNGKey,
    pf_model: PolicyNet,
    start_board_np: np.ndarray,
    n: int,
    max_steps: int,
) -> Tuple[jax.random.PRNGKey, List[np.ndarray], List[int], bool, int]:
    history = [start_board_np]
    actions = []
    current_jax = jnp.asarray(start_board_np)
    steps = 0
    solved = is_solved(current_jax).item()
    current_key = key

    for i in range(max_steps):
        if solved:
            steps = i
            break
        step_key, current_key = jax.random.split(current_key)
        logits = pf_model(current_jax.astype(jnp.float32))
        action = jax.random.categorical(step_key, logits).astype(jnp.int32)
        action_item = int(action.item())
        next_jax = toggle_tile(current_jax, action)
        history.append(np.array(next_jax))
        actions.append(action_item)
        current_jax = next_jax
        solved = is_solved(current_jax).item()
        steps = i + 1
    else:
        steps = max_steps
        solved = is_solved(current_jax).item()

    if solved and (not actions or actions[-1] != -1):
        actions.append(-1)
    return current_key, history, actions, solved, steps


def run_simulation_for_vis(
    key: jax.random.PRNGKey,
    pf_model: PolicyNet,
    n: int,
    max_steps: int,
    mode: SimMode,
    retries: int,
) -> Tuple[jax.random.PRNGKey, List[np.ndarray], List[int], bool, int]:
    sim_key, board_key = jax.random.split(key)
    next_key, start_board = generate_truly_random_board(board_key, n)

    if mode == 'greedy':
        hist, actions, solved, steps = simulate_greedy_solve_for_vis(
            pf_model, start_board, n, max_steps
        )
        return next_key, hist, actions, solved, steps
    elif mode == 'stochastic':
        sim_key, hist, actions, solved, steps = simulate_stochastic_solve_for_vis(
            sim_key, pf_model, start_board, n, max_steps
        )
        return next_key, hist, actions, solved, steps
    elif mode == 'retry':
        current_sim_key = sim_key
        for attempt in range(retries):
            attempt_key, current_sim_key = jax.random.split(current_sim_key)
            _, hist, actions, solved, steps = simulate_stochastic_solve_for_vis(
                attempt_key, pf_model, start_board, n, max_steps
            )
            if solved:
                print(f" Solution found via retry on attempt {attempt+1}.")
                return next_key, hist, actions, solved, steps
        print(f" No solution found after {retries} stochastic retries.")
        return next_key, hist, actions, solved, steps
    else:
        raise ValueError(f"Unknown simulation mode: {mode}")


def evaluate_model_on_all_states(
    container: TrainableStateContainer, n: int
) -> Tuple[float, int, int]:
    print("\n--- Starting Quantitative Evaluation (Greedy) ---")
    eval_max_steps = config.EVAL_MAX_STEPS
    print(f"[INFO] Max steps: {eval_max_steps}")
    all_boards_np = generate_all_boards(n)
    total_boards = all_boards_np.shape[0]
    expected_boards = 2**(n * n)
    print(f"[Sanity Check] Generated {total_boards} boards.")
    if total_boards != expected_boards:
        print(f"[ERROR] Expected {expected_boards}")
        return -1.0, -1, -1
    if total_boards == 0:
        print("[Warning] No boards")
        return 0.0, 0, 0
    vmap_greedy_solve = jax.vmap(greedy_solve_board, in_axes=(None, 0))
    batch_size = 512
    num_solved = 0
    start_time = time.time()
    print(f"Evaluating on {total_boards} states...")
    for i in range(0, total_boards, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, total_boards)
        batch_jnp = jnp.asarray(all_boards_np[batch_start:batch_end])
        flags_batch, _ = vmap_greedy_solve(container, batch_jnp)
        jax.block_until_ready(flags_batch)
        num_solved += jnp.sum(flags_batch).item()
    elapsed = time.time() - start_time
    print(f"Evaluation took {elapsed:.2f}s.")
    accuracy = (
        (float(num_solved) / total_boards) * 100.0 if total_boards > 0 else 0.0
    )
    print(
        f"\nQuantitative Result: Solved {num_solved}/{total_boards} ({accuracy:.2f}%)"
    )
    print("--- End Evaluation ---\n")
    return accuracy, num_solved, total_boards


gui_plot_state: Dict[str, Any] = {
    "board_history": [],
    "action_history": [],
    "current_vis_step": 0,
    "total_steps_taken": 0,
    "is_solved_final": False,
    "fig": None,
    "ax": None,
    "im": None,
    "n": N,
    "pf_model": None,
    "key": None,
    "overall_accuracy": -1.0,
    "overall_solved_count": -1,
    "overall_total_count": -1,
    "sim_mode": 'greedy',
    "retries": 10,
}


def gui_on_key(event):
    global gui_plot_state
    state = gui_plot_state
    n = state["n"]
    current = state["current_vis_step"]
    total = len(state["board_history"]) - 1
    if not state["board_history"] or state["pf_model"] is None:
        return
    if event.key == 'right':
        state["current_vis_step"] = min(current + 1, total)
    elif event.key == 'left':
        state["current_vis_step"] = max(current - 1, 0)
    elif event.key in ['up', 'down', 'n']:
        print(
            f"Generating new board (Mode: {state['sim_mode']}, Retries: {state['retries']})..."
        )
        next_key, hist, acts, solved, steps = run_simulation_for_vis(
            state["key"],
            state["pf_model"],
            n,
            VISUALIZATION_MAX_STEPS,
            state["sim_mode"],
            state["retries"],
        )
        state.update({
            "key": next_key,
            "board_history": hist,
            "action_history": acts,
            "current_vis_step": 0,
            "is_solved_final": solved,
            "total_steps_taken": steps,
        })
        print(f" Sim: {'Solved' if solved else 'Failed'} in {steps} steps.")
    elif event.key == 'escape':
        if state["fig"]:
            plt.close(state["fig"])
            print("GUI closed.")
            return
    else:
        return
    gui_update_plot()


def gui_update_plot():
    global gui_plot_state
    state = gui_plot_state
    if state["fig"] is None or not state["board_history"] or state["ax"] is None:
        return
    step = state["current_vis_step"]
    board = state["board_history"][step]
    total_hist = len(state["board_history"]) - 1
    n = state["n"]
    acc, sc, tc = (
        state["overall_accuracy"],
        state["overall_solved_count"],
        state["overall_total_count"],
    )

    if state["im"] is None:
        cmap = mcolors.ListedColormap(['black', 'yellow'])
        state["im"] = state["ax"].imshow(
            board.reshape(n, n),
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation='nearest',
        )
        state["ax"].set_xticks(np.arange(-0.5, n, 1), minor=True)
        state["ax"].set_yticks(np.arange(-0.5, n, 1), minor=True)
        state["ax"].grid(
            which='minor', color='grey', linestyle='-', linewidth=1
        )
        state["ax"].tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
    else:
        state["im"].set_data(board.reshape(n, n))

    sim_m = state['sim_mode'].capitalize()
    rtr = f" (Retries:{state['retries']})" if state['sim_mode'] == 'retry' else ""
    title = f"Mode:{sim_m}{rtr} | Acc:{acc:.1f}% | Vis:{step}/{total_hist}"
    if step == 0:
        title += " (Initial)"
    elif step > 0 and step <= len(state["action_history"]):
        action = state["action_history"][step - 1]
        if action != -1:
            row, col = divmod(action, n)
            title += f" | Act:{action}(@{row},{col})"
    is_last = step == total_hist
    sim_solved = state["is_solved_final"]
    sim_steps = state["total_steps_taken"]
    if is_last:
        title += f" | Final:{'Solved' if sim_solved else 'Failed'} ({sim_steps} steps)"
    elif (
        step > 0
        and step <= len(state["action_history"])
        and state["action_history"][step - 1] == -1
    ):
        title += f" | Solved ({sim_steps} steps total)"
    state["ax"].set_title(title, fontsize=9)
    state["fig"].canvas.draw_idle()


def run_gui_matplotlib(
    pf_model, n, key, accuracy, solved_count, total_count, sim_mode, retries
):
    global gui_plot_state
    print(f"Initializing GUI (Mode: {sim_mode}, Retries: {retries})...")
    next_key, hist, acts, solved, steps = run_simulation_for_vis(
        key, pf_model, n, VISUALIZATION_MAX_STEPS, sim_mode, retries
    )
    print(f" Initial sim: {'Solved' if solved else 'Failed'} in {steps} steps.")
    gui_plot_state.update({
        "board_history": hist,
        "action_history": acts,
        "current_vis_step": 0,
        "is_solved_final": solved,
        "total_steps_taken": steps,
        "n": n,
        "pf_model": pf_model,
        "key": next_key,
        "overall_accuracy": accuracy,
        "overall_solved_count": solved_count,
        "overall_total_count": total_count,
        "sim_mode": sim_mode,
        "retries": retries,
    })
    if (
        gui_plot_state["fig"] is None
        or not plt.fignum_exists(gui_plot_state["fig"].number)
    ):
        gui_plot_state["fig"], gui_plot_state["ax"] = plt.subplots(figsize=(5, 5.8))
        gui_plot_state["fig"].canvas.mpl_connect(
            'key_press_event', gui_on_key
        )
        gui_plot_state["im"] = None
        plt.subplots_adjust(top=0.9)
    else:
        gui_plot_state["ax"].clear()
        gui_plot_state["im"] = None
    print("GUI Ready. Arrows: Navigate steps. N/Up/Down: New board. Esc: Close.")
    gui_update_plot()
    plt.show(block=True)


def plot_board_for_gif(ax, board, title, n):
    ax.clear()
    cmap = mcolors.ListedColormap(['black', 'yellow'])
    ax.imshow(
        board.reshape(n, n),
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation='nearest',
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='grey', ls='-', lw=1)
    ax.tick_params(
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def create_gif(
    filename, trajectories, solved_flags, solved_steps, n, overall_pct, sim_mode_str
):
    num_boards = len(trajectories)
    max_len = max(len(t) for t in trajectories) if trajectories else 0
    if num_boards == 0 or max_len == 0:
        print("No trajectories for GIF.")
        return
    ncols = min(num_boards, 5)
    nrows = (num_boards + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 3.5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()
    [ax.axis('off') for ax in axes_flat[num_boards:]]
    fig.suptitle(
        f"LightsOut {n}x{n} | Mode: {sim_mode_str} | Acc: {overall_pct:.1f}%",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_gif_frame(f_idx):
        print(f" GIF frame {f_idx+1}/{max_len}", end='\r')
        artists = []
        for i in range(num_boards):
            ax = axes_flat[i]
            traj = trajectories[i]
            solved = solved_flags[i]
            steps = solved_steps[i]
            curr_step = min(f_idx, len(traj) - 1)
            board = traj[curr_step]
            title = f"Sample {i+1} | Step {curr_step}"
            if f_idx >= len(traj) - 1:
                title += f"\n{'Solved' if solved else 'Failed'} ({steps} steps)"
            plot_board_for_gif(ax, board, title, n)
            artists.extend(ax.images)
            artists.extend(ax.texts)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update_gif_frame,
        frames=max_len,
        interval=350,
        blit=False,
        repeat_delay=1500,
    )
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving GIF ({num_boards} samples, {max_len} steps) to {out_path}...")
    try:
        ani.save(out_path, writer='pillow', fps=3)
        print(" GIF saved.")
    except Exception as e:
        print(f"\n[ERROR] Saving GIF: {e}")
        traceback.print_exc()
    finally:
        plt.close(fig)


def run_gif_generation(
    pf_model, n, num_samples, gif_filename, key, overall_accuracy, sim_mode, retries
):
    sim_mode_str = sim_mode.capitalize() + (
        f" (Retries:{retries})" if sim_mode == 'retry' else ""
    )
    print(
        f"\n--- Generating {num_samples} samples for GIF (Mode: {sim_mode_str}) ---"
    )
    trajectories, flags, steps_list = [], [], []
    current_key = key
    for i in range(num_samples):
        current_key, traj, _, solved, steps = run_simulation_for_vis(
            current_key, pf_model, n, VISUALIZATION_MAX_STEPS, sim_mode, retries
        )
        trajectories.append(traj)
        flags.append(solved)
        steps_list.append(steps)
        print(f"  Sample {i+1}: {'Solved' if solved else 'Failed'} in {steps} steps.")
    if not trajectories:
        print("[ERROR] No trajectories generated.")
        return
    create_gif(
        gif_filename,
        trajectories,
        flags,
        steps_list,
        n,
        overall_accuracy,
        sim_mode_str,
    )


def main():
    parser = argparse.ArgumentParser(description="Infer/Visualize LightsOut GFN.")
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to Orbax checkpoint dir/file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['gui', 'gif', 'eval'],
        default='gui',
        help="Run mode.",
    )
    parser.add_argument(
        "--sim_mode",
        type=str,
        choices=['greedy', 'stochastic', 'retry'],
        default='greedy',
        help="Simulation mode for GUI/GIF.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Num stochastic retries if sim_mode='retry'.",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Num samples for GIF."
    )
    parser.add_argument(
        "--gif_name", type=str, default="lightsout_solve.gif"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path).resolve()
    ckpt_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    if not ckpt_dir.is_dir():
        print(
            f"[ERROR] Checkpoint dir '{ckpt_dir}' not found.", file=sys.stderr
        )
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_dir}")
    container: Optional[TrainableStateContainer] = None
    try:
        def create_template():
            k = jax.random.PRNGKey(0)
            r = nnx.Rngs(params=k)
            pf = PolicyNet(
                FLAT_DIM, config.HIDDEN_DIM, config.ACTION_DIM, rngs=r
            )
            pb = PolicyNet(
                FLAT_DIM, config.HIDDEN_DIM, config.ACTION_DIM, rngs=r
            )
            z = nnx.Param(jnp.array(0.0))
            return TrainableStateContainer(pf, pb, z)

        graphdef, abstract_state = nnx.split(nnx.eval_shape(create_template))
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(ckpt_dir, abstract_state)
        checkpointer.wait_until_finished()
        container = nnx.merge(graphdef, restored)
        print(" Checkpoint loaded.")
        assert isinstance(container, TrainableStateContainer)
        print(f" Loaded Z: {container.z_param.value.item():.4f}")
    except Exception as e:
        print(f"[ERROR] Load failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    accuracy, solved_count, total_count = evaluate_model_on_all_states(
        container, N
    )
    if accuracy < 0:
        print("[ERROR] Evaluation failed.")
        sys.exit(1)

    sim_key = jax.random.PRNGKey(int(time.time()))
    if args.mode == 'eval':
        print("Mode 'eval': Done.")
    elif args.mode == 'gif':
        run_gif_generation(
            container.pf_model,
            N,
            args.samples,
            args.gif_name,
            sim_key,
            accuracy,
            args.sim_mode,
            args.retries,
        )
    else:  # 'gui'
        run_gui_matplotlib(
            container.pf_model,
            N,
            sim_key,
            accuracy,
            solved_count,
            total_count,
            args.sim_mode,
            args.retries,
        )

    print("\nInference script finished.")


if __name__ == "__main__":
    main()
