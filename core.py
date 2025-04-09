import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Tuple, List
from flax import nnx

import config
from models import PolicyNet

class TrainableStateContainer(nnx.Module):
    pf_model: PolicyNet
    pb_model: PolicyNet
    z_param: nnx.Param

@jax.jit
def get_neighbors(index: int) -> jnp.ndarray:
    row, col = jnp.divmod(index, config.N)
    rel_pos = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
    pos = jnp.stack([row, col])[None, :] + rel_pos
    pos = jnp.clip(pos, 0, config.N - 1)
    return (pos[:, 0] * config.N + pos[:, 1]).astype(jnp.int32)


@jax.jit
def toggle_tile(board: jnp.ndarray, action_idx: int) -> jnp.ndarray:
    return board.at[get_neighbors(action_idx)].add(1) % 2


@jax.jit
def is_solved(board: jnp.ndarray) -> bool:
    return jnp.all(board == 0, axis=-1)


# FIXME: Terrible for bigger than 3x3.
def generate_all_boards(n: int) -> jnp.ndarray:
    dim = n * n
    num_states = 2**dim
    all_boards = []
    for i in range(num_states):
        binary_repr = bin(i)[2:].zfill(dim)
        board = jnp.array([int(bit) for bit in binary_repr], dtype=jnp.int8)
        all_boards.append(board)
    return jnp.stack(all_boards)


@partial(jax.jit, static_argnames=["n"])
def _get_neighbors_eval(index: int, n: int) -> jnp.ndarray:
    row, col = jnp.divmod(index, n)
    rel_pos = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
    pos = jnp.stack([row, col])[None, :] + rel_pos
    pos = jnp.clip(pos, 0, n - 1)
    return (pos[:, 0] * n + pos[:, 1]).astype(jnp.int32)


@partial(jax.jit, static_argnames=["n"])
def _toggle_tile_eval(board: jnp.ndarray, action_idx: int, n: int) -> jnp.ndarray:
    neighbor_indices = _get_neighbors_eval(action_idx, n)
    return board.at[neighbor_indices].add(1) % 2


@partial(jax.jit, static_argnames=["n", "n_actions"])
def apply_k_random_actions_eval(
    key: jnp.ndarray, n_actions: int, n: int
) -> jnp.ndarray:
    flat_dim = config.FLAT_DIM
    action_dim = config.ACTION_DIM
    start_board = jnp.zeros((flat_dim,), dtype=jnp.int8)
    actions_to_apply = jax.random.randint(key, (n_actions,), 0, action_dim)
    _toggle_tile_static_n = partial(_toggle_tile_eval, n=n)
    final_board, _ = jax.lax.scan(
        lambda board, action: (_toggle_tile_static_n(board, action), None),
        start_board,
        actions_to_apply,
    )
    return final_board.astype(jnp.int8)


@partial(nnx.jit)
def sample_trajectory(
    key: jnp.ndarray, container: TrainableStateContainer, start_board: jnp.ndarray
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    max_traj_len = config.MAX_TRAJECTORY_LEN
    max_theo_steps = config.MAX_THEORETICAL_STEPS
    pf_model = container.pf_model

    @nnx.jit
    def _sample_action_mlp(
        key: jnp.ndarray, board: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits = pf_model(board)
        valid_action_mask = jnp.where(is_solved(board), -jnp.inf, 0.0)
        masked_logits = logits + valid_action_mask
        action = jax.random.categorical(key, masked_logits)
        log_prob = jax.nn.log_softmax(masked_logits)[action]
        return action.astype(jnp.int32), log_prob.astype(jnp.float32)

    def step_fn(carry, _):
        key, current_board, solved_flag = carry
        action_key, next_key = jax.random.split(key)
        state_before_action = current_board
        action_ns, logp_ns = _sample_action_mlp(action_key, current_board)
        no_op_action, no_op_logp = jnp.int32(-1), jnp.float32(0.0)
        action, pf_log_prob = jax.lax.cond(
            solved_flag,
            lambda: (no_op_action, no_op_logp),
            lambda: (action_ns, logp_ns),
        )
        next_board = jax.lax.cond(
            action == no_op_action,
            lambda b: b,
            lambda b: toggle_tile(b, action),
            current_board,
        ).astype(jnp.int8)
        next_solved_flag = jnp.logical_or(solved_flag, is_solved(next_board))
        step_output = (state_before_action, action, pf_log_prob)
        next_carry = (next_key, next_board, next_solved_flag)
        return next_carry, step_output

    init_carry = (key, start_board.astype(jnp.int8), is_solved(start_board))
    _, (states_before, actions, pf_log_probs) = jax.lax.scan(
        step_fn, init_carry, None, length=max_traj_len
    )

    def get_next_state(board, action):
        next_b = jax.lax.cond(
            action == -1, lambda b: b, lambda b: toggle_tile(b, action), board
        )
        return next_b, next_b

    _, all_states_post_action = jax.lax.scan(get_next_state, start_board, actions)
    full_states_trajectory = jnp.concatenate(
        [start_board[None, :], all_states_post_action], axis=0
    ).astype(jnp.int8)
    solved_mask_trajectory = jax.vmap(is_solved)(full_states_trajectory)
    first_solved_state_idx = jnp.argmax(solved_mask_trajectory)
    was_solved = jnp.any(solved_mask_trajectory)
    solved_steps = jax.lax.cond(
        was_solved, lambda idx: idx, lambda idx: max_traj_len, first_solved_state_idx
    )
    solved_steps = jnp.minimum(solved_steps, max_traj_len)
    mask = jnp.arange(max_traj_len) < solved_steps
    log_reward = jax.lax.cond(
        was_solved,
        lambda s: config.REWARD_EXP_MULTIPLIER
        * (max_theo_steps - s.astype(jnp.float32)),
        lambda s: jnp.float32(config.MIN_REWARD_LOG),
        solved_steps,
    )
    return full_states_trajectory, actions, pf_log_probs, log_reward, mask, solved_steps


def batch_sample_trajectories(
    key: jnp.ndarray, container: TrainableStateContainer, start_boards: jnp.ndarray
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    keys = jax.random.split(key, start_boards.shape[0])
    batch_fn = jax.vmap(sample_trajectory, in_axes=(0, None, 0))
    return batch_fn(keys, container, start_boards)

@partial(nnx.jit)
def compute_tb_loss(
    container: TrainableStateContainer, batch_data: Tuple
) -> jnp.ndarray:
    pf_model = container.pf_model
    pb_model = container.pb_model
    z_param = container.z_param

    states, actions, _, log_rewards, masks, _ = batch_data
    B, T_plus_1, F = states.shape
    T = config.MAX_TRAJECTORY_LEN
    action_dim = config.ACTION_DIM

    @nnx.jit
    def get_log_probs(
        model: PolicyNet,
        input_states: jnp.ndarray,
        actions_taken: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> jnp.ndarray:
        B_in, T_slice, F_in = input_states.shape
        input_states_flat = input_states.reshape(B_in * T_slice, F_in)
        logits = model(input_states_flat)
        logits = logits.reshape(B_in, T_slice, action_dim)
        all_log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_probs_taken = jnp.take_along_axis(
            all_log_probs, actions_taken[:, :, None], axis=2
        ).squeeze(-1)
        return jnp.sum(log_probs_taken * masks, axis=1)

    sum_pf_log_probs = get_log_probs(pf_model, states[:, :T, :], actions, masks)
    sum_pb_log_probs = get_log_probs(pb_model, states[:, 1 : T + 1, :], actions, masks)

    z_param_casted, sum_pf_log_probs, log_rewards, sum_pb_log_probs = jax.tree.map(
        lambda x: x.astype(jnp.float32),
        (z_param, sum_pf_log_probs, log_rewards, sum_pb_log_probs),
    )
    tb_terms = z_param_casted + sum_pf_log_probs - log_rewards - sum_pb_log_probs
    loss = jnp.mean(tb_terms**2)
    return loss


@partial(nnx.jit, static_argnames=("all_boards_shape",))
def train_step(
    container: TrainableStateContainer,
    optimizer: nnx.Optimizer,
    key: jnp.ndarray,
    all_boards: jnp.ndarray,
    all_boards_shape: Tuple,
) -> Tuple[Dict, jnp.ndarray]:
    num_all_boards = all_boards_shape[0]
    board_key, sample_key, next_key = jax.random.split(key, 3)

    board_indices = jax.random.choice(
        board_key, num_all_boards, shape=(config.BATCH_SIZE,), replace=True
    )
    start_boards = all_boards[board_indices]
    batch_data = batch_sample_trajectories(sample_key, container, start_boards)

    def loss_fn(container_arg: TrainableStateContainer):
        return compute_tb_loss(container_arg, batch_data)

    # Calculate gradients w.r.t the container module
    loss_val, grads = nnx.value_and_grad(loss_fn)(container)

    # Apply gradients using the optimizer.
    # This updates the state of 'container' and 'optimizer' passed as arguments implicitly.
    optimizer.update(grads)

    loss_is_finite = jnp.isfinite(loss_val)
    # If loss is not finite, the updates might have still happened with NaNs.
    # Optax often includes checks, but relying on that implicitly isn't robust.
    # For simplicity matching the tutorial, we don't explicitly prevent the update here,
    # but we report the flag. The main loop should check this flag.

    # Extract metrics
    _, _, _, log_rewards, _, solved_steps = batch_data
    max_traj_len = config.MAX_TRAJECTORY_LEN
    solved_mask = solved_steps < max_traj_len
    solved_count = jnp.sum(solved_mask)
    solved_pct = jnp.mean(solved_mask) * 100
    avg_steps_solved = jnp.sum(jnp.where(solved_mask, solved_steps, 0)) / jnp.maximum(
        1e-6, solved_count
    )
    reported_avg_steps = jnp.where(
        solved_count > 0, avg_steps_solved, max_traj_len
    ).astype(float)

    metrics = {
        "loss": loss_val,
        "solved_pct": solved_pct,
        "avg_steps_solved": reported_avg_steps,
        "avg_log_reward": jnp.mean(log_rewards),
        "loss_finite": loss_is_finite,
    }

    return metrics, next_key


@partial(nnx.jit)
def greedy_solve_board(
    container: TrainableStateContainer, start_board: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    eval_max_steps = config.EVAL_MAX_STEPS
    start_solved = is_solved(start_board)
    pf_model = container.pf_model
    init_carry = (
        start_board.astype(jnp.int8),
        start_solved,
        jnp.int32(eval_max_steps + 1),
        jnp.int32(0),
    )

    def step_fn(carry, _):
        current_board, already_solved, solve_step_found, step_idx = carry

        def select_action_branch():
            logits = pf_model(current_board)
            return jnp.argmax(logits).astype(jnp.int32)

        action = jax.lax.cond(
            already_solved, lambda: jnp.int32(-1), select_action_branch
        )
        next_board = jax.lax.cond(
            already_solved,
            lambda: current_board,
            lambda: toggle_tile(current_board, action),
        ).astype(jnp.int8)
        currently_solved = is_solved(next_board)
        newly_solved_this_step = jnp.logical_and(
            currently_solved, jnp.logical_not(already_solved)
        )
        current_step_val = step_idx + 1
        next_solve_step_found = jnp.where(
            newly_solved_this_step, current_step_val, solve_step_found
        )
        next_already_solved = jnp.logical_or(already_solved, currently_solved)
        next_carry = (
            next_board,
            next_already_solved,
            next_solve_step_found,
            step_idx + 1,
        )
        return next_carry, None

    final_carry, _ = jax.lax.scan(step_fn, init_carry, None, length=eval_max_steps)
    final_first_solve_step = final_carry[2]
    solved_step = jnp.where(start_solved, 0, final_first_solve_step).astype(jnp.int32)
    solved_flag_final = solved_step <= eval_max_steps
    return solved_flag_final, solved_step


def evaluate(
    container: TrainableStateContainer,
    k_values: List[int],
    num_samples_per_k: int,
    eval_key: jnp.ndarray,
) -> Tuple[Dict, jnp.ndarray]:
    all_metrics = {}
    total_solved = 0
    total_samples = 0
    n = config.N
    _vmap_apply_k_eval = jax.vmap(
        partial(apply_k_random_actions_eval, n=n), in_axes=(0, None)
    )
    _vmap_greedy_solve = jax.vmap(
        greedy_solve_board, in_axes=(None, 0)
    )  # Pass container via None

    for k in k_values:
        k_key, eval_key = jax.random.split(eval_key)
        eval_board_keys = jax.random.split(k_key, num_samples_per_k)
        eval_boards = _vmap_apply_k_eval(eval_board_keys, k)
        initial_unsolved_mask = ~is_solved(eval_boards)
        num_trivial = num_samples_per_k - jnp.sum(initial_unsolved_mask)
        solved_flags, solve_steps = _vmap_greedy_solve(
            container, eval_boards
        )  # Pass container
        solved_non_trivial_mask = jnp.logical_and(initial_unsolved_mask, solved_flags)
        solved_count_k = jnp.sum(solved_non_trivial_mask)
        total_solved_k_incl_trivial = solved_count_k + num_trivial
        eval_solved_pct_k = (total_solved_k_incl_trivial / num_samples_per_k) * 100.0
        sum_steps_k = jnp.sum(jnp.where(solved_non_trivial_mask, solve_steps, 0))
        avg_steps_k = jnp.where(
            solved_count_k > 0, sum_steps_k / solved_count_k, config.EVAL_MAX_STEPS + 1
        ).astype(float)
        all_metrics[f"eval_solved_pct_k{k}"] = float(eval_solved_pct_k)
        all_metrics[f"eval_avg_steps_k{k}"] = float(avg_steps_k)
        total_solved += total_solved_k_incl_trivial
        total_samples += num_samples_per_k

    overall_solved_pct = (
        (total_solved / total_samples * 100.0) if total_samples > 0 else 0.0
    )
    all_metrics["eval_solved_pct_overall"] = overall_solved_pct
    return all_metrics, eval_key
