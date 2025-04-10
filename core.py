import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax import nnx

import config
from models import PolicyNet

class TrainableStateContainer(nnx.Module):
    pf_model: PolicyNet
    pb_model: PolicyNet
    z_param: nnx.Param

    def __init__(
        self, pf_model: PolicyNet, pb_model: PolicyNet, z_param: nnx.Param
    ):
        self.pf_model = pf_model
        self.pb_model = pb_model
        self.z_param = z_param


@partial(jax.jit, static_argnames=("n",))
def get_neighbors(index: int, n: int) -> jnp.ndarray:
    row, col = jnp.divmod(index, n)
    rel_pos = jnp.array(
        [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32
    )
    abs_rows = row + rel_pos[:, 0]
    abs_cols = col + rel_pos[:, 1]
    clipped_rows = jnp.clip(abs_rows, 0, n - 1)
    clipped_cols = jnp.clip(abs_cols, 0, n - 1)
    neighbor_indices = (clipped_rows * n + clipped_cols).astype(jnp.int32)
    return neighbor_indices


@jax.jit
def toggle_tile(board: jnp.ndarray, action_idx: int) -> jnp.ndarray:
    n = config.N
    action_idx = jnp.asarray(action_idx, dtype=jnp.int32)
    affected_indices = get_neighbors(action_idx, n)
    updates = jnp.zeros_like(board, dtype=jnp.int8).at[affected_indices].set(
        1, mode='promise_in_bounds'
    )
    return jnp.bitwise_xor(board.astype(jnp.int8), updates)


@jax.jit
def is_solved(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(board == 0)


def generate_all_boards(n: int = config.N) -> np.ndarray:
    if (
        not hasattr(generate_all_boards, "cache")
        or generate_all_boards.cache is None
        or generate_all_boards.cache_n != n
    ):
        dim = n * n
        num_states = 2**dim
        if num_states > (2**25)+1:
            raise MemoryError(
                f"Attempting to generate {num_states} states for N={n}, too large."
            )
        print(f"Generated {num_states} boards for N={n}.")
        indices = np.arange(num_states, dtype=np.uint32)
        powers_of_2 = (2 ** np.arange(dim, dtype=np.uint32)).astype(np.uint32)
        all_boards_np = ((indices[:, None] & powers_of_2) != 0).astype(np.int8)
        generate_all_boards.cache = all_boards_np
        generate_all_boards.cache_n = n
    return generate_all_boards.cache

generate_all_boards.cache = None
generate_all_boards.cache_n = -1


@partial(jax.jit, static_argnames=("n", "n_actions"))
def apply_k_random_actions_eval(
    key, n_actions, n=config.N
) -> jnp.ndarray:
    flat_dim = n * n
    start_board = jnp.zeros((flat_dim,), dtype=jnp.int8)
    actions = jax.random.randint(key, (n_actions,), 0, flat_dim)
    def scan_step(bs, a):
        return toggle_tile(bs, a), None
    final, _ = jax.lax.scan(scan_step, start_board, actions)
    return final.astype(jnp.int8)


@partial(nnx.jit)
def sample_trajectory(key, container, start_board):
    max_traj_len = config.MAX_TRAJECTORY_LEN
    pf_model = container.pf_model

    @nnx.jit
    def _sample_action_mlp(key, board):
        logits = pf_model(board.astype(jnp.float32))
        valid_mask = jnp.where(is_solved(board), -jnp.inf, 0.0)
        masked_logits = logits + valid_mask
        action = jax.random.categorical(key, masked_logits)
        log_prob = jax.nn.log_softmax(masked_logits)[action]
        return action.astype(jnp.int32), log_prob.astype(jnp.float32)

    def step_fn(carry, _):
        key, cb, solved = carry
        ak, nk = jax.random.split(key)
        state_before = cb
        a_ns, lp_ns = _sample_action_mlp(ak, cb)
        noop_a, noop_lp = jnp.int32(-1), jnp.float32(0.0)
        a, pf_lp = jax.lax.cond(
            solved, lambda: (noop_a, noop_lp), lambda: (a_ns, lp_ns)
        )
        nb = jax.lax.cond(
            a == noop_a, lambda b: b, lambda b: toggle_tile(b, a), cb
        ).astype(jnp.int8)
        next_s = jnp.logical_or(solved, is_solved(nb))
        return (nk, nb, next_s), (state_before, a, pf_lp)

    init_carry = (key, start_board.astype(jnp.int8), is_solved(start_board))
    _, (states_before, actions, pf_log_probs) = jax.lax.scan(
        step_fn, init_carry, None, length=max_traj_len
    )

    def get_next(b, a):
        nb = jax.lax.cond(a == -1, lambda x: x, lambda x: toggle_tile(x, a), b)
        return nb, nb

    _, states_after_scan = jax.lax.scan(get_next, start_board, actions)
    full_states_traj = jnp.concatenate(
        [start_board[None, :], states_after_scan], axis=0
    ).astype(
        jnp.int8
    )
    solved_mask_traj = jax.vmap(is_solved)(full_states_traj)
    first_solved_idx = jnp.argmax(solved_mask_traj)
    was_solved = jnp.any(solved_mask_traj)
    solved_steps = jnp.where(was_solved, first_solved_idx, max_traj_len)
    solved_steps = jnp.minimum(solved_steps, max_traj_len)
    mask = jnp.arange(max_traj_len) < solved_steps
    log_reward = jax.lax.cond(
        was_solved,
        lambda s: -jnp.log(
            1.0 + config.REWARD_INV_STEPS_C * s.astype(jnp.float32)
        ),
        lambda s: jnp.float32(config.MIN_REWARD_LOG),
        solved_steps,
    ).astype(jnp.float32)

    return full_states_traj, actions, pf_log_probs, log_reward, mask, solved_steps


def batch_sample_trajectories(key, container, start_boards):
    keys = jax.random.split(key, start_boards.shape[0])
    start_boards_jax = jnp.asarray(start_boards)
    return jax.vmap(sample_trajectory, in_axes=(0, None, 0))(
        keys, container, start_boards_jax
    )


@partial(nnx.jit)
def compute_tb_loss(container, batch_data):
    pf = container.pf_model
    pb = container.pb_model
    z = container.z_param.value
    s, a, _, r, m, _ = batch_data
    T = config.MAX_TRAJECTORY_LEN
    ad = config.ACTION_DIM

    @nnx.jit
    def get_lp(model, states, actions, masks):
        B, T_s, _ = states.shape
        logits = model(states.reshape(B * T_s, -1).astype(jnp.float32)).reshape(
            B, T_s, ad
        )
        lp = jnp.take_along_axis(
            jax.nn.log_softmax(logits, -1), actions[:, :, None], 2
        ).squeeze(-1)
        return jnp.sum(lp * masks, axis=1)

    pf_lp = get_lp(pf, s[:, :T, :], a, m)
    pb_lp = get_lp(pb, s[:, 1 : T + 1, :], a, m)
    z_f, pf_f, r_f, pb_f = jax.tree.map(
        lambda x: x.astype(jnp.float32), (z, pf_lp, r, pb_lp)
    )
    loss = jnp.mean((z_f + pf_f - r_f - pb_f) ** 2)
    return loss


@partial(nnx.jit, static_argnames=("all_boards_shape",))
def train_step(container, optimizer, key, all_boards, all_boards_shape):
    num_b = all_boards_shape[0]
    bk, sk, nk = jax.random.split(key, 3)
    idx = jax.random.choice(
        bk, num_b, shape=(config.BATCH_SIZE,), replace=True
    )
    sb = all_boards[idx]
    batch = batch_sample_trajectories(sk, container, sb)
    lv, gr = nnx.value_and_grad(lambda c: compute_tb_loss(c, batch))(container)
    optimizer.update(gr)
    lf = jnp.isfinite(lv)
    _, _, _, lr, _, ss = batch
    T = config.MAX_TRAJECTORY_LEN
    sm = ss < T
    sc = jnp.sum(sm)
    sp = jnp.mean(sm) * 100.0
    avs = jnp.sum(jnp.where(sm, ss, 0)) / jnp.maximum(1e-9, sc)
    rs = jnp.where(sc > 0, avs, T).astype(float)
    mets = {
        "loss": lv,
        "solved_pct": sp,
        "avg_steps_solved": rs,
        "avg_log_reward": jnp.mean(lr),
        "loss_finite": lf,
    }
    return mets, nk


@partial(nnx.jit)
def greedy_solve_board(container, start_board):
    eval_max_steps = config.EVAL_MAX_STEPS
    start_s = is_solved(start_board)
    pf = container.pf_model
    init = (
        jnp.asarray(start_board, dtype=jnp.int8),
        start_s,
        jnp.int32(eval_max_steps + 1),
        jnp.int32(0),
    )

    def step(c, _):
        cb, s, fs, i = c
        a = jax.lax.cond(
            s,
            lambda: -1,
            lambda: jnp.argmax(pf(cb.astype(jnp.float32))).astype(jnp.int32),
        )
        nb = jax.lax.cond(
            a == -1, lambda: cb, lambda: toggle_tile(cb, a)
        ).astype(jnp.int8)
        cs = is_solved(nb)
        nly = jnp.logical_and(cs, ~s)
        nf = jnp.where(nly, i + 1, fs)
        ns = jnp.logical_or(s, cs)
        return (nb, ns, nf, i + 1), None

    final, _ = jax.lax.scan(step, init, None, length=eval_max_steps)
    sr = jnp.where(start_s, 0, final[2]).astype(jnp.int32)
    sf = sr <= eval_max_steps
    return sf, sr


# FIXME: prohibitive for large grids!
@partial(nnx.jit)
def evaluate_on_all_states(container):
    n = config.N
    all_b_np = generate_all_boards(n)
    all_b = jnp.asarray(all_b_np)
    total_b = all_b.shape[0]
    if total_b == 0:
        return {"eval_all_states_greedy_solved_pct": jnp.array(0.0)}
    flags, _ = jax.vmap(greedy_solve_board, in_axes=(None, 0))(
        container, all_b
    )
    acc = jnp.mean(flags) * 100.0
    return {"eval_all_states_greedy_solved_pct": acc}


@partial(nnx.jit)
def stochastic_solve_board(key, container, start_board):
    eval_max_steps = config.EVAL_MAX_STEPS
    start_s = is_solved(start_board)
    pf = container.pf_model
    init = (
        key,
        jnp.asarray(start_board, dtype=jnp.int8),
        start_s,
        jnp.int32(eval_max_steps + 1),
        jnp.int32(0),
    )

    def step(c, _):
        k, cb, s, fs, i = c
        sk, nk = jax.random.split(k)
        a = jax.lax.cond(
            s,
            lambda k_: -1,
            lambda k_: jax.random.categorical(
                k_, pf(cb.astype(jnp.float32))
            ).astype(jnp.int32),
            sk,
        )
        nb = jax.lax.cond(
            a == -1, lambda: cb, lambda: toggle_tile(cb, a)
        ).astype(jnp.int8)
        cs = is_solved(nb)
        nly = jnp.logical_and(cs, ~s)
        nf = jnp.where(nly, i + 1, fs)
        ns = jnp.logical_or(s, cs)
        return (nk, nb, ns, nf, i + 1), None

    final, _ = jax.lax.scan(step, init, None, length=eval_max_steps)
    sr = jnp.where(start_s, 0, final[3]).astype(jnp.int32)
    sf = sr <= eval_max_steps
    return sf, sr


def evaluate_stochastically_on_all_states(
    key, container, num_stochastic_runs
):
    n = config.N
    all_b_np = generate_all_boards(n)
    all_b = jnp.asarray(all_b_np)
    total_b = all_b.shape[0]
    mname = f"eval_all_states_stochastic{num_stochastic_runs}_solved_pct"
    if total_b == 0:
        return {mname: jnp.array(0.0)}
    keys = jax.random.split(key, total_b * num_stochastic_runs).reshape(
        (total_b, num_stochastic_runs, 2)
    )
    vmap_runs = jax.vmap(stochastic_solve_board, in_axes=(0, None, None))
    vmap_boards = jax.vmap(vmap_runs, in_axes=(0, None, 0))
    flags, _ = vmap_boards(keys, container, all_b)
    acc = jnp.mean(jnp.any(flags, axis=1)) * 100.0
    return {mname: acc}


def evaluate(container, k_values, num_samples_per_k, eval_key):
    all_metrics = {}
    total_solved_overall = 0
    total_samples_overall = 0
    n = config.N
    v_apply_k = jax.vmap(
        partial(apply_k_random_actions_eval, n=n), in_axes=(0, None)
    )
    v_greedy = jax.vmap(greedy_solve_board, in_axes=(None, 0))

    for k in k_values:
        k_key, eval_key = jax.random.split(eval_key)
        b_keys = jax.random.split(k_key, num_samples_per_k)
        boards = v_apply_k(b_keys, k)
        jax.block_until_ready(boards)
        flags, steps = v_greedy(container, boards)
        jax.block_until_ready(flags)

        num_solved_k = jnp.sum(flags).item()
        pct_k = (num_solved_k / num_samples_per_k) * 100.0
        sum_steps = jnp.sum(jnp.where(flags, steps, 0)).item()

        avg_steps = (
            (sum_steps / max(1e-9, num_solved_k))
            if num_solved_k > 0
            else (config.EVAL_MAX_STEPS + 1)
        )

        all_metrics[f"eval_solved_pct_k{k}"] = float(pct_k)
        all_metrics[f"eval_avg_steps_k{k}"] = float(avg_steps)
        total_solved_overall += num_solved_k
        total_samples_overall += num_samples_per_k

    overall_pct = (
        (total_solved_overall / total_samples_overall * 100.0)
        if total_samples_overall > 0
        else 0.0
    )
    all_metrics["eval_k_perturb_solved_pct_overall"] = overall_pct
    return all_metrics, eval_key
