import os
import time
import traceback
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm
import orbax.checkpoint as ocp

import config
from core import (
    evaluate,
    generate_all_boards,
    train_step,
    TrainableStateContainer,
    evaluate_on_all_states,
    evaluate_stochastically_on_all_states,
)
from models import PolicyNet


def main():
    ALL_BOARDS = generate_all_boards(config.N)
    print(f"Generated {ALL_BOARDS.shape[0]} unique board states for N={config.N}.")
    all_boards_device = jax.device_put(ALL_BOARDS)

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXP)
    run_name = f"GFN_LightsOut_N{config.N}_NNX_MLP_Container_{int(time.time())}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        run_params = {
            k: v
            for k, v in config.__dict__.items()
            if not k.startswith("__")
            and isinstance(v, (int, float, str, list, tuple))
        }
        run_params.pop("EMBED_DIM", None)
        run_params.pop("NUM_HEADS", None)
        run_params.pop("NUM_LAYERS", None)
        mlflow.log_params(run_params)

        key = jax.random.PRNGKey(int(time.time()))
        pf_key, pb_key, loop_key = jax.random.split(key, 3)

        print("Initializing NNX models...")
        pf_rngs = nnx.Rngs(params=pf_key)
        pb_rngs = nnx.Rngs(params=pb_key)
        pf_model = PolicyNet(
            in_dim=config.FLAT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            out_dim=config.ACTION_DIM,
            rngs=pf_rngs,
        )
        pb_model = PolicyNet(
            in_dim=config.FLAT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            out_dim=config.ACTION_DIM,
            rngs=pb_rngs,
        )
        z_param = nnx.Param(jnp.array(0.0, dtype=jnp.float32))

        trainable_container = TrainableStateContainer(
            pf_model=pf_model, pb_model=pb_model, z_param=z_param
        )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.LR,
            warmup_steps=config.WARMUP_STEPS,
            decay_steps=config.TRAINING_STEPS - config.WARMUP_STEPS,
            end_value=config.LR * 0.01,
        )
        optax_optimizer = optax.adamw(
            learning_rate=lr_schedule, weight_decay=config.WEIGHT_DECAY
        )

        optimizer = nnx.Optimizer(trainable_container, optax_optimizer)
        print("Optimizer initialized.")

        jitted_train_step = partial(train_step, all_boards_shape=ALL_BOARDS.shape)

        print("Starting NNX training...")
        pbar = tqdm(range(1, config.TRAINING_STEPS + 1), desc="Starting...")
        start_time = time.time()
        last_log_time = start_time

        current_container = trainable_container
        current_optimizer = optimizer
        current_loop_key = loop_key
        final_step = 0

        try:
            for step in pbar:
                final_step = step
                step_key, current_loop_key = jax.random.split(current_loop_key)

                train_metrics, _ = jitted_train_step(
                    current_container,
                    current_optimizer,
                    step_key,
                    all_boards=all_boards_device,
                )
                train_metrics_np = jax.device_get(train_metrics)

                loss_finite_flag = train_metrics_np.get("loss_finite", True)
                current_loss = train_metrics_np.get("loss")

                if not loss_finite_flag:
                    print(
                        f"\nWarning: Non-finite loss ({current_loss}) detected at step {step}. Stopping training."
                    )
                    mlflow.log_metric("nan_loss_step", step, step=step)
                    break

                if step % config.LOG_EVERY == 0 or step == 1:
                    current_lr = lr_schedule(step)
                    train_metrics_log = dict(train_metrics_np)
                    train_metrics_log.pop("loss_finite", None)
                    train_metrics_log["learning_rate"] = current_lr

                    logz = float(current_container.z_param.value)
                    train_metrics_log["logZ"] = logz

                    mlflow.log_metrics(train_metrics_log, step=step)

                    loss = float(train_metrics_np.get("loss", np.nan))
                    solved = float(train_metrics_np.get("solved_pct", np.nan))
                    avg_steps_rep = float(
                        train_metrics_np.get("avg_steps_solved", np.nan)
                    )
                    current_time = time.time()
                    steps_since_log = (
                        max(1, step % config.LOG_EVERY) if step > 1 else 1
                    )
                    time_since_log = current_time - last_log_time
                    sps = (
                        steps_since_log / time_since_log
                        if time_since_log > 0
                        else 0
                    )
                    last_log_time = current_time
                    pbar.set_description(
                        f"L:{loss:.3f}|Z:{logz:.2f}|TrS%:{solved:.1f}|TrSt:{avg_steps_rep:.1f}|LR:{current_lr:.1e}|SPS:{sps:.1f}"
                    )

                if step % config.EVAL_EVERY == 0 or step == 1:
                    (
                        eval_k_key,
                        eval_greedy_key,
                        eval_stoch_key,
                        current_loop_key,
                    ) = jax.random.split(current_loop_key, 4)

                    k_eval_metrics, _ = evaluate(
                        current_container,
                        config.EVAL_K_VALUES,
                        config.EVAL_SAMPLES_PER_K,
                        eval_k_key,
                    )
                    k_eval_metrics_np = jax.device_get(k_eval_metrics)
                    mlflow.log_metrics(k_eval_metrics_np, step=step)
                    k_overall_pct = k_eval_metrics_np.get(
                        "eval_k_perturb_solved_pct_overall", np.nan
                    )

                    all_states_eval_metrics = evaluate_on_all_states(
                        current_container
                    )
                    all_states_eval_metrics_np = jax.device_get(
                        all_states_eval_metrics
                    )
                    mlflow.log_metrics(all_states_eval_metrics_np, step=step)
                    all_states_greedy_pct = all_states_eval_metrics_np.get(
                        "eval_all_states_greedy_solved_pct", np.nan
                    )

                    num_runs = 10
                    stochastic_eval_metrics = evaluate_stochastically_on_all_states(
                        eval_stoch_key,
                        current_container,
                        num_stochastic_runs=num_runs,
                    )
                    stochastic_eval_metrics_np = jax.device_get(
                        stochastic_eval_metrics
                    )
                    mlflow.log_metrics(stochastic_eval_metrics_np, step=step)
                    stochastic_pct = stochastic_eval_metrics_np.get(
                        f"eval_all_states_stochastic{num_runs}_solved_pct",
                        np.nan,
                    )

                    tqdm.write(
                        f"\n--- Eval @ Step {step} --- | K-Perturb Solved={k_overall_pct:.1f}% "
                        f"| All States Greedy={all_states_greedy_pct:.1f}% "
                        f"| All States Stochastic({num_runs})={stochastic_pct:.1f}% ---"
                    )

        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        finally:
            total_time = time.time() - start_time
            print("\n--- Training Finished (NNX) ---")
            print(f"Total steps executed: {final_step}")
            print(f"Total time: {total_time:.2f}s")

            final_z_val = current_container.z_param.value.item()
            print(f"Final LogZ estimate: {final_z_val:.3f}")
            mlflow.log_metric("final_logZ", final_z_val, step=final_step)

            print("\nRunning final evaluation...")
            try:
                (
                    final_k_key,
                    final_greedy_key,
                    final_stoch_key,
                    _,
                ) = jax.random.split(current_loop_key, 4)

                final_k_metrics, _ = evaluate(
                    current_container,
                    config.EVAL_K_VALUES,
                    config.EVAL_SAMPLES_PER_K * 4,
                    final_k_key,
                )
                final_k_metrics_np = jax.device_get(final_k_metrics)
                final_k_overall_pct = final_k_metrics_np.get(
                    'eval_k_perturb_solved_pct_overall', np.nan
                )
                print(
                    f"Final K-Perturb Eval Overall Solved: {final_k_overall_pct:.2f}%"
                )
                mlflow.log_metrics(
                    {
                        f"final_k_perturb_{k.replace('eval_', '')}": v
                        for k, v in final_k_metrics_np.items()
                    },
                    step=final_step,
                )

                final_all_states_metrics = evaluate_on_all_states(
                    current_container
                )
                final_all_states_metrics_np = jax.device_get(
                    final_all_states_metrics
                )
                final_all_greedy_pct = final_all_states_metrics_np.get(
                    'eval_all_states_greedy_solved_pct', np.nan
                )
                print(
                    f"Final All States Eval Greedy Solved: {final_all_greedy_pct:.2f}%"
                )
                mlflow.log_metrics(
                    {
                        f"final_greedy_{k.replace('eval_', '')}": v
                        for k, v in final_all_states_metrics_np.items()
                    },
                    step=final_step,
                )

                num_runs_final = 20
                final_stochastic_metrics = evaluate_stochastically_on_all_states(
                    final_stoch_key,
                    current_container,
                    num_stochastic_runs=num_runs_final,
                )
                final_stochastic_metrics_np = jax.device_get(
                    final_stochastic_metrics
                )
                final_stochastic_pct = final_stochastic_metrics_np.get(
                    f"eval_all_states_stochastic{num_runs_final}_solved_pct",
                    np.nan,
                )
                print(
                    f"Final All States Eval Stochastic ({num_runs_final} runs) Solved: {final_stochastic_pct:.2f}%"
                )
                mlflow.log_metrics(
                    {
                        f"final_stochastic_{k.replace('eval_', '')}": v
                        for k, v in final_stochastic_metrics_np.items()
                    },
                    step=final_step,
                )

            except Exception as e:
                print(f"Final evaluation failed: {e}")
                traceback.print_exc()

            print("\nSaving final model checkpoint...")
            try:
                save_dir_base = Path("./ckpt")
                save_dir = save_dir_base / run.info.run_id
                _, state_to_save = nnx.split(current_container)
                checkpointer = ocp.StandardCheckpointer()
                checkpointer.save(save_dir.resolve(), state_to_save, force=True)
                checkpointer.wait_until_finished()
                resolved_path = save_dir.resolve()
                print(f"Checkpoint saved successfully in directory: {resolved_path}")
                mlflow.log_param("checkpoint_path", str(resolved_path))
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
                traceback.print_exc()

            print("\nTraining complete!")


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
