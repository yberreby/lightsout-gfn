import os
import time
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

import config
from core import evaluate, generate_all_boards, train_step
from models import PolicyNet


class TrainableStateContainer(nnx.Module):
    def __init__(self, pf_model: PolicyNet, pb_model: PolicyNet, z_param: nnx.Param):
        self.pf_model = pf_model
        self.pb_model = pb_model
        self.z_param = z_param


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
            if not k.startswith("__") and isinstance(v, (int, float, str, list, tuple))
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
        z_param_container = nnx.Param(jnp.array(0.0, dtype=jnp.float32))

        trainable_container = TrainableStateContainer(
            pf_model, pb_model, z_param_container
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
                        step - ((step - 1) // config.LOG_EVERY * config.LOG_EVERY)
                        if step > 1
                        else 1
                    )
                    time_since_log = current_time - last_log_time
                    sps = steps_since_log / time_since_log if time_since_log > 0 else 0
                    last_log_time = current_time
                    pbar.set_description(
                        f"L:{loss:.3f}|Z:{logz:.2f}|S%:{solved:.1f}|St:{avg_steps_rep:.1f}|LR:{current_lr:.1e}|SPS:{sps:.1f}"
                    )

                if step % config.EVAL_EVERY == 0 or step == 1:
                    eval_key, current_loop_key = jax.random.split(current_loop_key)
                    eval_metrics, _ = evaluate(
                        current_container,
                        config.EVAL_K_VALUES,
                        config.EVAL_SAMPLES_PER_K,
                        eval_key,
                    )
                    eval_metrics_np = jax.device_get(eval_metrics)
                    mlflow.log_metrics(eval_metrics_np, step=step)
                    overall_pct = eval_metrics_np.get("eval_solved_pct_overall", np.nan)
                    tqdm.write(
                        f"\n--- Eval @ Step {step} --- | Overall Solved={overall_pct:.1f}% ---"
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
            mlflow.log_metric(
                "total_training_time_seconds", total_time, step=final_step
            )

            print("\nRunning final evaluation...")
            try:
                eval_key, _ = jax.random.split(current_loop_key)
                final_metrics, _ = evaluate(
                    current_container,
                    config.EVAL_K_VALUES,
                    config.EVAL_SAMPLES_PER_K * 4,
                    eval_key,
                )
                final_metrics_np = jax.device_get(final_metrics)
                print(
                    f"Final Eval Overall Solved: {final_metrics_np['eval_solved_pct_overall']:.2f}%"
                )
                final_k_metrics = {
                    k: final_metrics_np.get(f"eval_avg_steps_k{k}")
                    for k in config.EVAL_K_VALUES
                }
                for k in sorted(final_k_metrics.keys()):
                    avg_step_val = final_k_metrics[k]
                    print(
                        f"  K={k}: Avg Steps={'N/A' if avg_step_val is None else avg_step_val:.2f}"
                    )
                mlflow.log_metrics(
                    {f"final_{k}": v for k, v in final_metrics_np.items()},
                    step=final_step,
                )
            except Exception as e:
                print(f"Final evaluation failed: {e}")
                traceback.print_exc()

            print("Training complete!")


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
