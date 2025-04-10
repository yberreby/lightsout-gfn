# lightsout-gfn

Experimental [JAX](https://docs.jax.dev/en/latest/) (with [NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)) codebase using [GFlowNets](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3) to solve 3x3 [LightsOut](https://link.springer.com/chapter/10.1007/978-3-642-40273-9_13) puzzles.

<img src="./demo.gif" alt="Demo">

_Demo generated with `infer.py`_

Experiment tracking with [MLFlow](https://mlflow.org/).

Dependencies managed with [`uv`](https://docs.astral.sh/uv/).

- **Train**: `uv run -m train`.
- **Monitor**: `uv run mlflow server --host 127.0.0.1 --port 8080`
- **Infer** (with a GUI / to GIF): `uv run -m infer ./some_checkpoint`

## Why?

As of writing, I am very new to both NNX and GFlowNets. This is a way for me to familiarize myself with them, using a toy environment: 'Lights Out!'.

This is a small pilot experiment, cobbled together quickly. Feel free to tell me what I'm doing wrong in the GitHub issues!

## Sample training run on RTX 4060

```
❯ uv run -m train
Generated 512 boards for N=3.
Generated 512 unique board states for N=3.
MLflow Run ID: 705f6d566bad4f01b99643d2badf160f
Initializing NNX models...
Optimizer initialized.
Starting NNX training...

--- Eval @ Step 1 --- | K-Perturb Solved=7.3% | All States Greedy=0.8% | All States Stochastic(10)=25.6% ---

--- Eval @ Step 1000 --- | K-Perturb Solved=100.0% | All States Greedy=100.0% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 2000 --- | K-Perturb Solved=91.3% | All States Greedy=90.2% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 3000 --- | K-Perturb Solved=90.5% | All States Greedy=88.3% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 4000 --- | K-Perturb Solved=86.1% | All States Greedy=87.1% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 5000 --- | K-Perturb Solved=84.0% | All States Greedy=84.4% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 6000 --- | K-Perturb Solved=88.9% | All States Greedy=89.3% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 7000 --- | K-Perturb Solved=81.9% | All States Greedy=86.1% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 8000 --- | K-Perturb Solved=81.8% | All States Greedy=86.7% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 9000 --- | K-Perturb Solved=83.0% | All States Greedy=88.9% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 10000 --- | K-Perturb Solved=79.5% | All States Greedy=87.7% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 11000 --- | K-Perturb Solved=89.4% | All States Greedy=91.4% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 12000 --- | K-Perturb Solved=78.8% | All States Greedy=85.7% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 13000 --- | K-Perturb Solved=78.5% | All States Greedy=84.6% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 14000 --- | K-Perturb Solved=83.7% | All States Greedy=87.7% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 15000 --- | K-Perturb Solved=83.2% | All States Greedy=86.5% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 16000 --- | K-Perturb Solved=84.2% | All States Greedy=88.9% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 17000 --- | K-Perturb Solved=83.2% | All States Greedy=86.5% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 18000 --- | K-Perturb Solved=84.7% | All States Greedy=87.3% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 19000 --- | K-Perturb Solved=82.1% | All States Greedy=86.5% | All States Stochastic(10)=100.0% ---

--- Eval @ Step 20000 --- | K-Perturb Solved=89.1% | All States Greedy=88.3% | All States Stochastic(10)=100.0% ---
L:1.270|Z:-4.11|TrS%:98.2|TrSt:8.1|LR:1.0e-05|SPS:3.6: 100%|███████| 20000/20000 [01:05<00:00, 304.55it/s]

--- Training Finished (NNX) ---
Total steps executed: 20000
Total time: 65.67s
Final LogZ estimate: -4.114

Running final evaluation...
Final K-Perturb Eval Overall Solved: 86.89%
Final All States Eval Greedy Solved: 88.28%
Final All States Eval Stochastic (20 runs) Solved: 100.00%

Saving final model checkpoint...
WARNING:absl:[process=0][thread=async_save] Skipped cross-host ArrayMetadata validation because only one process is found: process_index=0.
Checkpoint saved successfully in directory: /home/yberreby/code/lightsout-gfn/ckpt/705f6d566bad4f01b99643d2badf160f

Training complete!
```


## Disclaimers and remarks

- A $$N \times N$$ grid has $$2^{N^2}$$ states. For 3x3, this is just $$2^9 = 512$$ states; this is nothing and can easily be memorized. However, the point here was to show that, and how, the GFlowNet training objective could be used in order to learn how to solve this game.
- The code as it stands has data leakage. Partitioning based on starting states is insufficient, due to the effect of sampled actions.
- It would be much more meaningful to do this on bigger grids. Better yet, to use a Transformer or other sequence model in order to generalize across grid sizes.
- There have long been [deterministic solvers for LightsOut](https://github.com/pmneila/Lights-Out). This repository is not attempting to beat them, or to be more than a proof of concept.


## License

MIT licensed. See `./LICENSE`.
