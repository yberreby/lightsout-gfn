import jax.numpy as jnp
from flax import nnx


class PolicyNet(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        out_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.dense0 = nnx.Linear(in_dim, hidden_dim_1, rngs=rngs)
        self.dense1 = nnx.Linear(hidden_dim_1, hidden_dim_2, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim_2, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = None) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        return x
