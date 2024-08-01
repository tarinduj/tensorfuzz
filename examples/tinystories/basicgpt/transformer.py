import jax
import jax.numpy as jnp
import numpy as np
from typing import Literal, TypedDict
from flax.training import train_state
import optax
from time import perf_counter

# Assuming these are imported from your tiny_stories module
from basicgpt.tiny_stories import TinyStoriesDataset, TOKENIZER_SIZE

D_TYPE = jnp.float32
MAX_LEN = 64

def init_params(key, shape):
    return jax.random.normal(key, shape, dtype=D_TYPE) * 0.02

def unstable_softmax(logits: jnp.ndarray, axis:int = -1) -> jnp.ndarray:
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=axis, keepdims=True)

# def unstable_softmax(logits: jnp.ndarray, axis:int = -1) -> jnp.ndarray:
#     exp_logits = jnp.exp(logits)
#     sum_exp_logits = jnp.zeros_like(jnp.sum(exp_logits, axis=axis, keepdims=True))
#     return exp_logits / sum_exp_logits

class Dense:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def init(self, key):
        k1, k2 = jax.random.split(key)
        return {
            'w': init_params(k1, (self.in_dim, self.out_dim)),
            'b': jnp.zeros((self.out_dim,))
        }

    def __call__(self, params, x):
        return jnp.dot(x, params['w']) + params['b']

class LayerNorm:
    def __init__(self, dim):
        self.dim = dim

    def init(self, key):
        return {
            'g': jnp.ones((self.dim,)),
            'b': jnp.zeros((self.dim,))
        }

    def __call__(self, params, x, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return params['g'] * (x - mean) / jnp.sqrt(var + eps) + params['b']

class MLP:
    def __init__(self, d_model, d_ff):
        self.dense1 = Dense(d_model, d_ff)
        self.dense2 = Dense(d_ff, d_model)

    def init(self, key):
        k1, k2 = jax.random.split(key)
        return {
            'dense1': self.dense1.init(k1),
            'dense2': self.dense2.init(k2)
        }

    def __call__(self, params, x):
        x = self.dense1(params['dense1'], x)
        x = jax.nn.gelu(x)
        x = self.dense2(params['dense2'], x)
        return x

class MLPBlock:
    def __init__(self, d_model, d_ff):
        self.layer_norm = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def init(self, key):
        k1, k2 = jax.random.split(key)
        return {
            'layer_norm': self.layer_norm.init(k1),
            'mlp': self.mlp.init(k2)
        }

    def __call__(self, params, x):
        y = self.layer_norm(params['layer_norm'], x)
        y = self.mlp(params['mlp'], y)
        return x + y

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = Dense(d_model, d_model)
        self.wk = Dense(d_model, d_model)
        self.wv = Dense(d_model, d_model)
        self.wo = Dense(d_model, d_model)

    def init(self, key):
        keys = jax.random.split(key, 4)
        return {
            'wq': self.wq.init(keys[0]),
            'wk': self.wk.init(keys[1]),
            'wv': self.wv.init(keys[2]),
            'wo': self.wo.init(keys[3])
        }

    def __call__(self, params, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.wq(params['wq'], x).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.wk(params['wk'], x).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.wv(params['wv'], x).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.d_head)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn = jnp.where(mask == 0, float('-inf'), attn)
        attn = unstable_softmax(attn, axis=-1)

        out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.wo(params['wo'], out)

class SelfAttentionBlock:
    def __init__(self, d_model, n_heads):
        self.layer_norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)

    def init(self, key):
        k1, k2 = jax.random.split(key)
        return {
            'layer_norm': self.layer_norm.init(k1),
            'attention': self.attention.init(k2)
        }

    def __call__(self, params, x):
        y = self.layer_norm(params['layer_norm'], x)
        y = self.attention(params['attention'], y)
        return x + y

class Transformer:
    def __init__(self, d_model, n_heads, n_layers, d_ff):
        self.d_model = d_model
        self.token_embedding = Dense(TOKENIZER_SIZE, d_model)
        self.position_embedding = Dense(MAX_LEN, d_model)
        self.blocks = [(SelfAttentionBlock(d_model, n_heads), MLPBlock(d_model, d_ff)) for _ in range(n_layers)]
        self.final_layer_norm = LayerNorm(d_model)
        self.output_proj = Dense(d_model, TOKENIZER_SIZE)

    def init(self, key):
        keys = jax.random.split(key, 4 + len(self.blocks) * 2)
        params = {
            'token_embedding': self.token_embedding.init(keys[0]),
            'position_embedding': self.position_embedding.init(keys[1]),
            'blocks': [],
            'final_layer_norm': self.final_layer_norm.init(keys[-2]),
            'output_proj': self.output_proj.init(keys[-1])
        }
        
        for i, (attn, mlp) in enumerate(self.blocks):
            params['blocks'].append({
                'attention': attn.init(keys[2+i*2]),
                'mlp': mlp.init(keys[3+i*2])
            })
        
        return params

    def __call__(self, params, x):
        positions = jnp.arange(x.shape[1])
        x = self.token_embedding(params['token_embedding'], jax.nn.one_hot(x, TOKENIZER_SIZE)) + \
            self.position_embedding(params['position_embedding'], jax.nn.one_hot(positions, MAX_LEN))
        
        for i, (attn, mlp) in enumerate(self.blocks):
            x = attn(params['blocks'][i]['attention'], x)
            x = mlp(params['blocks'][i]['mlp'], x)
        
        x = self.final_layer_norm(params['final_layer_norm'], x)
        x = self.output_proj(params['output_proj'], x)
        return x

def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6

class Config(TypedDict):
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    learning_rate: float
    max_steps: int
    batch_size: int
    weight_decay: float
    block_type: Literal["MLP", "KAN", "Hybrid"]

def masked_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray):
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    mask = mask[:, 1:]
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    probs = unstable_softmax(logits)
    log_probs = jnp.log(probs)
    loss = -jnp.sum(log_probs * one_hot_targets, axis=-1)
    loss = loss * mask
    total_tokens = jnp.sum(mask.flatten())
    return jnp.sum(loss.flatten()) / total_tokens

def create_train_state(params, config):
    model = Transformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff']
    )
    params = model.init(rng)
    optimizer = optax.adamw(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    return train_state.TrainState.create(
        apply_fn=model.__call__,
        params=params,
        tx=optimizer
    )

@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray, mask: jnp.ndarray):
    def loss_fn(params):
        logits = state.apply_fn(params, batch)
        return masked_cross_entropy(logits, batch, mask)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

rng = jax.random.PRNGKey(0)

config = Config(
    d_model=128,
    n_heads=8,
    n_layers=1,
    d_ff=512,
    learning_rate=1e-5,
    batch_size=16,
    weight_decay=0.001,
    block_type="MLP", 
    max_steps=10000
)

if __name__ == "__main__":
    print("Creating model...")
    state = create_train_state(rng, config)
    print("Number of parameters: ", param_count(state.params))

    for step, (batch, mask) in enumerate(TinyStoriesDataset(max_len=MAX_LEN).create_batches(config['batch_size'])):
        step_start_time = perf_counter()
        state, loss = train_step(state, batch, mask)
        step_end_time = perf_counter()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss}")
            print(f"Time taken for step: {step_end_time - step_start_time}")
        if step % 1000 == 0:
            print("Saving params...")
            model_type = config["block_type"]
            np.save(f"checkpoints/params_{model_type}.npy", state.params)
            print("Params saved.")
        if step >= config['max_steps']:
            break