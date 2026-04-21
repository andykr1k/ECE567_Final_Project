import argparse
import os
import sys
import time
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
import distrax
import wandb

from craftax.craftax_env import make_craftax_env_from_name
from logz.batch_logging import batch_log, create_log_dict
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

# -------------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------------

class ActorCritic(nn.Module):
    action_dim: int
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        # Actor
        actor_mean = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic
        critic = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class FeatureNetwork(nn.Module):
    """Randomly initialized and frozen feature network for RLE intrinsic rewards."""
    layer_width: int
    latent_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        x = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return x

# -------------------------------------------------------------------------
# TRAIN LOOP & RLE LOGIC
# -------------------------------------------------------------------------

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    z: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # ---------------- INIT NETWORKS ----------------
        network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"], activation=config["ACTIVATION"])
        
        obs_shape = env.observation_space(env_params).shape
        assert len(obs_shape) == 1, "Only configured for 1D symbolic observations"
        obs_dim = obs_shape[0]

        rng, _rng = jax.random.split(rng)
        
        # Policy takes state + latent Z concatenated
        init_x = jnp.zeros((1, obs_dim + config["RLE_LATENT_DIM"]))
        network_params = network.init(_rng, init_x)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Frozen Feature Network for RLE
        feature_network = FeatureNetwork(
            layer_width=config["LAYER_SIZE"], 
            latent_dim=config["RLE_LATENT_DIM"], 
            activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        feature_params = feature_network.init(_rng, jnp.zeros((1, obs_dim)))

        # ---------------- INIT RLE STATE ----------------
        rng, _rng = jax.random.split(rng)
        z = jax.random.normal(_rng, (config["NUM_ENVS"], config["RLE_LATENT_DIM"]))
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        steps_since_resample = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

        # ---------------- INIT ENV ----------------
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # ---------------- TRAIN LOOP ----------------
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, z, steps_since_resample, feature_params, rng, update_step = runner_state

                # SELECT ACTION (Policy conditioned on state + z)
                rng, _rng = jax.random.split(rng)
                obs_z = jnp.concatenate([last_obs, z], axis=-1)
                pi, value = network.apply(train_state.params, obs_z)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(_rng, env_state, action, env_params)

                # COMPUTE RLE INTRINSIC REWARD (F(s_{t+1}, z))
                phi = feature_network.apply(feature_params, obsv)
                phi_norm = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
                reward_i = jnp.sum(phi_norm * z, axis=-1) * config["RLE_ALPHA"]
                
                # Combined reward
                reward = reward_e + reward_i

                # RLE Z RESAMPLING LOGIC
                steps_since_resample += 1
                needs_resample = done | (steps_since_resample >= config["RLE_RESAMPLE_FREQ"])
                
                # As per the paper: treat the resampling boundary as an episodic terminal state for GAE
                effective_done = needs_resample

                rng, _rng = jax.random.split(rng)
                new_z = jax.random.normal(_rng, (config["NUM_ENVS"], config["RLE_LATENT_DIM"]))
                new_z = new_z / (jnp.linalg.norm(new_z, axis=-1, keepdims=True) + 1e-8)

                next_z = jnp.where(needs_resample[:, None], new_z, z)
                next_steps_since_resample = jnp.where(needs_resample, 0, steps_since_resample)

                transition = Transition(
                    done=effective_done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    z=z,
                    info=info,
                )
                
                runner_state = (train_state, env_state, obsv, next_z, next_steps_since_resample, feature_params, rng, update_step)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, z, steps_since_resample, feature_params, rng, update_step = runner_state
            
            last_obs_z = jnp.concatenate([last_obs, z], axis=-1)
            _, last_val = network.apply(train_state.params, last_obs_z)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        obs_z_batch = jnp.concatenate([traj_batch.obs, traj_batch.z], axis=-1)
                        pi, value = network.apply(params, obs_z_batch)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])

            train_state = update_state[0]
            rng = update_state[-1]

            # Logging metrics
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / (traj_batch.info["returned_episode"].sum() + 1e-8),
                traj_batch.info,
            )
            metric["reward_i"] = traj_batch.reward_i.mean()
            metric["reward_e"] = traj_batch.reward_e.mean()

            if config["DEBUG"] and config["USE_WANDB"]:
                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)
                jax.debug.callback(callback, metric, update_step)

            runner_state = (train_state, env_state, last_obs, z, steps_since_resample, feature_params, rng, update_step + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, z, steps_since_resample, feature_params, _rng, 0)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        
        return {"runner_state": runner_state}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"] + "-" + str(int(config["TOTAL_TIMESTEPS"] // 1e6)) + "M-RLE",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"] and config["SAVE_POLICY"]:
        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states) # extract first repeat
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )
        _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9) 
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # RANDOM LATENT EXPLORATION (RLE) ARGUMENTS
    parser.add_argument("--rle_alpha", type=float, default=0.1, help="Intrinsic Reward Multiplier for RLE")
    parser.add_argument("--rle_latent_dim", type=int, default=32, help="Dimension of Random Latent Vector z")
    parser.add_argument("--rle_resample_freq", type=int, default=1280, help="Resample limit (timesteps) for Random Latent z")

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)