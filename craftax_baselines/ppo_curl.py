import argparse
import os
import sys
import time
import flax.linen as nn

import jax
import jax.numpy as jnp
import chex
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from craftax.craftax.envs.craftax_symbolic_env import (
    get_map_obs_shape,
    get_flat_map_obs_shape,
    get_inventory_obs_shape,
)

from logz.batch_logging import batch_log, create_log_dict
from models.embedding_actor_critic import (
    ActorCritic,
    ActorCriticConv,
    Encoder,
    EmbeddingActorCritic,
    CURLHead,
)
from models.icm import ICMEncoder, ICMForward, ICMInverse
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    delayed_obs: jnp.ndarray
    info: jnp.ndarray

def unflatten_obs(obs: chex.Array) -> tuple[chex.Array, chex.Array]:
    """Split a flat Craftax symbolic observation into (map, inventory).

    Args:
        obs: Array of shape (..., flat_map_size + inventory_size). Any number
            of leading batch dimensions is allowed.

    Returns:
        map_obs: Array of shape (..., H, W, C).
        inventory: Array of shape (..., inventory_size).
    """
    map_shape = get_map_obs_shape()          # (H, W, C)
    flat_map_size = get_flat_map_obs_shape() # H * W * C
    inv_size = get_inventory_obs_shape()     # 51
    expected = flat_map_size + inv_size

    if obs.shape[-1] != expected:
        raise ValueError(
            f"Expected last dim {expected} (= {flat_map_size} map + {inv_size} "
            f"inventory), got {obs.shape[-1]}."
        )

    batch_shape = obs.shape[:-1]
    map_flat = obs[..., :flat_map_size]
    inventory = obs[..., flat_map_size:]
    map_obs = map_flat.reshape(*batch_shape, *map_shape)
    return map_obs, inventory


def flatten_obs(map_obs: chex.Array, inventory: chex.Array) -> chex.Array:
    """Inverse of `unflatten_obs`: concatenate (map, inventory) back to flat.

    Args:
        map_obs: Array of shape (..., H, W, C).
        inventory: Array of shape (..., inventory_size).

    Returns:
        Array of shape (..., flat_map_size + inventory_size) matching the
        layout produced by `render_craftax_symbolic`.
    """
    map_shape = get_map_obs_shape()          # (H, W, C)
    inv_size = get_inventory_obs_shape()

    if map_obs.shape[-3:] != map_shape:
        raise ValueError(
            f"Expected map trailing shape {map_shape}, got {map_obs.shape[-3:]}."
        )
    if inventory.shape[-1] != inv_size:
        raise ValueError(
            f"Expected inventory last dim {inv_size}, got {inventory.shape[-1]}."
        )
    if map_obs.shape[:-3] != inventory.shape[:-1]:
        raise ValueError(
            f"Batch dims disagree: map has {map_obs.shape[:-3]}, "
            f"inventory has {inventory.shape[:-1]}."
        )

    batch_shape = inventory.shape[:-1]
    map_flat = map_obs.reshape(*batch_shape, -1)
    return jnp.concatenate([map_flat, inventory], axis=-1)

def visual_block_dropout(key: chex.PRNGKey, map_obs: chex.Array, block_size: int = 3) -> chex.Array:
    """Masks out a random square block of the visual map."""
    H, W, C = map_obs.shape
    key_x, key_y = jax.random.split(key)
    
    # Select top-left corner
    x = jax.random.randint(key_x, (), 0, W - block_size + 1)
    y = jax.random.randint(key_y, (), 0, H - block_size + 1)
    
    # Create spatial grid and mask
    yy, xx = jnp.mgrid[:H, :W]
    mask = (xx >= x) & (xx < x + block_size) & (yy >= y) & (yy < y + block_size)
    
    # Broadcast mask across channel dimension
    return jnp.where(mask[..., None], 0, map_obs)

def edge_dropout(key: chex.PRNGKey, map_obs: chex.Array, max_edge: int = 2) -> chex.Array:
    """Simulates a random crop by masking out random borders."""
    H, W, C = map_obs.shape
    keys = jax.random.split(key, 4)
    
    # Determine how much to mask on each side
    top = jax.random.randint(keys[0], (), 0, max_edge + 1)
    bottom = jax.random.randint(keys[1], (), 0, max_edge + 1)
    left = jax.random.randint(keys[2], (), 0, max_edge + 1)
    right = jax.random.randint(keys[3], (), 0, max_edge + 1)
    
    yy, xx = jnp.mgrid[:H, :W]
    mask = (yy < top) | (yy >= H - bottom) | (xx < left) | (xx >= W - right)
    
    return jnp.where(mask[..., None], 0, map_obs)

def visual_speckle(key: chex.PRNGKey, map_obs: chex.Array, prob: float = 0.05) -> chex.Array:
    """Randomly masks individual grid cells across the map."""
    # Apply mask spatially (H, W) so all channels in a chosen cell are zeroed out
    shape = map_obs.shape[:2] 
    mask = jax.random.bernoulli(key, prob, shape)
    
    return jnp.where(mask[..., None], 0, map_obs)

def inventory_dropout(key: chex.PRNGKey, inventory: chex.Array, prob: float = 0.1) -> chex.Array:
    """Randomly zeroes out values in the inventory."""
    mask = jax.random.bernoulli(key, prob, inventory.shape)
    return jnp.where(mask, 0, inventory)

def single_curl_augment(key: chex.PRNGKey, flat_obs: chex.Array) -> chex.Array:
    """
    Applies a random mixture of augmentations to a single flat observation.
    """
    map_obs, inv_obs = unflatten_obs(flat_obs)
    
    # Split key for different operations and probability checks
    k_block, k_edge, k_speck, k_inv, k_probs = jax.random.split(key, 5)
    prob_keys = jax.random.split(k_probs, 4)
    
    # Hyperparameters for application probabilities
    P_BLOCK = 0.3
    P_EDGE = 0.3
    P_SPECKLE = 0.2
    P_INV = 0.2

    # 1. Visual Block Dropout
    map_obs = jax.lax.cond(
        jax.random.bernoulli(prob_keys[0], P_BLOCK),
        lambda x: visual_block_dropout(k_block, x),
        lambda x: x,
        map_obs
    )
    
    # 2. Edge Dropout
    map_obs = jax.lax.cond(
        jax.random.bernoulli(prob_keys[1], P_EDGE),
        lambda x: edge_dropout(k_edge, x),
        lambda x: x,
        map_obs
    )
    
    # 3. Visual Speckle
    map_obs = jax.lax.cond(
        jax.random.bernoulli(prob_keys[2], P_SPECKLE),
        lambda x: visual_speckle(k_speck, x),
        lambda x: x,
        map_obs
    )
    
    # 4. Inventory Dropout
    inv_obs = jax.lax.cond(
        jax.random.bernoulli(prob_keys[3], P_INV),
        lambda x: inventory_dropout(k_inv, x),
        lambda x: x,
        inv_obs
    )
    
    return flatten_obs(map_obs, inv_obs)

# Vectorize the function so it can handle batches of observations natively
# in_axes=(0, 0) means it expects a batch of PRNGKeys and a batch of observations
batched_curl_augment = jax.vmap(single_curl_augment, in_axes=(0, 0))


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
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
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        class CURLActorCriticWrapper(nn.Module):
            action_dim: int
            layer_width: int
            latent_dim: int

            def setup(self):
                self.encoder = Encoder(self.latent_dim, self.layer_width)
                self.actor_critic = EmbeddingActorCritic(self.action_dim, self.layer_width)

            def __call__(self, x):
                z = self.encoder(x)
                return self.actor_critic(z)

        # INIT NETWORK
        if config.get("USE_CURL", False):
            assert "Symbolic" in config["ENV_NAME"], "CURL is only implemented for Symbolic Craftax"
            latent_dim = config.get("CURL_LATENT_SIZE", 64)
            network = CURLActorCriticWrapper(
                env.action_space(env_params).n, config["LAYER_SIZE"], latent_dim
            )
        elif "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
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

        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
            "curl_head": None,
            "curl_target_encoder_params": None,
        }

        if config.get("USE_CURL", False):
            curl_head = CURLHead(latent_dim)
            rng, _rng = jax.random.split(rng)
            init_z = jnp.zeros((1, latent_dim))
            curl_head_params = curl_head.init(_rng, init_z, init_z)
            
            # TrainState for the bilinear CURL head
            curl_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config.get("CURL_LR", config["LR"]), eps=1e-5),
            )
            ex_state["curl_head"] = TrainState.create(
                apply_fn=curl_head.apply,
                params=curl_head_params,
                tx=curl_tx,
            )
            
            # Initialize momentum target encoder with exact same weights as the active encoder.
            # Because of our wrapper module, the encoder params live inside the 'encoder' key.
            ex_state["curl_target_encoder_params"] = jax.tree.map(
                jnp.copy, network_params["params"]["encoder"]
            )

        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

            if config["USE_E3B"]:
                ex_state["e3b_matrix"] = (
                    jnp.repeat(
                        jnp.expand_dims(
                            jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                        ),
                        config["NUM_ENVS"],
                        axis=0,
                    )
                    / config["E3B_LAMBDA"]
                )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        m = config.get("CURL_FRAME_DELAY", 3)
        
        # Use max(1, m) so JAX doesn't crash if m=0. 
        # (If m=0, the buffer exists but we just ignore it later).
        ex_state["obs_history"] = jnp.repeat(
            jnp.expand_dims(obsv, axis=0), max(1, m), axis=0
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                # rng, _rng = jax.random.split(rng, 2)

                # # Apply data augmentation if CURL is enabled
                # if config.get("USE_CURL", False):
                #     aug_keys = jax.random.split(_aug_rng, config["NUM_ENVS"])
                #     active_obs = batched_curl_augment(aug_keys, last_obs)
                # else:
                #     active_obs = last_obs
                rng, _rng = jax.random.split(rng)

                # The policy needs the unaugmented observation.
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                reward_i = jnp.zeros(config["NUM_ENVS"])

                # 1. Grab the delayed observation (oldest in buffer)
                delayed_obs = ex_state["obs_history"][0]
                
                # 2. Shift the buffer to make room for the new observation
                new_history = jnp.roll(ex_state["obs_history"], shift=-1, axis=0)
                
                # 3. Insert the current 'last_obs' at the end of the buffer
                new_history = new_history.at[-1].set(last_obs)
                
                # 4. Handle episode resets! If an env is done, we don't want to contrast 
                # the new episode with the old one. Overwrite the history for that env 
                # with the newly reset observation.
                reset_history = jnp.repeat(jnp.expand_dims(obsv, axis=0), m, axis=0)
                ex_state["obs_history"] = jnp.where(
                    done[None, :, None], # broadcast across the buffer and obs dimensions
                    reset_history, 
                    new_history
                )

                if config["TRAIN_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]

                    if config["USE_E3B"]:
                        # Embedding is (NUM_ENVS, 128)
                        # e3b_matrix is (NUM_ENVS, 128, 128)
                        us = jax.vmap(jnp.matmul)(ex_state["e3b_matrix"], latent_obs)
                        bs = jax.vmap(jnp.dot)(latent_obs, us)

                        def update_c(c, b, u):
                            return c - (1.0 / (1 + b)) * jnp.outer(u, u)

                        updated_cs = jax.vmap(update_c)(ex_state["e3b_matrix"], bs, us)
                        new_cs = (
                            jnp.repeat(
                                jnp.expand_dims(
                                    jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                                ),
                                config["NUM_ENVS"],
                                axis=0,
                            )
                            / config["E3B_LAMBDA"]
                        )
                        ex_state["e3b_matrix"] = jnp.where(
                            done[:, None, None], new_cs, updated_cs
                        )

                        e3b_bonus = jnp.where(
                            done, jnp.zeros((config["NUM_ENVS"],)), bs
                        )

                        reward_i = e3b_bonus * config["E3B_REWARD_COEFF"]

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    delayed_obs=delayed_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            
            rng, _aug_rng = jax.random.split(rng)
            
            if config.get("USE_CURL", False):
                aug_keys = jax.random.split(_aug_rng, config["NUM_ENVS"])
                active_last_obs = batched_curl_augment(aug_keys, last_obs)
            else:
                active_last_obs = last_obs

            _, last_val = network.apply(train_state.params, active_last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
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
                # def _update_minbatch(carry, batch_info):
                #     train_state, ex_state, rng = carry  # Added rng to carry
                #     traj_batch, advantages, targets = batch_info

                #     # 2. DO AUGMENTATION OUTSIDE THE LOSS FUNCTION
                #     rng, _aug_rng = jax.random.split(rng)
                    
                #     # 3. FIX THE SHAPE MISMATCH
                #     batch_size = traj_batch.obs.shape[0]
                    
                #     if config.get("USE_CURL", False):
                #         aug_keys = jax.random.split(_aug_rng, batch_size)
                #         augmented_obs_q = batched_curl_augment(aug_keys, traj_batch.obs)
                #     else:
                #         augmented_obs_q = traj_batch.obs # Dummy fallback

                #     # Pass aug_obs_q into the loss function explicitly
                #     def _loss_fn(params, curl_params, traj_batch, gae, targets, target_encoder_params, aug_obs_q):
                #         # RERUN NETWORK
                #         pi, value = network.apply(params, traj_batch.obs) 
                #         log_prob = pi.log_prob(traj_batch.action)

                #         # CALCULATE VALUE LOSS
                #         value_pred_clipped = traj_batch.value + (
                #             value - traj_batch.value
                #         ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                #         value_losses = jnp.square(value - targets)
                #         value_losses_clipped = jnp.square(value_pred_clipped - targets)
                #         value_loss = (
                #             0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                #         )

                #         # CALCULATE ACTOR LOSS
                #         ratio = jnp.exp(log_prob - traj_batch.log_prob)
                #         gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                #         loss_actor1 = ratio * gae
                #         loss_actor2 = (
                #             jnp.clip(
                #                 ratio,
                #                 1.0 - config["CLIP_EPS"],
                #                 1.0 + config["CLIP_EPS"],
                #             )
                #             * gae
                #         )
                #         loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                #         loss_actor = loss_actor.mean()
                #         entropy = pi.entropy().mean()

                #         # 2. CALCULATE CURL (InfoNCE) LOSS
                #         curl_loss = 0.0
                #         if config.get("USE_CURL", False):
                #             encoder = Encoder(config.get("CURL_LATENT_SIZE", 64), config["LAYER_SIZE"])
                            
                #             # Split RNG to generate TWO independent augmentations from the clean observation
                #             rng, rng_q, rng_k = jax.random.split(rng, 3)
                #             aug_keys_q = jax.random.split(rng_q, batch_size)
                #             aug_keys_k = jax.random.split(rng_k, batch_size)
                            
                #             aug_obs_q = batched_curl_augment(aug_keys_q, traj_batch.obs)
                #             aug_obs_k = batched_curl_augment(aug_keys_k, traj_batch.obs)
                            
                #             # Query uses the active encoder
                #             z_q = encoder.apply({'params': params['params']['encoder']}, aug_obs_q)
                            
                #             # Key uses the target (momentum) encoder
                #             z_k = encoder.apply({'params': target_encoder_params}, aug_obs_k)
                #             z_k = jax.lax.stop_gradient(z_k)

                #             logits = ex_state["curl_head"].apply_fn(curl_params, z_q, z_k)
                #             labels = jnp.arange(logits.shape[0])
                #             curl_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

                #         total_loss = (
                #             loss_actor
                #             + config["VF_COEF"] * value_loss
                #             - config["ENT_COEF"] * entropy
                #             + config.get("CURL_COEF", 1.0) * curl_loss # Added safe fallback
                #         )
                #         return total_loss, (value_loss, loss_actor, entropy, curl_loss)

                #     if config.get("USE_CURL", False):
                #         grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                #         (total_loss, aux), grads = grad_fn(
                #             train_state.params,
                #             ex_state["curl_head"].params,
                #             traj_batch,
                #             advantages,
                #             targets,
                #             ex_state["curl_target_encoder_params"],
                #             augmented_obs_q
                #         )
                #         main_grads, curl_head_grads = grads
                #     else:
                #         grad_fn = jax.value_and_grad(_loss_fn, argnums=0, has_aux=True)
                #         (total_loss, aux), main_grads = grad_fn(
                #             train_state.params,
                #             None,
                #             traj_batch,
                #             advantages,
                #             targets,
                #             None,
                #             augmented_obs_q
                #         )

                def _update_minbatch(carry, batch_info):
                    train_state, ex_state, rng = carry
                    traj_batch, advantages, targets = batch_info

                    # 1. HANDLE ALL RNG AND AUGMENTATIONS OUTSIDE THE LOSS FUNCTION
                    rng, _aug_rng = jax.random.split(rng)
                    batch_size = traj_batch.obs.shape[0]
                    
                    if config.get("USE_CURL", False):
                        rng_q, rng_k = jax.random.split(_aug_rng, 2)
                        aug_keys_q = jax.random.split(rng_q, batch_size)
                        aug_keys_k = jax.random.split(rng_k, batch_size)
                        
                        aug_obs_q = batched_curl_augment(aug_keys_q, traj_batch.obs)
                        if config.get("CURL_FRAME_DELAY", 0) == 0:
                            base_obs_k = traj_batch.obs
                        else:
                            base_obs_k = traj_batch.delayed_obs
                        
                        aug_obs_k = batched_curl_augment(aug_keys_k, base_obs_k)
                    else:
                        # Dummy fallbacks so the function signature remains consistent
                        aug_obs_q = traj_batch.obs 
                        aug_obs_k = traj_batch.obs 

                    # 2. ADD aug_obs_k TO THE LOSS FUNCTION SIGNATURE
                    def _loss_fn(params, curl_params, traj_batch, gae, targets, target_encoder_params, aug_obs_q, aug_obs_k):
                        
                        # PPO PIPELINE (Using unaugmented observations)
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # 3. CURL LOSS (Using pre-computed augmented observations)
                        curl_loss = 0.0
                        if config.get("USE_CURL", False):
                            encoder = Encoder(config.get("CURL_LATENT_SIZE", 64), config["LAYER_SIZE"])
                            
                            # Query uses active encoder + aug_obs_q
                            z_q = encoder.apply({'params': params['params']['encoder']}, aug_obs_q)
                            
                            # Key uses target encoder + aug_obs_k
                            z_k = encoder.apply({'params': target_encoder_params}, aug_obs_k)
                            z_k = jax.lax.stop_gradient(z_k)

                            logits = ex_state["curl_head"].apply_fn(curl_params, z_q, z_k)
                            labels = jnp.arange(logits.shape[0])
                            curl_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config.get("CURL_COEF", 1.0) * curl_loss
                        )
                        return total_loss, (value_loss, loss_actor, entropy, curl_loss)

                    # 4. PASS THE NEW ARGUMENTS INTO THE GRAD_FN CALLS
                    if config.get("USE_CURL", False):
                        grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                        (total_loss, aux), grads = grad_fn(
                            train_state.params,
                            ex_state["curl_head"].params,
                            traj_batch,
                            advantages,
                            targets,
                            ex_state["curl_target_encoder_params"],
                            aug_obs_q,
                            aug_obs_k
                        )
                        main_grads, curl_head_grads = grads
                    else:
                        grad_fn = jax.value_and_grad(_loss_fn, argnums=0, has_aux=True)
                        (total_loss, aux), main_grads = grad_fn(
                            train_state.params,
                            None,
                            traj_batch,
                            advantages,
                            targets,
                            None,
                            aug_obs_q,
                            aug_obs_k
                        )

                    value_loss, loss_actor, entropy, curl_loss = aux

                    # Apply main gradients
                    train_state = train_state.apply_gradients(grads=main_grads)

                    # Update CURL specific states
                    if config.get("USE_CURL", False):
                        # Step the contrastive head
                        ex_state["curl_head"] = ex_state["curl_head"].apply_gradients(grads=curl_head_grads)
                        
                        # Soft momentum update for the target encoder using Optax
                        momentum = config.get("CURL_MOMENTUM", 0.999) 
                        ex_state["curl_target_encoder_params"] = optax.incremental_update(
                            new_tensors=train_state.params["params"]["encoder"],
                            old_tensors=ex_state["curl_target_encoder_params"],
                            step_size=1.0 - momentum
                        )

                    losses = (total_loss, value_loss, loss_actor, entropy, curl_loss)
                    return (train_state, ex_state, rng), losses

                (
                    train_state,
                    ex_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                # Notice we pass ex_state into the scan carry so it persists through minibatches
                (train_state, ex_state, rng), losses = jax.lax.scan(
                    _update_minbatch, (train_state, ex_state, rng), minibatches
                )
                
                update_state = (
                    train_state,
                    ex_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            # Initialize update_state with ex_state included
            update_state = (
                train_state,
                ex_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            ex_state = update_state[1]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            # Unpack the accumulated losses from the scan
            total_loss, value_loss, actor_loss, entropy, curl_loss = loss_info
            
            # Take the mean across epochs and minibatches and add to wandb metric dict
            metric["train/total_loss"] = total_loss.mean()
            metric["train/value_loss"] = value_loss.mean()
            metric["train/actor_loss"] = actor_loss.mean()
            metric["train/entropy"] = entropy.mean()
            metric["train/curl_loss"] = curl_loss.mean()

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    ex_state["icm_encoder"] = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    ex_state["icm_inverse"] = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
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

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
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

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
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
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)
    # CURL
    parser.add_argument("--use_curl", action="store_true")
    parser.add_argument("--curl_latent_size", type=int, default=64)
    parser.add_argument("--curl_lr", type=float, default=2e-4)
    parser.add_argument("--curl_momentum", type=float, default=0.999)
    parser.add_argument("--curl_coef", type=float, default=1.0)
    parser.add_argument("--curl_frame_delay", type=int, default=0)
    

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
