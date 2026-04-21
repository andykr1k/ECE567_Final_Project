<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax_Baselines/main/images/logo.png" />
</p>

# Craftax Baselines

This repository contains the code for running the baselines from the [Craftax paper](https://arxiv.org/abs/2402.16801).
For packaging reasons, this is separate to the [main repository](https://github.com/MichaelTMatthews/Craftax/).

# Installation
```commandline
git clone https://github.com/andykr1k/ECE567_Final_Project
cd ECE567_Final_Project/craftax_baselines
pip install -r requirements.txt
```

# Run Experiments

### PPO
```commandline
python ppo.py --seed 42
```

### PPO-RNN
```commandline
python ppo_rnn.py --seed 42 --save_policy --total_timesteps 5e9 --wandb_project craftax_baselines_new_jax
```

### ICM
```commandline
python ppo.py --train_icm --seed 42 --save_policy --total_timesteps 5e9 --wandb_project craftax_baselines_new_jax --icm_reward_coeff 0.001
```

### E3B
```commandline
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0.001
```

### RND
```commandline
python ppo_rnd.py --seed 42 --save_policy --total_timesteps 5e9 --wandb_project craftax_baselines_new_jax
```

### ICM Inv Only Tests
Regular ICM
```commandline
python ppo_inv_only_icm.py \
    --wandb_project craftax_inv_only_icm \
    --train_icm \
    --icm_reward_coeff 1.0 \
    --icm_inv_only \
```

E3B
```commandline
python ppo_inv_only_icm.py \
    --wandb_project craftax_inv_only_icm \
    --train_icm \
    --e3b_reward_coeff 0.001 \
    --icm_reward_coeff 0.0 \
    --use_e3b \
    --icm_inv_only \
```

### CURL
```
python ppo_curl.py \
    --wandb_project craftax_ppo_curl \
    --seed 43 \
    --use_curl \
    --total_timesteps 5e9
```

```
python ppo_curl.py \
    --wandb_project craftax_ppo_curl \
    --seed 43 \
    --total_timesteps 5e9
```

```
python ppo_curl.py \
    --wandb_project craftax_ppo_curl \
    --seed 43 \
    --use_curl \
    --train_icm \
    --e3b_reward_coeff 0.001 \
    --icm_reward_coeff 0.0 \
    --use_e3b \
    --total_timesteps 5e9
```

```
python ppo_curl.py \
    --wandb_project craftax_ppo_curl \
    --seed 43 \
    --use_curl \
    --curl_frame_delay 5 \
    --total_timesteps 5e9
```

### Random Latent Exploration
```
python ppo_rle.py \
    --wandb_project craftax_ppo_curl \
    --rle_alpha 0.1 \
    --rle_resample_freq 128 \
    --total_timesteps 5e9 \
    --seed 42
```

```
python ppo_rle_2.py \
    --wandb_project craftax_rle_2 \
    --seed 42
```

# COnv
```
python ppo.py --seed 42 --env_name "Craftax-Pixels-v1"
```

# Visualisation
You can save trained policies with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).