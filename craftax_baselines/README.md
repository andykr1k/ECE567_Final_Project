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

# Visualisation
You can save trained policies with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).