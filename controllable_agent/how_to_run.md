# How to Run

## Env Setup

```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
uv pip install ./third_party/gym-0.21.0
grep -v '^gym==0.21.0$' requirements. requirements.no-gym.txt
uv pip install -r requirements.no-gym.txt
uv pip install "mujoco==2.3.7"
uv pip install -r requirements.no-gym.txt
```

## Run Pretraining

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1 # or other GPU ID
python -m url_benchmark.pretrain   agent=diayn   task=walker_walk   use_wandb=1   use_tb=1   seed=1
```
