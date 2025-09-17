## CartPole RL (Gymnasium + RLlib)

CartPole project using Gymnasium and Ray RLlib.

### Install

1. Create a virtual environment (recommended)
2. Install dependencies:
```
pip install -r requirements.txt
```

### Usage

Train a DQN agent and save checkpoints:
```
python -m cartpole_rl.cli train --iterations 100 --save-dir .
```

Run evaluation rollouts and save to JSON:
```
python -m cartpole_rl.cli rollout --checkpoint-dir . --episodes 10 --output rollout_data.json
```

### Package Structure

- `cartpole_rl/config.py`: Build preconfigured DQN config
- `cartpole_rl/train.py`: Training loop wrapper
- `cartpole_rl/rollout.py`: Evaluation and rollout recording
- `cartpole_rl/cli.py`: CLI for training and rollouts
