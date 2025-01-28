```
python train_back.py --group myoback_1 --num_envs 1 --learning_rate 0.0002 --clip_range 0.1 --seed 7
```
- `--group`: Wandb training group name
- `--num_envs`: Number of envs to train in parallel
- `--learning_rate`: learning rate for PPO
- `--clip_range`: clip range for PPO
- `--seed`: env seeding
