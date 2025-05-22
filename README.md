## Human Standing Experiment Data

Data from the human standing experiments can be downloaded from the following links:

- Elderly: [Download Dataset 1](https://mcgill-my.sharepoint.com/:u:/g/personal/huiyi_wang_mcgill_ca/EUZgLtCoiqhEk_RNkDK3iMkBkitHmJm5ylNRvazxXKsndA?e=Kwe3Ke)
- Young: [Download Dataset 2](https://mcgill-my.sharepoint.com/:u:/g/personal/huiyi_wang_mcgill_ca/ETkbJZyX7ZpFlYT3xbJXWI8Bn_hmwEKocIkhYikC6lBKCg?e=AupJDO)

These datasets include the raw and processed data used in the experiments described in this repository.

For more details about the experiments, please refer to the following papers:

- [Afschrift et al. 2018](https://doi.org/10.1016/j.gaitpost.2017.10.003)
- [Afschrift et al. 2016](https://doi.org/10.1152/jn.00127.2016)

**If you use these data in your research, please cite this project or contact the authors for additional information.**

```
python train_back.py --group myoback_1 --num_envs 8 --learning_rate 0.0002 --clip_range 0.1 --seed 7
```
- `--group`: Wandb training group name
- `--num_envs`: Number of envs to train in parallel
- `--learning_rate`: learning rate for PPO
- `--clip_range`: clip range for PPO
- `--seed`: env seeding
