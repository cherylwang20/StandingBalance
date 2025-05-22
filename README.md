## Demo

<img src="videos/SI_movie.gif" width="600">

## Download Pretrained Standing Balance Policies in Simulation

Pretrained standing balance policies for use in simulation can be downloaded as a single ZIP archive:

- [Download Pretrained Policies (ZIP)](https://mcgill-my.sharepoint.com/:u:/g/personal/huiyi_wang_mcgill_ca/EXPpRqGm5QVOkrQuSRjeOzYBNjmkWX-j4nHxhmMcp4DMOQ?e=DKqDms)

### Folder Descriptions

After extracting the archive, you will find the following folders, each corresponding to a different virtual agent condition:


- **MSR_pretrained/**  
  Reference standing plicies for extracting Muscle Synergy Representation (MSR)

- **healthy/**  
  Pretrained policy for the healthy musculoskeletal model.

- **Sarco_80/**  
  Pretrained policy for the mild sarcopenia agent.

- **Sarco_60/**  
  Pretrained policy for the moderate sarcopenia agent.

- **Sarco_40/**  
  Pretrained policy for the severe sarcopenia agent.

If you use these pretrained policies in your research or projects, please cite this repository or contact the authors for further information.


## Human Standing Experiment Data

Data from the human standing experiments can be downloaded from the following links:

- Elderly: [Download Dataset 1](https://mcgill-my.sharepoint.com/:u:/g/personal/huiyi_wang_mcgill_ca/EUZgLtCoiqhEk_RNkDK3iMkBkitHmJm5ylNRvazxXKsndA?e=Kwe3Ke)
- Young: [Download Dataset 2](https://mcgill-my.sharepoint.com/:u:/g/personal/huiyi_wang_mcgill_ca/ETkbJZyX7ZpFlYT3xbJXWI8Bn_hmwEKocIkhYikC6lBKCg?e=AupJDO)

These datasets include the raw and processed data used in the experiments described in this repository.

For more details about the experiments, please refer to the following papers:

- [Afschrift et al. 2018](https://doi.org/10.1016/j.gaitpost.2017.10.003)
- [Afschrift et al. 2016](https://doi.org/10.1152/jn.00127.2016)

**If you use these data in your research, please cite this project or contact the authors for additional information.**

---

## Training a Policy

To train a standing balance policy, use the following command:

```bash
python MSR_extraction/train_syn.py --group balancing_1 --num_envs 8 --learning_rate 0.0002 --clip_range 0.1 --seed 7
```

**Arguments:**
- `--group`: WandB training group name.
- `--num_envs`: Number of parallel environments for training.
- `--learning_rate`: Learning rate for PPO.
- `--clip_range`: Clip range for PPO.
- `--seed`: Random seed for reproducibility.

---

## Loading a Pretrained Policy

To evaluate a pretrained policy, use:

```bash
python MSR_extraction/eval_syn.py --env_name "myoTorsoBalance_v0" --policy_name "Healthy"
```

For sarcopenia agents, set `--env_name` to `"myoSarcTorsoBalance_v0"` and `--policy_name` to the corresponding sarcopenia model (`Sarco_80`, `Sarco_60`, or `Sarco_40`).

**Important:**  
To change the sarcopenia severity level, you must also manually update the level inside  
`myosuite/envs/myo/base_v0.py`.  
Find the parameter or setting related to sarcopenia level, and set it to the desired value (`0.8`, `0.6`, or `0.4`) before running the evaluation script.

**Example for sarcopenia:**
```bash
python MSR_extraction/eval_syn.py --env_name "myoSarcTorsoBalance_v0" --policy_name "Sarco_80"
```

Make sure the sarcopenia level in `base_v0.py` matches the policy you want to evaluate.

---
