#!/bin/bash 
#SBATCH --account=def-durandau
#SBATCH --job-name=back_qpos_pelvis_up
#SBATCH --cpus-per-task=12
#SBATCH --time=0-23:00
#SBATCH --array=1
#SBATCH --mem=48G
#SBATCH --mail-user=huiyi.wang@mail.mcgill.ca
#SBATCH --mail-type=ALL

export PYTHONPATH="$PYTHONPATH:/home/cheryl16/projects/def-durandau/cheryl16/MyoBack"

cd /home/cheryl16/projects/def-durandau/cheryl16/MyoBack

module load StdEnv/2023
module load gcc opencv/4.9.0 cuda/12.2 python/3.10 mpi4py mujoco/3.1.6

source /home/cheryl16/py310/bin/activate

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

wandb offline

#parallel -j 10 python train_back.py --env_name 'myoStandingBack-v1' --group 'myoback_3' --num_envs 8 --learning_rate 0.0002 --clip_range 0.1 --seed ::: {1..10} 
parallel -j 10 python train_back.py --env_name 'myoTorsoReachFixed-v0' --group 'myoback_full_7' --num_envs 4 --learning_rate 0.0002 --clip_range 0.1 --seed ::: {1..10}
python train_full_body.py --env_name 'myoTorsoReachFixed-v1' --group 'fp_1' --num_envs 4 