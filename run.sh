#!/bin/bash

#SBATCH -p gpu
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --constraint=a100  # if you want a particular type of GPU

module load python cuda
source /mnt/home/yxu10/venv_10/bin/activate
cd $SLURM_SUBMIT_DIR

python single_event_pe.py --event "GW191216_213338" --gps 1260567236.4 --duration 4 --Mc_prior 3.0 30.0 --ifos "H1" "V1" --heterodyned True
# python single_event_pe.py --event "GW200129_065458" --gps 1264316116.4 --duration 8 --Mc_prior 14.0 49.0 --ifos "H1" "L1" "V1"
# python single_event_pe.py --event "GW200224_222234" --gps 1266618172.4 --duration 4 --Mc_prior 22.0 58.0 --ifos "H1" "L1" "V1"
# python single_event_pe.py --event "GW200112_155838" --gps 1262879936.1 --duration 4 --Mc_prior 21.0 50.0 --ifos "L1" "V1"
# python single_event_pe.py --event "GW200311_115853" --gps 1267963151.4 --duration 4 --Mc_prior 21.0 48.0 --ifos "H1" "L1" "V1"
# python single_event_pe.py --event "GW191204_171526" --gps 1259514944.1 --duration 8 --Mc_prior 8.0 11.0 --ifos "H1" "L1"
# python single_event_pe.py --event "GW191109_010717" --gps 1257296855.2 --duration 4 --Mc_prior 26.0 98.0 --ifos "H1" "L1"
# python single_event_pe.py --event "GW191222_033537" --gps 1261020955.1 --duration 8 --Mc_prior 29.0 83.0 --ifos "H1" "L1"
# python single_event_pe.py --event "GW200225_060421" --gps 1266645879.4 --duration 8 --Mc_prior 12.0 24.0 --ifos "H1" "L1"
