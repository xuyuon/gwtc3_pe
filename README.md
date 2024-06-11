To run a single event: <br />
`sbatch run.sh`
<br />

To run the programme: <br />
`sbatch -p gpu --gpus-per-task=4 --cpus-per-task=32 --ntasks=1 --mem=128G -N5 disBatch run_disBatch.sh` <br />
