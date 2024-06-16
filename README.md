To run a single event: <br />
`sbatch run.sh`
<br />

To generate a configuration file: <br />
`python gen_config.py`
<br />

To run all events in gwtc3: <br />
`sbatch run_all.sh`
<br />

To run the programme: <br />
`sbatch -p gpu --gpus-per-task=4 --cpus-per-task=32 --ntasks=1 --mem=128G -N5 disBatch run_disBatch.sh` <br />


##Milestone
* single event parameter estimation
- [ ] Using same prior as bilby (i.e. Uniform component mass on chirp mass)
- [x] Getting reasonable posterior for high SNR events

* GWTC-3 events parameter estimation
- [x] Run parameter estimation automatically on multiple events
- [ ] Run parameter estimation using 
