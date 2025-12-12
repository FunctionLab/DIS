module add disBatch

. /mnt/home/yhao1/miniforge3/envs/dis/bin/activate python

sbatch -p ccb -n 2 -c 20 disBatch test/predict/test_var_predict.sh --prefix test/predict/
