export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
python -m experiments.run_eval -c experiments/configs/celeba_cem2.yaml --rerun
#python -m experiments.run_eval -c experiments/configs/celeba_cem1.yaml --rerun

