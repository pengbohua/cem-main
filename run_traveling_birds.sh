export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0

python -m experiments.run_experiments -c experiments/configs/travelingbirds.yaml

