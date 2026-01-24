export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
mkdir -p results/cub_complete/
set -o pipefail
python -u -m experiments.run_experiments -c experiments/configs/cub.yaml 2>&1 | tee results/cub_complete/log.txt

