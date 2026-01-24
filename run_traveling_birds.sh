export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
mkdir -p results/travelingbirds_complete
set -o pipefail
python -u -m experiments.run_experiments -c experiments/configs/travelingbirds.yaml 2>&1 | tee results/travelingbirds_complete/log.txt

