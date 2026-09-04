#!/bin/bash
# Convenience wrapper — choose dataset to train
# Usage:
#   bash shell/run_llm_training.sh mvtec
#   bash shell/run_llm_training.sh visa

case "${1:-mvtec}" in
    mvtec) bash "$(dirname "$0")/run_llm_training_mvtec.sh" ;;
    visa)  bash "$(dirname "$0")/run_llm_training_visa.sh" ;;
    *)     echo "Usage: $0 [mvtec|visa]"; exit 1 ;;
esac
