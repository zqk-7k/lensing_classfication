#!/bin/bash
set -e

mkdir -p ./logs
LOG_FILE="./logs/ablation_suite_final.log"

> ./logs/final_ablation_results.txt

DATASETS=("SIS" "PM")

echo "Starting Full Ablation Suite..." | tee -a $LOG_FILE

for DATASET in "${DATASETS[@]}"; do
    echo "=============================================" | tee -a $LOG_FILE
    echo " Running Ablations for Dataset: $DATASET" | tee -a $LOG_FILE
    echo "=============================================" | tee -a $LOG_FILE

    python main_ablation.py --dataset $DATASET --exp_name "PI-ResNet_Baseline" 2>&1 | tee -a $LOG_FILE
    python main_ablation.py --dataset $DATASET --no_physics_fusion --exp_name "wo_PhysicsFusion" 2>&1 | tee -a $LOG_FILE
    python main_ablation.py --dataset $DATASET --no_se --exp_name "wo_SE_Block" 2>&1 | tee -a $LOG_FILE
    python main_ablation.py --dataset $DATASET --use_snake --exp_name "w_Snake_Activation" 2>&1 | tee -a $LOG_FILE
done

echo "All Done! Check logs/final_ablation_results.txt" | tee -a $LOG_FILE
