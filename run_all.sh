#!/bin/bash
# This script runs all experiments and stops if any command fails.
set -e

echo "--- (1/6) Running DGD Tuning Experiment ---"
python -m experiments.tune_dgd

echo ""
echo "--- (2/6) Running Synthetic Data Experiment ---"
python -m experiments.run_synthetic_experiment

echo ""
echo "--- (3/6) Running Multi-Dataset Comparison ---"
python -m experiments.run_multi_dataset_comparison

echo ""
echo "--- (4/6) Running Network Connectivity Experiment ---"
python -m experiments.run_connectivity_experiment

echo ""
echo "--- (5/6) Running Gossip Trade-off Experiment ---"
python -m experiments.run_gossip_tradeoff_experiment

echo ""
echo "--- (6/6) Running Review Baseline and Communication-Cost Experiment ---"
python -m experiments.run_review_baseline_experiment

echo ""
echo "All experiments completed successfully!"

echo "--- Generating Plots ---"
python -m plots.plot_script
echo "Plots generated successfully!"
