#!/bin/bash
# This script runs all experiments and stops if any command fails.
set -e

echo "--- (1/5) Running DGD Tuning Experiment ---"
python -m experiments.tune_dgd

echo ""
echo "--- (2/5) Running Synthetic Data Experiment ---"
python -m experiments.run_synthetic_experiment

echo ""
echo "--- (3/5) Running Multi-Dataset Comparison ---"
python -m experiments.run_multi_dataset_comparison

echo ""
echo "--- (4/5) Running Network Connectivity Experiment ---"
python -m experiments.run_connectivity_experiment

echo ""
echo "--- (5/5) Running Gossip Trade-off Experiment ---"
python -m experiments.run_gossip_tradeoff_experiment

echo ""
echo "All experiments completed successfully!"

echo "--- Generating Plots ---"
python -m plots.plot_script
echo "Plots generated successfully!"