#!/bin/bash
# This script runs all experiments and stops if any command fails.
set -e

echo "--- (1/7) Running Online Baseline Learning-Rate Sweeps ---"
uv run --locked --no-dev python -m experiments.tune_dgd

echo ""
echo "--- (2/7) Running Synthetic Data Experiment ---"
uv run --locked --no-dev python -m experiments.run_synthetic_experiment

echo ""
echo "--- (3/7) Running Multi-Dataset Comparison ---"
uv run --locked --no-dev python -m experiments.run_multi_dataset_comparison

echo ""
echo "--- (4/7) Running Network Connectivity Experiment ---"
uv run --locked --no-dev python -m experiments.run_connectivity_experiment

echo ""
echo "--- (5/7) Running Gossip Trade-off Experiment ---"
uv run --locked --no-dev python -m experiments.run_gossip_tradeoff_experiment

echo ""
echo "--- (6/7) Running Review Synthetic Baseline and Communication-Cost Experiment ---"
uv run --locked --no-dev python -m experiments.run_review_baseline_experiment

echo ""
echo "--- (7/7) Running Review Real-Data Baseline and Communication-Cost Experiment ---"
uv run --locked --no-dev python -m experiments.run_review_real_data_experiment

echo ""
echo "All experiments completed successfully!"

echo "--- Generating Plots ---"
uv run --locked --no-dev python -m plots.plot_script
echo "Plots generated successfully!"
