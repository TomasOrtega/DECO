@echo off
REM This script runs all experiments required to generate the data for the paper's plots.
REM It stops if any experiment fails.

echo --- (1/6) Running DGD Tuning Experiment ---
python -m experiments.tune_dgd
if %ERRORLEVEL% neq 0 (
    echo DGD tuning failed. Aborting.
    exit /b %ERRORLEVEL%
)

echo.
echo --- (2/6) Running Synthetic Data Experiment (Cycle and Complete Graphs) ---
python -m experiments.run_synthetic_experiment
if %ERRORLEVEL% neq 0 (
    echo Synthetic experiment failed. Aborting.
    exit /b %ERRORLEVEL%
)

echo.
echo --- (3/6) Running Multi-Dataset Comparison ---
python -m experiments.run_multi_dataset_comparison
if %ERRORLEVEL% neq 0 (
    echo Multi-dataset comparison failed. Aborting.
    exit /b %ERRORLEVEL%
)

echo.
echo --- (4/6) Running Network Connectivity Experiment ---
python -m experiments.run_connectivity_experiment
if %ERRORLEVEL% neq 0 (
    echo Connectivity experiment failed. Aborting.
    exit /b %ERRORLEVEL%
)

echo.
echo --- (5/6) Running Gossip Trade-off Experiment ---
python -m experiments.run_gossip_tradeoff_experiment
if %ERRORLEVEL% neq 0 (
    echo Gossip trade-off experiment failed. Aborting.
    exit /b %ERRORLEVEL%
)

echo.
echo --- (6/6) Running Review Baseline and Communication-Cost Experiment ---
python -m experiments.run_review_baseline_experiment
if %ERRORLEVEL% neq 0 (
    echo Review baseline experiment failed. Aborting.
    exit /b %ERRORLEVEL%
)

echo.
echo All experiments completed successfully!

echo --- Generating Plots ---
python -m plots.plot_script
if %ERRORLEVEL% neq 0 (
    echo Plot generation failed.
    exit /b %ERRORLEVEL%
)
echo Plots generated successfully!
