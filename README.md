# DECO: Decentralized Coin-Betting for Online Learning

A Python implementation of decentralized coin-betting algorithms for online learning in multi-agent systems.
This repository contains the code for the paper "Decentralized Parameter-Free Online Learning".

## Project Structure

```
DECO/
  ├── README.md
  ├── Dockerfile
  ├── pyproject.toml
  ├── uv.lock
  ├── run_all.bat
  ├── run_all.sh
  ├── .dockerignore
  ├── experiments/
  │   ├── run_connectivity_experiment.py
  │   ├── run_gossip_tradeoff_experiment.py
  │   ├── run_multi_dataset_comparison.py
  │   ├── run_review_baseline_experiment.py
  │   ├── run_review_real_data_experiment.py
  │   ├── run_synthetic_experiment.py
  │   └── tune_dgd.py
  ├── plots/
  │   └── plot_script.py
  └── src/
      └── deco/
          ├── agents.py
          ├── algorithms.py
          ├── download_datasets.py
          ├── environments.py
          ├── graph.py
          ├── metrics.py
          ├── potentials.py
          └── utils.py
```

## Installation & Setup

You can reproduce all figures from the paper using one of the two methods below. The Docker method is recommended for guaranteeing an identical environment.
For either method, you need to **clone the repository** first
```bash
    git clone https://github.com/TomasOrtega/DECO.git
    cd DECO
```
Next, follow the instructions for one of the two methods:

### Method 1: Local Execution (Using uv)

1.  **Install uv:**
    Follow the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available.

2.  **Create the environment and install the locked dependencies:**
    ```bash
    uv sync --locked
    ```
3.  **Download regression datasets:**
    The original LIBSVM experiments require several public datasets. This script will download them into a `data/` directory.
    ```bash
    uv run --locked python src/deco/download_datasets.py
    ```
    The review real-data experiment below also supports bundled scikit-learn datasets, so it can run even when external dataset mirrors are unavailable.

4.  **Run the full workflow:**
    This single script will execute all experiments, save the results to the `results/` directory, and then generate the final manuscript plots in the `tex/Figs/` directory.

      * On **macOS or Linux**:
        ```bash
        bash run_all.sh
        ```
      * On **Windows**:
        ```bash
        run_all.bat
        ```

5.  **Install the development hooks (optional):**
    The hooks run Ruff's linter and formatter through `prek`.
    ```bash
    uv run --locked prek install
    ```

### Review Baseline and Communication-Cost Experiments

The review-response revision adds two focused experiment scripts.

`experiments/run_review_baseline_experiment.py` compares DECO against three representative online decentralized adaptive-gradient baselines: AdaGrad, RMSProp, and Adam. Each method predicts, observes one local gradient, updates its local optimizer state, and then gossips its decision vector. The script evaluates multiple base learning-rate scales and reports cumulative network loss and scalar communication cost, distinguishing DECO-i's `(wealth, G)` messages from DECO-ii's `G`-only messages.

`experiments/run_review_real_data_experiment.py` repeats the same comparison on real datasets. It defaults to bundled scikit-learn datasets (`diabetes`, `breast_cancer`, and `digits`) for offline/Docker reproducibility while preserving the existing LIBSVM dataset support in `RealDataEnvironment`.

The lightweight default runs are included in `run_all.sh` and `run_all.bat`. To run larger versions manually, increase the number of rounds or seeds:

```bash
uv run --locked python -m experiments.run_review_baseline_experiment --T 1000 --seeds 0 1 2
uv run --locked python -m experiments.run_review_real_data_experiment --T 1000 --seeds 0 1 2 --datasets diabetes breast_cancer digits
```

The summaries are written to `results/review_baseline_summary.csv` and `results/review_real_data_summary.csv` with the initial learning rate, cumulative network loss, average network loss, cumulative local loss, and total communicated scalars. The full workflow also regenerates the manuscript PDFs directly under `tex/Figs/`.

### Method 2: Docker (Recommended for Full Reproducibility)

This is the easiest and most reliable method. It uses Docker to build a self-contained image with all code, data, and dependencies, ensuring the results are identical regardless of your local machine's configuration.

**Prerequisite:** You must have [Docker installed](https://docs.docker.com/get-docker/).

1.  **Build the Docker image:**
    From the root directory of the project, run the following command. This will build an image named `deco-repro`.

    ```bash
    docker build -t deco-repro .
    ```

2.  **Run the container:**
    This command will run the entire workflow inside the container. It uses volumes (`-v`) to ensure the generated data and plots are saved directly to the `results/` and `tex/Figs/` folders on your local machine.
    **If you are on Windows, make sure to use PowerShell.**
    ```bash
    docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/tex/Figs:/app/tex/Figs" deco-repro
    ```

3.  **Run only the review experiments inside Docker:**
    ```bash
    docker run --rm -v "$(pwd)/results:/app/results" deco-repro \
        uv run --locked --no-dev python -m experiments.run_review_baseline_experiment --T 1000 --seeds 0 1 2

    docker run --rm -v "$(pwd)/results:/app/results" deco-repro \
        uv run --locked --no-dev python -m experiments.run_review_real_data_experiment --T 1000 --seeds 0 1 2 --datasets diabetes breast_cancer digits
    ```

After the command completes, all results and figures will be available in their respective local directories.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ortega2025decentralizedparameterfreeonlinelearning,
      title={Decentralized Parameter-Free Online Learning}, 
      author={Tomas Ortega and Hamid Jafarkhani},
      year={2025},
      eprint={2510.15644},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15644}, 
}
```
