# DECO: Decentralized Coin-Betting for Online Learning

A Python implementation of decentralized coin-betting algorithms for online learning in multi-agent systems.
This repository contains the code for the paper "Decentralized Parameter-Free Online Learning".

## Project Structure

```
DECO/
  ├── README.md
  ├── Dockerfile
  ├── environment.yml
  ├── requirements.txt
  ├── run_all.bat
  ├── run_all.sh
  ├── .dockerignore
  ├── experiments/
  │   ├── run_connectivity_experiment.py
  │   ├── run_gossip_tradeoff_experiment.py
  │   ├── run_multi_dataset_comparison.py
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

### Method 1: Local Execution (Using Conda)

To run the code locally, it is recommended to use a virtual environment for consistency.
In these instructions we will use the Anaconda or Miniconda package manager.
1.  **Install Conda:**
    If you don't have Conda installed, you can download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  **Create and activate the Conda environment:**
    This command will create a new environment named `deco-env` with the exact dependencies listed in the `environment.yml` file.
    ```bash
    conda env create -f environment.yml
    conda activate deco-env
    ```
3.  **Download regression datasets:**
    The experiments require several public datasets. This script will download them into a `data/` directory.
    ```bash
    python src/deco/download_datasets.py
    ```
4.  **Run the script:**
    This single script will execute all experiments, save the results to the `results/` directory, and then generate the final plots in the `plots/Figs/` directory.

      * On **macOS or Linux**:
        ```bash
        bash run_all.sh
        ```
      * On **Windows**:
        ```bash
        run_all.bat
        ```

### Method 2: Docker (Recommended for Full Reproducibility)

This is the easiest and most reliable method. It uses Docker to build a self-contained image with all code, data, and dependencies, ensuring the results are identical regardless of your local machine's configuration.

**Prerequisite:** You must have [Docker installed](https://docs.docker.com/get-docker/).

1.  **Build the Docker image:**
    From the root directory of the project, run the following command. This will build an image named `deco-repro`.

    ```bash
    docker build -t deco-repro .
    ```

2.  **Run the container:**
    This command will run the entire workflow inside the container. It uses volumes (`-v`) to ensure the generated data and plots are saved directly to the `results/` and `plots/Figs/` folders on your local machine.
    **If you are on Windows, make sure to use PowerShell.**
    ```bash
    docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/plots/Figs:/app/plots/Figs" deco-repro
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
