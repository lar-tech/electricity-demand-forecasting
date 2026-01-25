# electricity-demand-forecasting
ML-based short-term (day-ahead) electricity demand forecasting.  
This repository contains code and experiments for training, tuning, and evaluating demand forecasting models.

The goal of this project is to forecast day-ahead electricity demand using machine learning models for our module "Project Lab Automation" at TU Berlin.
The current workflow includes:

1. Baseline estimator without exogenous features  
2. Estimator with exogenous features  
3. Hyperparameter tuning (`tuning.py`)  
4. Evaluation with tuned parameters  
5. Final evaluation with `refit=True`

## Repository Structure
```
.
├── main.ipynb        # Jupyter notebook for exploration and experiments
├── main.py           # Script entry point for running experiments
├── requirements.txt  # Python dependencies
├── dataset.csv       # Example dataset
├── core/             # Core logic
├── docs/             # Project Report, Progress Notes, Final Presentation
└── results/          # Generated results and outputs
```

## Installation
We tested our Code with Python 3.13.0 on a MacBook Pro M1 with 32GB RAM but most scripts are run on only 2-3 cores.

1.  Clone the repository
```bash
git clone https://github.com/lar-tech/electricity-demand-forecasting.git
cd electricity-demand-forecasting
```
2. Create and activate a virtual environment (macOS/Linux)
```
python -m venv .venv
source .venv/bin/activate # .venv\Scripts\Activate.ps1 for windows
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Quickstart
We recommend for exploration to take a look at the jupyter notebook and run the cells if interested or run the main.py directly. There we only showcase different models but not explain our thought process. 
A sample `dataset.csv` is included in the repository but you can create your own in `core/create_dataset.py` but be aware that it takes some time due to the api design of the `smard-api`