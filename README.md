# bzan554-project2

Goal: Identify the sensitivity of the algorithm to different kinds of tuning parameters.

| Variable | Type | Role |
|----------|-------------|----------|
| `sku` | Categorical | Input |
| `price` | Numerical | Input |
| `order` | Numerical | Input |
| `duration` | Numerical | Input |
| `category` | Categorical | Input |
| `quantity` | Numerical | Response |


| File / folder | Purpose |
|---|---|
| `AGENTS.md` | Assignment requirements and instructions for AI assistance |
| `README.md` | Project overview, structure, setup, and run instructions |
| `.gitignore` | Exclude confidential data, virtual environments, and temporary files from Git |
| `.python-version` | Specify the Python version used |
| `pyproject.toml` | Define project dependencies and configuration |
| `uv.lock` | Record exact dependency versions for reproducibility |
| `data/pricing.csv` | Source data used to predict quantity sold |
|---|---|
| `src/__init__.py` | Mark `src` as a Python package |
| `src/data.py` | Load data incrementally, split it, and preprocess inputs using training statistics |
| `src/model.py` | Build and compile a configurable neural network |
| `src/train.py` | Train a model with selected settings and return its learning history |
| `src/evaluate.py` | Calculate MSE, MAE, and R² for a specified data split |
|---|---|
| `experiments/__init__.py` | Mark `experiments` as a Python package |
| `experiments/baseline.py` | Run and record the reference model for comparisons |
| `experiments/batch_size.py` | Compare mini-batch sizes |
| `experiments/hidden_layers.py` | Compare the number of hidden layers |
| `experiments/hidden_neurons.py` | Compare neurons per hidden layer |
| `experiments/activation_functions.py` | Compare hidden-layer activation functions |
| `experiments/optimizers.py` | Compare optimization algorithms |
| `experiments/learning_rates.py` | Compare learning rates |
|---|---|
| `results/metrics.csv` | Store each run’s settings, metrics, seed, and training time |
| `results/histories/` | Store training and validation metrics by epoch for each run |
| `results/figures/` | Store learning curves and tuning comparison charts |
| `presentation/group_ID_assignment2.pptx` | Present tuning insights, training/testing details, and team contributions |

bzan554-project2/
├── AGENTS.md
├── README.md
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
│
├── data/
│   └── pricing.csv
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
│   ├── __init__.py
│   ├── baseline.py
│   ├── batch_size.py
│   ├── hidden_layers.py
│   ├── hidden_neurons.py
│   ├── activation_functions.py
│   ├── optimizers.py
│   └── learning_rates.py
│
├── results/
│   ├── metrics.csv
│   ├── histories/
│   └── figures/
│
└── presentation/
    └── group_ID_assignment2.pptx
