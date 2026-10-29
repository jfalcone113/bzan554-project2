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

| File / Folder | Purpose | Status |
|---|---|---|
| `AGENTS.md` | Assignment requirements and instructions for AI assistance | Keep |
| `README.md` | Project overview, structure, setup, and execution instructions | Keep |
| `.gitignore` | Exclude confidential data, virtual environments, and generated files from Git | Keep |
| `.python-version` | Specify the Python version used | Keep |
| `pyproject.toml` | Define project dependencies and configuration | Keep |
| `uv.lock` | Record exact dependency versions for reproducibility | Keep |
| `data/pricing.csv` | Source data used to predict quantity sold | Keep |
| **Source Code** | | |
| `src/__init__.py` | Mark `src` as a Python package | Keep |
| `src/data.py` | Load data incrementally, split observations, preprocess inputs, and encode SKUs | Implemented |
| `src/model.py` | Build and compile configurable neural networks, including SKU embeddings | Implemented |
| `src/train.py` | Train incrementally, validate, track metrics, and restore the best model weights | Implemented |
| `src/evaluate.py` | Calculate MSE, MAE, and R² on the original response scale | Implemented |
| `src/tensorboard_logger.py` | Create TensorBoard writers and record training and validation metrics | Implemented |
| `src/visualize.py` | Generate training curves and hyperparameter comparison figures | To implement |
| **Experiments** | | |
| `experiments/__init__.py` | Mark `experiments` as a Python package | Keep |
| `experiments/utils.py` | Execute experiments and save configurations, metrics, and training histories | To implement |
| `experiments/baseline.py` | Run and record the reference model for comparison | To implement |
| `experiments/batch_size.py` | Compare mini-batch sizes | To implement |
| `experiments/hidden_layers.py` | Compare the number of hidden layers | To implement |
| `experiments/hidden_neurons.py` | Compare the number of neurons per hidden layer | To implement |
| `experiments/activation_functions.py` | Compare hidden-layer activation functions | To implement |
| `experiments/optimizers.py` | Compare optimization algorithms | To implement |
| `experiments/learning_rates.py` | Compare learning rates | To implement |
| **Outputs** | | |
| `logs/` | Store TensorBoard event files for visualizing and comparing training runs | Implemented |
| `results/metrics.csv` | Store experiment configurations, evaluation metrics, random seeds, and training times | To implement |
| `results/histories/` | Store epoch-by-epoch training and validation metrics for each experiment | To implement |
| `results/figures/` | Store learning curves and hyperparameter comparison charts | To implement |
| `presentation/group_ID_assignment2.pptx` | Present model tuning, training/testing results, and team contributions | Keep |

bzan554-project2/
│
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
│   ├── evaluate.py
│   ├── tensorboard_logger.py
│   └── visualize.py
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
├── logs/
│   ├── adam_sigmoid_...
│   ├── adam_relu_...
│   └── sgd_sigmoid_...
│
├── results/
│   ├── metrics.csv
│   ├── histories/
│   └── figures/
│
└── presentation/
    └── group_ID_assignment2.pptx