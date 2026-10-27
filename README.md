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


| **File** | **Purpose** |
|----------|-------------|
| `data.py` | Load, preprocess, encode, and split the data |
| `model.py` | Define a configurable neural network architecture |
| `train.py` | Train models using the selected hyperparameters |
| `evaluate.py` | Calculate performance metrics and record results |
| `baseline.py` | Reproduce the original model |
| `batch_size.py` | Compare different batch sizes |
| `hidden_layers.py` | Compare different network depths |
| `hidden_neurons.py` | Compare different network widths |
| `activation_functions.py` | Compare sigmoid, tanh, ReLU, etc. |
| `optimizers.py` | Compare optimization algorithms |
| `learning_rates.py` | Analyze learning-rate sensitivity |
| `metrics.csv` | Store experimental results |


bzan554-project2/
│
├── README.md
│
├── data/
│   └── pricing.csv
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
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
│   └── figures/
│
├── presentation/
│
├── requirements.txt
│
└── .gitignore