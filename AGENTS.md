# BZAN 554 model-tuning context

Use this course context when helping tune a neural network. Instructor: Michel Ballings, Deep Learning for Business Applications. Scope: the three supplied decks, not a complete inventory of the entire course.

## Read first

- `resources/MODEL_TUNING_REFERENCE.md`: page-cited inventory of model choices, training controls, diagnostics, and lecture examples.
- `resources/CLASS_RESOURCES.md`: supplied materials, referenced but missing class files, and every external hyperlink extracted from the three decks.
- Primary sources: `resources/introduction (2).pdf` (50 pages), `resources/fundamentals.pdf` (212 pages), `resources/deep.pdf` (105 pages). Citations use PDF page numbers, matching slide numbers.
- `resources/class-notes-text/`: searchable, page-labeled extractions of all three PDFs. These are verbatim machine extractions, not corrected notes; inspect PDFs for equations, figures, and code formatting.
- `resources/deep-slide-notes.md` and `resources/deep-slide-notes.pdf` are pre-existing secondary study aids. Prefer the primary slides when determining what was taught.

## Course inventory

- Architecture: dense/fully connected MLPs; number of hidden layers; neurons per layer; depth versus width; input/output shapes; biases; single/multiple continuous responses; binary, multi-label, and multi-class responses; categorical embeddings and embedding dimension.
- Hidden activations: sigmoid, tanh, ReLU, Leaky ReLU, PReLU, ELU, Softplus, Swish/SiLU, and squared ReLU (ReLU2). Leaky ReLU slope and ELU alpha are hyperparameters; PReLU slopes are learned parameters.
- Output/loss pairs: linear plus MSE/MAE/Huber for regression; sigmoid plus binary cross-entropy for binary/multi-label classification; softmax plus categorical or sparse categorical cross-entropy for mutually exclusive classes. Output restrictions can motivate Softplus, sigmoid, or tanh for regression.
- Training: batch size; full-batch, single-instance SGD, mini-batch GD; epochs; learning rate; plain SGD, momentum, Nesterov, AdaGrad, RMSprop, Adam; exponential learning-rate scheduling; backpropagation/autodiff; numerical versus analytical derivatives.
- Stability: normalized, Glorot/Xavier, and He initialization; uniform/normal distributions; activation saturation; vanishing/exploding gradients; batch normalization before/after activation, moving-statistic momentum, epsilon, scale/shift; gradient value/norm clipping.
- Generalization: early stopping and patience; best checkpoints; L1/L2/elastic net; dropout rate and placement; Monte Carlo dropout; realistic data augmentation; bias/variance and training versus held-out error; sample size and capacity.
- Interpretation/foundations: permutation importance, partial dependence, bootstrap diagnostics, PAC bounds, perceptron, matrix multiplication, tensors/dataflow graphs, incremental/streaming learning with FIFO buffers, TensorFlow Python and Keras Functional API.
- Mentioned only: RNNs, sequence-to-sequence, attention, CNNs, Transformer/Primer, AlexNet, residual networks. No detailed sequence architecture deck was supplied. LSTM/GRU-specific tuning is not established by these sources.

## How to assist with tuning

1. Inspect the user's code/data and establish target type, evaluation metric, feature types/shapes, split design, baseline, software versions, and compute budget. Request only missing information that affects the work.
2. Prefer TensorFlow Python with `tf.keras` Functional API to match class examples. Follow an existing project framework if the user already uses another one.
3. Start with a reproducible baseline and validate preprocessing, labels, output activation, and loss before enlarging the search. Fit preprocessing on training data only. Use validation for tuning and reserve test data for final evaluation; use time/group-aware splits when required by the data.
4. Propose a bounded search using course-covered controls. Label any suggested ranges not explicitly given in the slides as proposed experiments. Slide examples and activation/optimizer rankings are not universal best settings.
5. Diagnose from learning curves: high training and validation error can indicate underfitting or optimization failure; low training error with worse validation error suggests overfitting. Investigate unstable/nonfinite training before adding capacity.
6. Record configuration, seed, split, monitored metric, best epoch, validation score, runtime, parameter count, and checkpoint for each run. Compare with a consistent budget and repeat promising configurations across seeds when feasible.
7. Keep the best validation model. Distinguish checkpoint selection from stopping, and restored weights from restored optimizer state. Report measured results separately from hypotheses.
8. Cite the deck and page for course claims. Clearly label implementation clarifications and outside-course additions. Check the installed API or current official documentation before writing version-sensitive code; do not silently copy malformed PDF code.
9. If sequence/image-specific tuning is needed, identify the missing course material and make useful progress from available inputs without pretending those details were taught here.

## Import into another session

For a repository session, copy this file and the companion references into the project root; preserve existing project instructions when integrating. For an upload-based chat, attach these three Markdown files; attach source PDFs or the extracted text when exact source details are needed. A ZIP is a transport bundle: extract it before use. Ask the next assistant to read `AGENTS.md` first. The original course-context handoff predated the selected model; see the current active project layout below.

## Current active project layout

The selected standalone model and its exact preprocessing are under `model/`. Use `model/pricing_model.py` to train, predict, evaluate, or verify. The current presentation is `presentation/Presentation.qmd` / `.pdf`; regenerate it with `presentation/build_slides.py`. Source data remains at `data/pricing.csv`. All course PDFs and tuning references are now under `resources/`. Superseded experiment folders were removed after the selected model reproduced both predictions and from-scratch training. The full numeric trial log is preserved at `presentation/all_runs.csv`.

## Expanded tuning result

The improved selected bundle is `tuning/best_model/`, using RMSprop and L2 0.00001 with the same 256/128/64 sigmoid network, 5% hidden dropout, 50% SKU dropout, and linear output. Use the existing generic prediction CLI with `--bundle tuning/best_model`; `tuning/retrain_best.py` reproduces only the winning configuration. The original `model/` remains the frozen baseline. See `tuning/IMPROVEMENT_LOG.md`, `RESULTS.md`, and `all_runs.csv` for the matched-budget confirmation and all losing trials. The user accepted a measurable gain; the original +0.01 threshold was superseded explicitly. The current presentation includes only improving steps.
