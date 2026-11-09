# BZAN 554: model-tuning inventory

Prepared from Michel Ballings's `introduction (2).pdf`, `fundamentals.pdf`, and `deep.pdf`. Abbreviations: **I** = introduction, **F** = fundamentals, **D** = deep. All references below are PDF pages, which match slide numbers. This inventory covers these supplied notes; referenced files have not been read unless present locally.

## Architecture and representation

| Choice/concept | What to vary or check | Source |
|---|---|---|
| Hidden-layer count | Number of nonlinear transformations; compare depth against validation quality and training cost. No universal count is prescribed. | I 4-11; F 27, 58, 108; D 103-104 |
| Hidden neurons | Units per layer and width pattern; affects capacity, parameter count, and parallel computation. No universal width is prescribed. | I 5-10; D 105 |
| Dense connections and biases | Fully connected layers; correct matrix shapes and one bias per receiving unit. Dense parameter count with bias is `(inputs + 1) * units` (derived from lecture shapes). | I 5-11; F 14-15 |
| Nonlinear hidden activations | Affine layers without intervening nonlinearities collapse into one affine mapping. | F 105, 108, 111-113 |
| Output count/type | Choose from single/multiple regression, binary, independent multi-label, or mutually exclusive multi-class targets. | F 27-31, 58, 78-100 |
| Embeddings | Category vocabulary/input dimension, dense embedding dimension, number of IDs per observation, padding/shape, and how multiple embeddings are combined. Lecture example: 20 categories, 2 dimensions, four IDs for K-hot input. | D 3-4 |
| Depth versus width | Depth builds compositional abstractions; width represents feature co-occurrences and permits more parallel operations. Both alter capacity and need validation. | D 103-105 |

Embedding clarification: lookup returns one vector per ID. To reproduce a K-hot linear weighted sum, sum the selected vectors; concatenation/averaging are different representations. A dense layer's bias and embedding output shape must also be accounted for. This is an implementation clarification, not an additional architecture taught by the deck.

## Hidden activation functions

Here `x` is the preactivation, `sigma` is sigmoid, and `alpha` is a negative-branch coefficient.

| Function | Definition | Tuning relevance | Source |
|---|---|---|---|
| Sigmoid | `1 / (1 + exp(-x))` | Bounded in (0,1); saturates at either extreme; derivative `s*(1-s)`. Hidden-layer saturation can slow deep training. | F 61-62, 116; D 41-44 |
| Tanh | `2*sigmoid(2*x)-1` | Zero-centered, bounded in (-1,1); derivative `1-tanh(x)^2`; still saturates. | F 119-122 |
| ReLU | `max(0,x)` | Cheap; positive derivative 1; zero negative derivative can create dead units. | F 125-128 |
| Leaky ReLU | `x` if positive, otherwise `alpha*x` | Fixed slope is a hyperparameter; prose uses 0.01, code uses 0.2. Keeps negative-branch gradients for nonzero slope. | F 131-133 |
| PReLU | ReLU variant with learned negative slope | Alpha is learned during training, rather than selected as a fixed Leaky ReLU slope; adds parameters and potential overfitting. | F 136-137 |
| ELU | `x` if positive, otherwise `alpha*(exp(x)-1)` | Alpha is adjustable; example 1.0. Negative branch saturates; exponential costs more than ReLU. Range is (-alpha,infinity), not always (-1,infinity). | F 140-142; D 61 |
| Softplus | `log(1+exp(x))` | Smooth positive output; derivative sigmoid; small gradients for very negative inputs. Useful for positive regression output constraints. | F 27, 145-147 |
| Swish / SiLU | `x*sigmoid(beta*x)` | Deck uses beta=1 (SiLU). Smooth, non-monotonic; large negative inputs still have tiny gradients. General beta changes gating. | F 150-153 |
| Squared ReLU / ReLU2 | `max(0,x)^2` | Derivative `2*max(0,x)`; still dead on negative side; large positive activations/gradients can amplify instability. Square after ReLU, not before. | F 156-159 |

F 162 gives this **lecture heuristic**, not a guaranteed ranking: Swish/ELU/ReLU2 > Leaky ReLU/PReLU > Softplus/ReLU > tanh > sigmoid. D 61 contains an earlier ELU > PReLU > ReLU > tanh > sigmoid comparison. Choose by validation and training cost. Heaviside and sign activations are discussed for the historical perceptron (F 14-22), not as modern backpropagation hidden-layer choices. Linear and softmax are also covered as output activations.

## Output layers, labels, and losses

| Task | Outputs / activation | Loss and checks | Source |
|---|---|---|---|
| Single continuous target | One linear output | MSE, MAE, or Huber; choose based on errors/outliers. | F 27-28, 34-53 |
| Multiple continuous targets | One output per target, typically linear | Aggregate losses across responses; check relative target scales. | F 31 |
| Constrained continuous target | Softplus for positive, sigmoid for (0,1), tanh for (-1,1) | Constraints should match the target; these choices do not make the task classification. | F 27 |
| Binary target | One sigmoid output | Binary cross-entropy/log loss; positive-class probability, negative probability `1-p`. | F 58, 61-75 |
| Multi-label target | One sigmoid output per label | Binary cross-entropy aggregated across labels; labels can coexist. | F 78, 81, 84, 100 |
| Multi-class target | One softmax output per class | Categorical cross-entropy for one-hot targets or sparse categorical cross-entropy for integer class IDs; one class per instance. | F 87, 90-100 |

- **MSE:** mean squared residual; emphasizes large errors and is sensitive to outliers. **MAE:** mean absolute residual; more robust, with a nondifferentiable zero-residual point handled by a subgradient. **Huber:** quadratic for `abs(error) <= delta`, linear beyond; tune positive delta relative to the target's scale. F 27 mentions 3 as an illustration; F 50 uses delta=5. (F 34-53.)
- The deck compares MSE plus gradient clipping against robust losses (F 53). Clipping bounds updates; it does not turn MSE into a robust statistical objective.
- Prefer sigmoid plus cross-entropy for the class classification examples. The slide assertion that sigmoid+MSE cannot be used is too absolute: the practical concern is optimization/saturation; deep networks with cross-entropy also remain generally nonconvex. (F 72-75, 191; clarification.)
- Probabilities versus logits and the loss's `from_logits` setting must agree when implementing a model (implementation check).

## Batch size, training duration, and gradient computation

| Control/concept | Class coverage | Source |
|---|---|---|
| Full-batch GD | Entire training set produces one averaged gradient/update. Memory and throughput can be limiting. | F 194, 197 |
| Single-instance SGD | One example per update; noisy updates and online learning. | F 200-202 |
| Mini-batch GD / commonly called SGD | Average loss/gradients over a batch; batch size controls update frequency, noise, throughput, and memory. Slide gives 50-250 as common examples, not mandatory limits. | F 205-208 |
| Epochs | Passes over data; fit examples use 100 as a maximum/example. Epochs interact with batch size and stopping. | F 174, 197-205; D 84-87 |
| Ordering/shuffling | Slides suggest random sampling without replacement. For time-dependent tasks, preserve the data dependencies required by the task. | F 200, 205 |
| Backpropagation | Chain-rule gradient computation using intermediate forward values; distinguish gradient computation from optimizer update. | F 22, 113; D 6-7 |
| Numerical derivatives | Forward/central differences; finite-step and rounding limitations; expensive repeated loss evaluations. | I 45-47, 50 |
| Analytical derivatives/autodiff | Chain rule and TensorFlow automatic differentiation; matrix/vector shapes and vectorized gradients matter. | I 48-49; F 168-171, 208-212 |

With `N` training observations and batch size `B`, approximately `ceil(N/B)` updates occur per epoch if the final partial batch is kept. This is a derived implementation relationship; larger batches imply fewer updates for the same epoch count. Compare both updates and runtime when tuning batch size.

## Optimizers and learning rate

| Optimizer/control | Mechanism and knobs | Lecture examples | Source |
|---|---|---|---|
| Plain SGD | Gradient step; learning rate `eta`; no momentum | Learning-rate sensitivity example ranges from 0.0001 to 0.7; the stable value is specific to its synthetic problem. | F 165-184, 194-208; D 14 |
| Momentum | Retains moving gradient/update history; tune learning rate and momentum | SGD learning rate 0.1, momentum 0.9 | D 12-14 |
| Nesterov | Momentum with a look-ahead correction | SGD learning rate 0.1, momentum 0.9, `nesterov=True` | D 16-17 |
| AdaGrad | Cumulative squared-gradient accumulator, coordinate-wise effective rates; can decay too far | Learning rate 0.001, initial accumulator 0.1, epsilon 1e-7 | D 26-30 |
| RMSprop | Exponential average of squared gradients; tune learning rate, rho, optional momentum, epsilon | Learning rate 0.001, rho 0.9, momentum 0, epsilon 1e-7; D 32 prose separately mentions 0.999 | D 32-33 |
| Adam | First and second raw moment estimates with bias correction | Learning rate 0.001, beta_1 0.9, beta_2 0.999, epsilon 1e-7 | D 35-37 |
| Exponential scheduling | Shared base rate decreases: `eta_t = eta_0 * r^(t/d)`; tune starting rate, decay factor and interval | Concept: eta_0=0.1, r=0.95, d=20 epochs. Code: initial rate 0.1, decay_steps=10000, decay_rate=0.95. | D 20-22 |

D 9 groups momentum/Nesterov as global constant-base-rate methods, schedules as global adaptive-rate methods, and AdaGrad/RMSprop/Adam as local adaptive-rate methods. A schedule can be combined with an optimizer; it is not a mutually exclusive optimizer choice.

Implementation clarifications to preserve:

- A too-small learning rate can make progress very slow; too-large rates can oscillate or diverge. Convexity alone does not guarantee convergence for arbitrary step sizes. Deep-network optimization does not guarantee a global minimum. (F 177-184, 190-202.)
- Lecture momentum uses `(1-gamma)` on the new gradient; library SGD velocity conventions need not have the same scaling. Do not assume the same numeric learning rate makes both formulas identical. (D 13-14.)
- Adam's first moment is mean-like, second raw moment is a mean of squares rather than variance. D 35 has conflicting gamma/value labels; standard beta_1=0.9 and beta_2=0.999 agree with D 37 code. Bias-correction time starts at 1, and persistent moment accumulators should remain uncorrected. (D 35-37.)
- TensorFlow's optimizer schedule counts optimizer steps; the conceptual lecture formula uses epochs. Convert units deliberately when batch size changes. (D 20-22.)

## Gradient stability and initialization

| Method/control | Choices and checks | Source |
|---|---|---|
| Input scale | Feature scale affects loss geometry and penalties. Fit scaling on training data. | D 12, 40-46, 90 |
| Normalized initialization | Uniform bounds `+/-1/sqrt(fan_in)`, biases zero; lecture motivation is reduced initial saturation. | D 46-49 |
| Glorot/Xavier | Uniform or normal; fan-in/fan-out average; tanh examples use `glorot_uniform`/`glorot_normal`. | D 50-53, 56-57 |
| Sigmoid-scaled Glorot | Lecture uses 4 times Glorot's standard deviation/uniform half-width; code uses VarianceScaling scale=16, mode=fan_avg. | D 53, 56-57 |
| He | Uniform/normal; lecture distinguishes incoming-only `he_normal`/`he_uniform` from fan-average VarianceScaling scale=2. | D 54-58 |
| Activation choice | Avoid excessive saturation; ReLU variants avoid positive-side saturation but do not guarantee stable gradients. | D 38-61 |
| Batch normalization | Try absent/present and before/after activation; includes input normalization in the example. Tune moving-statistic momentum and epsilon as needed. Gamma/beta are learned scale/shift. | D 63-75 |
| Gradient value clipping | Cap each component independently; threshold 1.0 is the lecture example and may alter gradient direction. | D 77 |
| Gradient norm clipping | Rescale a vector when its norm exceeds a threshold; example 1.0. Check per-tensor versus global clipping in the installed API. | D 78 |

Initialization equations from D 52-56: Glorot normal SD `sqrt(2/(fan_in+fan_out))`, uniform half-width `sqrt(6/(fan_in+fan_out))`; sigmoid lecture multiplies each by 4; He fan-average lecture multiplies each by `sqrt(2)`. These are SDs/half-widths, not variances. Incoming-only He uses variance `2/fan_in`, which differs when layer widths differ. The slide's fan-average He should not be silently equated with the built-in incoming-only initializer.

BN training uses batch means/variances and updates moving statistics; inference uses stored moving statistics without updating from test data. Lecture momentum is around 0.99. Stored variance is nontrainable but changes during training; gamma/beta are trainable. When BN immediately follows a linear Dense layer, removing the dense bias can avoid redundancy. (D 65-74.)

## Regularization, selection, and prediction

| Method | Knobs / behavior | Source |
|---|---|---|
| Early stopping | Monitored validation metric, maximum epochs, patience, restore-best behavior. Patience 10-20 is a lecture illustration. | D 82, 86-87 |
| Checkpoints | Save frequency, best-only selection, filepath; saving best alone does not stop training. Save/load structure, weights, optimizer settings/state. | D 83-86 |
| L1 / lasso | Absolute-weight penalty; tune coefficient; favors sparse weights. | D 89-91 |
| L2 / ridge | Squared-weight penalty; tune coefficient; stronger shrinkage for larger weights. | D 89-91 |
| Elastic net | Combined L1/L2; tune both coefficients or theoretical strength/mixing fraction. | D 89-91 |
| Dropout | Rate, placement, and adequate training budget; class example range 10%-50%. Increase for overfit, reduce for underfit; consider only later hidden layers. | D 93-97 |
| MC dropout | Multiple stochastic predictions; tune number of draws, average predictions, examine their spread. | D 98 |
| BN plus dropout | Test combination versus each alone; performance may worsen. | D 99 |
| Augmentation | Realistic, label-preserving transformations; on-the-fly generation; image/CNN emphasis. | D 101 |

Clarifications: lecture L2 uses `lambda/2 * sum(w^2)`; implementation coefficient conventions may differ. L1's exact zeros depend on the optimizer. Penalty values must not be confused with penalty derivatives (D 89-91). Keras-style inverted dropout scales retained activations by `1/(1-p)` during training, so manual inference scaling is unnecessary (D 96, 98). `training=True` can also put BN into training mode, so MC dropout with BN requires selective handling. Repeated dropout masks are not guaranteed unique. Noise augmentation is task-dependent; the slide's blanket claim about white noise should not become a universal restriction.

## Diagnosis, interpretation, and other covered concepts

- **Bias/variance:** repeated fits, bootstrap/out-of-bag predictions, average prediction, variability and error; complexity/sample-size tradeoff. High training and validation errors suggest insufficient fit or failed optimization; a large gap suggests generalization trouble. Use validation to make tuning decisions even though the introductory illustrations label held-out curves as test error. (I 15-29.)
- **PAC:** the finite-hypothesis-class bound relates training error, sample size, hypothesis count, and confidence. Its complexity term is a generalization bound, not a direct numerical measurement of statistical variance; continuous neural-network hypothesis spaces need additional theory. (I 27-29; clarification.)
- **Permutation importance:** compare predictive performance before and after permuting a feature; adjust the sign for higher-is-better versus lower-is-better metrics and restore the original feature between permutations. (I 38-39; implementation clarification.)
- **Partial dependence:** replace a feature with grid values, predict across observations, average predictions, and plot. The result is a predictive summary; correlated features can create unrealistic combinations and the curve is not automatically causal. (I 40; interpretation clarification.)
- **Incremental/online learning:** update on small portions of data, fixed-memory streaming for a fixed architecture/batch size; FIFO disk buffer, periodic retraining versus continuous updates, anytime models and computational footprint. Such designs do not guarantee high accuracy at all times. (I 30-34, 42.)
- **Foundations/framework:** perceptron, Hebbian/perceptron learning rules, linear separability, step/sign functions, MLPs, forward/backward passes; matrix multiplication and weight dimensions; tensors and dataflow; TensorFlow eager execution and Python API; Sequential, Functional, Subclassing API mentions, with Functional emphasized. (I 3-12, 44-50; F 4-22, 28.)

## Sequence, image, and named architectures: coverage limits

| Topic | What these decks actually establish | Source |
|---|---|---|
| RNN / sequence models | Named as suitable architecture families for text; recurrent gradient problems deferred to later coverage. | I 36; F 7; D 77 |
| Sequence-to-sequence | Named as a TensorFlow application for machine translation. | F 7 |
| Attention | Named as a sequence/text approach; no attention implementation or tuning lesson. | I 36 |
| CNNs | Named for image/video tasks, augmentation, and hierarchical image features. | I 36; D 101, 104 |
| Transformers / Primer | Squared ReLU example cites Primer; this is an activation example, not a Transformer architecture lesson. | F 159 |
| AlexNet / residual networks | Historical depth comparison; no detailed architecture or residual-block tuning. | D 103 |

No LSTM, GRU, recurrent dropout, sequence-window length, masking strategy, attention-head count, convolution-kernel search, or Transformer-specific search is taught in detail by these three decks. Those may be useful model-specific choices, but must be labeled additional material until the relevant class resources are supplied.

## Practical experiment order (proposed workflow, not a slide-prescribed search)

1. Establish the target, metric, clean split, preprocessing, correct output/loss, baseline model, and compute limit.
2. Check learning curves and numerical stability; compare learning rate, optimizer, and batch size before interpreting poor training as insufficient capacity.
3. Compare hidden activations with compatible initializers; vary depth and per-layer width on a bounded budget.
4. Add or adjust early stopping, penalties, dropout, and augmentation according to validation behavior; compare BN/clipping when relevant.
5. Refine promising combinations, repeat seeds when affordable, reload the best validation checkpoint, and evaluate the final test set once selection is complete.

Suggested trial record:

```text
trial_id, seed, data_split_id, preprocessing_id, hidden_units_per_layer,
hidden_activations, initializer, embedding_dimensions, batch_size,
optimizer, learning_rate, momentum, nesterov, rho, beta_1, beta_2, epsilon,
lr_schedule_and_units, batch_norm_placement, batch_norm_momentum,
clip_type, clip_threshold, l1, l2, dropout_rates_and_placement,
augmentation, output_activation, loss, huber_delta, monitored_metric,
max_epochs, patience, best_epoch, validation_score, train_score,
runtime, parameter_count, checkpoint_path, notes
```

Use only applicable fields. No concrete model, dataset, trial budget, or optimal hyperparameter settings have been established in this handoff.
