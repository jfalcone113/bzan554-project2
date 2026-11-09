# BZAN 554 class resource index

Scope: resources supplied here or explicitly referenced in `introduction (2).pdf`, `fundamentals.pdf`, and `deep.pdf`, all by Michel Ballings. This cannot establish every resource used elsewhere in the course. PDF page numbers equal slide numbers. External resources are indexed from the PDFs, including embedded hyperlinks; their contents and current availability were not checked for this compilation.

## Available primary materials

| File | Pages | Subject |
|---|---:|---|
| `introduction (2).pdf` | 50 | Introduction: Answers to Frequently Asked Questions; network notation, bias/variance, PAC, incremental learning, interpretation, derivatives, training from scratch |
| `fundamentals.pdf` | 212 | Fundamentals of Neural Networks and Getting Started with TensorFlow; response architectures/losses, activations, gradient descent |
| `deep.pdf` | 105 | Training Deep Neural Networks; embeddings, backpropagation, optimizers, gradient stability, regularization, depth/width |

## Available secondary and handoff materials

- `deep-slide-notes.md`, `deep-slide-notes.pdf`: pre-existing slide-by-slide notes for `deep.pdf`, including clarifications and additional documentation links. These are study aids, not separate evidence of what was assigned in class.
- `AGENTS.md`: concise instructions for a future model-tuning session.
- `MODEL_TUNING_REFERENCE.md`: organized inventory with page references and explicitly labeled clarifications.
- `CLASS_RESOURCES.md`: this index.
- `class-notes-text/introduction.txt`, `class-notes-text/fundamentals.txt`, `class-notes-text/deep.txt`: complete page-labeled machine-extracted slide text. Figures can be missing and equation/code spacing can be imperfect; retain the PDFs as authoritative visual sources.

## Referenced class files not present in this folder

These filenames are references, not files bundled or analyzed here. Preserve capitalization when searching your course downloads. The gradient-descent filename is displayed as `gradient descent.py` by PDF extraction; confirm its exact space/underscore spelling against your course files.

| Referenced resource | Purpose | Source |
|---|---|---|
| `NeuralNetFromScratch.py` | Network training from matrix multiplication, numerical derivatives, and learning rule | introduction p. 50 |
| `RegressionTF.py` | Single-response regression with TensorFlow | fundamentals p. 28 |
| `MultiOutputRegressionTF.py` | Multiple regression outputs | fundamentals p. 31 |
| `BinaryClassificationTF.py` | Binary sigmoid classifier | fundamentals p. 58 |
| `MultiLabelClassificationTF.py` | Independent-label classifier | fundamentals p. 78 |
| `PreprocessYforMultiClass.py` | Multi-class target preprocessing | fundamentals p. 87 |
| `MultiClassClassificationTF.py` | Softmax classifier | fundamentals p. 87 |
| `gradient descent.py` | Manual GD, synthetic regression, learning-rate sensitivity | fundamentals p. 174 |
| `embedding.py` | Categorical lookup embeddings | deep p. 4 |
| `backprop.pdf` | Fully worked backpropagation example | deep p. 7 |
| `dropout.py` | Complete dropout implementation example | deep p. 98 |

The bonus assignment mentioned at deep p. 87 implements early stopping for a manual epoch/instance loop; no assignment filename is supplied. Later recurrent-network and CNN coverage is mentioned, but no corresponding class files are identified in these decks.

## External linked resources

The following table is generated from the decks' external hyperlink annotations, deduplicated by URL within each deck. Page references include every occurrence. Shortened URLs are retained as supplied. This is a resource inventory, not an endorsement of current APIs or a claim that every link was separately assigned.

| Resource | Source / pages |
|---|---|
| [Derivative as a function (Ximera)](https://ximera.osu.edu/mooculus/calculus1/derivativeAsAFunction/digInTheDerivativeAsAFunction) | introduction pp. 45 |
| [TensorFlow API documentation](https://www.tensorflow.org/api_docs/) | fundamentals pp. 6 |
| [Dataflow programming](https://en.wikipedia.org/wiki/Dataflow_programming) | fundamentals pp. 8 |
| [TensorFlow graphs guide](https://www.tensorflow.org/guide/graphs) | fundamentals pp. 8 |
| [Keras documentation](https://keras.io/) | fundamentals pp. 28 |
| [TensorFlow Keras Functional API](https://www.tensorflow.org/guide/keras/functional) | fundamentals pp. 28 |
| [O’Reilly chapter linked for softmax](https://www.oreilly.com/library/view/neural-networks-and/9781492037354/ch01.html) | fundamentals pp. 90 |
| [Neural Networks and Deep Learning, chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) | fundamentals pp. 111 |
| [Searching for Activation Functions (Swish)](https://arxiv.org/abs/1710.05941) | fundamentals pp. 153 |
| [Primer: Searching for Efficient Transformers](https://arxiv.org/abs/2109.08668) | fundamentals pp. 159 |
| [Learning-rate explanation linked in class](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10) | fundamentals pp. 187 |
| [How optimization works](https://brohrer.github.io/how_optimization_works_1.html) | fundamentals pp. 190 |
| [Optimizer overview linked for momentum](https://arxiv.org/pdf/1609.04747) | deep pp. 13 |
| [SGD with momentum article](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d) | deep pp. 17 |
| [ExponentialDecay API](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay) | deep pp. 22 |
| [AdaGrad original paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) | deep pp. 30 |
| [Adam original paper](https://arxiv.org/pdf/1412.6980.pdf) | deep pp. 37 |
| [Glorot/Bengio initialization paper](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) | deep pp. 48, 49, 50 |
| [He rectifier paper (lecture short link)](https://bit.ly/2BQNZXy) | deep pp. 55 |
| [VarianceScaling API](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling) | deep pp. 57 |
| [Batch normalization original paper](https://arxiv.org/pdf/1502.03167.pdf) | deep pp. 66 |
| [Dropout original paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) | deep pp. 93 |
| [Batch normalization/dropout interaction paper](https://arxiv.org/pdf/1801.05134.pdf) | deep pp. 99 |

## Named research and historical sources

Some sources are mentioned by author/title without a separate link. Where a hyperlink exists, it is in the table above.

| Source named in slides | Lecture context | Location |
|---|---|---|
| F. Rosenblatt (1957), *The perceptron, a perceiving and recognizing automaton*, Cornell Aeronautical Laboratory | Perceptron and learning rule | fundamentals pp. 14, 17 |
| Hebb (1949); Lowel's summary of Hebbian learning | Historical motivation for strengthening connections | fundamentals p. 17 |
| Minsky and Papert (1969) | Perceptron limitations and linear separability | fundamentals p. 21 |
| Rumelhart, Hinton, and Williams (1986) | Backpropagation and differentiable activations | fundamentals pp. 22, 113 |
| Ramachandran et al. (2017), *Searching for Activation Functions* | Swish | fundamentals p. 153 |
| So et al. (2021), *Primer: Searching for Efficient Transformers* | Squared ReLU | fundamentals p. 159 |
| Duchi, Hazan, and Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* | AdaGrad | deep p. 30 |
| Kingma and Ba, *Adam: A Method for Stochastic Optimization* | Adam | deep p. 37 |
| Glorot and Bengio | Signal scaling / Xavier initialization paper | deep pp. 48-50 |
| Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* | He initialization | deep p. 55 |
| Ioffe and Szegedy (2015) | Batch normalization | deep pp. 63-66 |
| Srivastava, Hinton, et al. (2014) | Dropout | deep p. 93 |
| Gal and Ghahramani (2016) | Monte Carlo dropout, approximate Bayesian inference | deep p. 98; no hyperlink supplied on that page |
| Paper linked as arXiv 1801.05134 | BN/dropout interaction | deep p. 99 |

## Other reference strings and generated outputs

Deep p. 84 contains a ModelCheckpoint documentation URL inside code comments. The extraction inserts spaces into the code, so its readable reconstruction is `https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint`; it is not an external annotation in the table.

The fundamentals deck contains inline plotting code that saves demonstration figures (for example MSE, MAE, gradients, cross-entropy, and loss comparisons). These generated `.pdf` filenames are outputs of slide examples, not separately supplied class readings. Model filenames such as `my_model.keras` / `.h5` are likewise illustrative saved outputs, not class resources to retrieve.

The existing `deep-slide-notes.md` adds official Keras links for SGD, Adam, and batch normalization. Those links support secondary implementation clarifications; they are not presented here as additional resources used in the supplied lecture decks.

## Move this context to a new session

1. Extract `BZAN554-model-tuning-context.zip` into a folder, or copy the three Markdown handoff files and source PDFs directly.
2. For a Codex project, integrate `AGENTS.md` at the project root while preserving existing project instructions. For a chat upload, attach `AGENTS.md`, `MODEL_TUNING_REFERENCE.md`, and `CLASS_RESOURCES.md`; add PDFs/text for source review. Do not assume a new session can access this session's absolute paths or a ZIP without extraction.
3. Supply the model code, data/schema or representative sample, target definition, validation metric/split, current baseline results, constraints, and run budget.

Starter prompt:

> Read AGENTS.md and its companion references. Help tune my model using the BZAN 554 techniques documented here. Inspect the supplied code and data first. Use validation for selection and reserve the test set for final evaluation. Distinguish course-covered techniques, implementation corrections, and additional suggestions. Propose a bounded experiment plan, then implement/run the authorized work and keep a trial log. My target is [...], metric is [...], current result is [...], and compute budget is [...].
