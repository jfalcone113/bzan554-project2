# BZAN 554: Deep Learning for Business Applications
## Training Deep Neural Networks

**Michel Ballings**

---

## Outline

1. [Embeddings](#embeddings)
2. [Backpropagation](#backpropagation)
3. [Faster Optimizers](#faster-optimizers)
   - [Global, constant learning rate](#global-constant-learning-rate) — Momentum, Nesterov's Accelerated Gradient
   - [Global, adaptive learning rate](#global-adaptive-learning-rate) — Learning Rate Scheduling
   - [Local, adaptive learning rate](#local-per-parameter-adaptive-learning-rate) — AdaGrad, RMSprop, ADAM
4. [Vanishing/Exploding Gradient](#vanishingexploding-gradient)
   - [Problem statement](#problem-statement)
   - [Solution 1: Weight Initialization](#solution-1-weight-initialization)
   - [Solution 2: Nonsaturating Activation Functions](#solution-2-nonsaturating-activation-functions)
   - [Solution 3: Batch Normalization](#solution-3-batch-normalization)
   - [Solution 4: Gradient Clipping](#solution-4-gradient-clipping)
5. [Avoiding Overfitting with Regularization](#avoiding-overfitting-with-regularization)
   - [Early Stopping](#early-stopping)
   - [L-1 and L-2 Regularization](#l-1-and-l-2-regularization)
   - [Dropout](#dropout)
   - [Data Augmentation](#data-augmentation)
6. [Depth vs Width](#depth-vs-width)

---

## Embeddings

- An embedding is simply a dense layer with a linear activation function, but instead of using matrix multiplication requiring us to feed in a dummy vector, an embedding applies a fast **lookup** allowing us to feed in the integer corresponding to the category.
- Why do we call it an embedding and not a dense linear layer? The term "embedding" comes from mathematics and representation learning, and "dense layer" comes from engineering.

### One-hot encoding example

Consider a categorical variable with 20 unique categories. Only one category can be 1, all other categories are 0. Consider the first hidden layer to have 2 units.

```python
# Standard way: feed in dummy vector with 20 columns:
inputs = tf.keras.layers.Input(shape=(20,))
hidden = tf.keras.layers.Dense(2)(inputs)
# ...

# Embedding: feed in 1 integer indicating category that is 1
inputs = tf.keras.layers.Input(shape=(1,))
embedding = tf.keras.layers.Embedding(input_dim=20,
                                      output_dim=2)(inputs)
# ...
```

### K-hot encoding example

Consider a categorical variable with 20 unique categories. K (say 4) categories can be 1, all other categories are 0. Consider the first hidden layer to have 2 units.

```python
k = 4
# Standard way: feed in dummy vector with 20 columns:
inputs = tf.keras.layers.Input(shape=(20,))
hidden = tf.keras.layers.Dense(2)(inputs)
# ...

# Embedding: feed in k integers indicating categories that are 1
inputs = tf.keras.layers.Input(shape=(k,))
embedding = tf.keras.layers.Embedding(input_dim=20,
                                      output_dim=2)(inputs)
# ...
```

**Notes:**

- Embeddings cannot be used if the equivalent linear dense layer takes in values other than 0 or 1 (e.g., counts of words in text analysis).
- Situations with a dynamic $k$ can be solved for with padding.
- For embeddings: make sure to integer-encode the input with consecutive integers. The `input_dim` is then $\max(\text{categorical variable}) + 1$ (to account for zero-based counting).
- See **embedding.py**

---

## Backpropagation

**Process overview:**

| Step | Description |
|------|-------------|
| 1 | Random initialization of weights/model |
| 2 | Feed forward (inputs → actual outputs); the calculation graph is stacked automatically |
| 3 | Calculate loss function (uses desired outputs + loss function) → loss (error) metric at this step |
| 4 | Calculate the derivative of error → gradient for the last layer |
| 5 | Backpropagate (unstack the calculation graph) → gradients for all layers |
| 6 | Update the weights (governed by update frequency and optimizer function: delta rule / adadelta / …) |
| 7 | Iterate until convergence |

- Backpropagation is very important for training deep networks as it enables us to **recycle deeper calculations in the shallower calculations**. This increases efficiency substantially, thereby making deep learning possible.
- See **backprop.pdf** for a fully worked example.

---

## Faster Optimizers

There are other variations of Gradient Descent incorporating ideas to allow faster convergence to the optimum. The most popular algorithms are:

- **A global (same for all parameters) and constant (across learning updates) learning rate**
  - SGD with Momentum
  - Nesterov's Accelerated Gradient (NAG)
- **A global (same for all parameters) and adaptive (across learning updates) learning rate**
  - Learning rate scheduling
- **A local (per-parameter) adaptive (across learning updates) learning rate**
  - Adaptive gradient (AdaGrad)
  - RMSprop
  - Adam

---

### Global, constant learning rate

#### Momentum

SGD has trouble navigating plateaus. A plateau is common when variables have different scales (for example the elongated bowl happens because the feature representing the horizontal axis has much larger values than the one representing the y-axis).

*(Figure: (a) SGD without momentum — heavy zig-zag across the elongated contours; (b) SGD with momentum — far smoother traversal.)*

Picture a cost function that goes down steeply, then almost plateaus with a near-horizontal downward slope and then goes down steeply again. On the steep regions SGD will go fast, and on the horizontal region SGD will go very slowly. When momentum is added the plateau is traversed much faster.

This technique (<https://arxiv.org/pdf/1609.04747>) modifies the standard SGD by introducing velocity $\nu$, and momentum $\gamma$ which tries to control the velocity and prevents overshooting valleys while allowing faster descent as well as giving Gradient Descent a short-term memory.

$$\nu_t = \gamma \nu_{t-1} + (1-\gamma)\nabla_\theta J(\theta)$$

$$\theta = \theta - \eta \nu_t \tag{1}$$

The momentum term $\gamma$ is usually set to 0.9. Summarizing, the momentum algorithm replaces the gradient with a **weighted moving average of the gradient** and uses that to update the weights. It therefore memorizes how fast we were descending before and uses that speed to traverse plateaus. A way to think about this is a ball that rolls down a curve. If it hits a perfectly horizontal plateau it will continue rolling for a while until it has lost all its velocity. In plain vanilla SGD it would have stopped immediately when hitting the plateau.

**TensorFlow implementation:**

```python
tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
```

#### Nesterov's Accelerated Gradient

- We would like our ball to look ahead and slow down if the loss is going to go up. Nesterov's Accelerated Gradient is a small variation of momentum.
- Instead of calculating the gradient at the current position $\theta$, we calculate the gradient at the **approximate position that we will be in**. Which information do we have (no additional computations) that we can add to $\theta$ before we compute the gradient? We know for sure that we are going to subtract $\eta\gamma\nu_{t-1}$ from $\theta$!
- From a physical perspective, it makes sense to make judgments about our position based on the position that we know that we are going to be in a short while.

$$\nu_t = \gamma\nu_{t-1} + (1-\gamma)\nabla_\theta J(\theta - \eta\gamma\nu_{t-1})$$

$$\theta = \theta - \eta\nu_t \tag{2}$$

*(Figure: Momentum update vs. Nesterov momentum update — in the Nesterov version the "lookahead" gradient step is taken from the end of the momentum step, giving an actual step slightly different from the original.)*

- This strategy works because in general the momentum vector will be pointing towards the optimum, so it will be slightly more accurate to compute the gradient at a point closer to the optimum than at the original position.
- Nesterov's Accelerated Gradient will almost always speed up training compared to regular Momentum optimization.

```python
tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,
                        nesterov=True)
```

Reference: `towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d`

---

### Global, adaptive learning rate

#### Learning Rate Scheduling

- For Gradient Descent (with or without momentum and Nesterov) and especially Stochastic and Mini-batch Gradient Descent it is recommended that we add a learning schedule.
- Most popular: **exponential scheduling**: $\eta = \eta_0 r^{t/d}$ where $r$ is decay rate, $t$ is the epoch, $d$ the decay steps, and $\eta_0$ the initial learning rate.
- Suppose that we want to multiply the learning rate by a factor $r = 0.95$ every 20 epochs, and start with an initial learning rate of 0.1:

```python
lr0 = 0.1
r = 0.95
import numpy as np
t = np.arange(1, 1000, 1)
lr = lr0 * (r ** (t / 20))
```

*(Figure: Exponential Learning Rate Decay — learning rate falls smoothly from 0.10 at epoch 0 to roughly 0.008 by epoch 1000.)*

**Implementing a learning rate schedule with TensorFlow:**

```python
# initial_learning_rate * decay_rate ^ (current_step / decay_steps)
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.95

learning_sched = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate)

tf.keras.optimizers.SGD(learning_rate=learning_sched)
```

Docs: <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay>

---

### Local (per-parameter) adaptive learning rate

- Until now we have used a global and equal learning rate for all our parameters (momentum and Nesterov) and a global and adaptive learning rate (learning schedule).
- But what if we could speed up or slow down learning by increasing or decreasing this factor, **even for each parameter**, as the training progresses?
- We could adaptively tune learning and know into which direction to accelerate and decelerate. Several methods that use such adaptive learning rates have been proposed, most notably AdaGrad, RMSprop and ADAM.

#### Adaptive Gradient (AdaGrad)

- GD is guaranteed to reach the minimum loss given convexity as long as we reduce the learning rate as epochs roll on.
- For infrequently occurring features (i.e., they are zero a lot) we may get into a situation where we have reduced the learning rate too quickly. We can't just slow down the reduction in learning rate, because the frequently occurring features will then be jumping around their minima.
- We could keep a counter of how many times a feature was non-zero and divide the learning rate by this number. The learning rate for infrequently occurring features will therefore be higher than the learning rate for frequently occurring features.
- This fails whenever we do not quite have sparsity but rather just data where the gradients are often very small and only rarely large. After all, it is unclear where one would draw the line between something that qualifies as an observed feature or not (if features are small values; nearly zero; the updates will be small too).
- The solution is to take into account the **sum of the historical gradients** when we divide the learning rate.

Let's revisit the elongated bowl problem. GD quickly moves down the steepest slope, and then slowly moves down to the bottom. AdaGrad scales down the gradient vector along the steepest dimensions to point more toward the global minimum.

*(Figure: elongated cost contours with Theta 1 as the flatter dimension and Theta 2 as the steeper dimension; the AdaGrad path heads toward the minimum more directly than the Gradient Descent path.)*

$$g_t = g_{t-1} + \left(\nabla_{\theta_t} J(\theta_t)\right)^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\text{diag}(g_t) + \epsilon}}\nabla_{\theta_t} J(\theta_t) \tag{3}$$

- AdaGrad adapts the learning rate to the parameters by dividing the learning rate by the square root of the sum of squares of historic gradients. In the denominator:
  - the squares are taken to abstract away the sign of the gradients, and the root is taken to undo the effect of squaring, and
  - the sum of the previous gradients is taken so that smaller updates (smaller learning rates) will be used for parameters associated with frequently occurring features, and vice versa for parameters associated with infrequently occurring features.
- $g_t$ is a column vector of length equal to the number of parameters, where an element is the sum of squares of the historic gradients up to iteration $t$ for a parameter.
- $\text{diag}(g_t) \in \mathbb{R}^{d \times d}$ is a diagonal matrix (to matrix multiply with the gradient vector) where each diagonal element is the sum of the squares of the gradients w.r.t. $\theta_i$ up to iteration $t$, and $\epsilon$ (say $10^{-8}$) is a smoothing term to avoid division by zero.

**Summary:**

- The learning rate is updated automatically at each time step $t$
- The learning rate is different per parameter

**Pros:**

- Speeds up training
- Less need to manually tune learning rate

**Cons:**

- For each parameter a sum is computed of the squared historic gradients and stored in $g_t$. Since each element is positive, the sum keeps growing during training, eventually shrinking the learning rate to an infinitesimally small number, and thereby making weight updates impossible.

**TensorFlow implementation.** AdaGrad often performs well for simple problems but unfortunately it often stops too early when training neural networks. The learning rate gets scaled down so much that the algorithm ends up stopping entirely before reaching the global optimum.

```python
tf.keras.optimizers.Adagrad(
    learning_rate=0.001, initial_accumulator_value=0.1,
    epsilon=1e-07)
```

**Original paper:** <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>

#### RMSprop: Root Mean Square Prop

- A disadvantage of AdaGrad is that the $g_t$ part of the update is monotonically increasing. This can pose problems because the learning rate can steadily decrease to the point where it stops the learning altogether.
- RMSprop solves this problem by dividing the learning rate by a **running average** of squared gradients. For example, if $\gamma = 0.9$ then the first gradient will be $0.9 g_1$ in iteration 1, $0.9(0.9 g_1)$ in iteration 2, $0.9(0.9(0.9 g_1))$ in iteration 3, …
- Here $\gamma$ is called the decay rate with a typical value of 0.999.

$$g_t = \gamma g_{t-1} + (1-\gamma)\left(\nabla_{\theta_t} J(\theta_t)\right)^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\text{diag}(g_t) + \epsilon}}\nabla_{\theta_t} J(\theta_t) \tag{4}$$

Except on very simple problems, this optimizer almost always performs much better than AdaGrad. In fact, it was the preferred optimization algorithm of many researchers until Adam optimization came around.

> **Note:** `rho` below is $\gamma$ in Equation 4.

```python
# rho is discounting factor
tf.keras.optimizers.RMSprop(
    learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07)
```

Next, and finally, let's look at ADAM, which is a combination of momentum and RMSprop.

#### Adaptive Moment Estimation (ADAM)

- Adam is the state-of-the-art first order optimization algorithm. It is an improvement upon RMSprop by adding momentum to the update rule, combining the best of both momentum and adaptive learning.
- ADAM stores the running average of past squared gradients $g_t^2$ like RMSprop, and the running average of past gradients $v_t$ like momentum:

$$g_t = \gamma_1 g_{t-1} + (1-\gamma_1)\left(\nabla_{\theta_t} J(\theta_t)\right)^2$$

$$v_t = \gamma_2 v_{t-1} + (1-\gamma_2)\nabla_{\theta_t} J(\theta_t) \tag{5}$$

The moving average $v_t$ is the first raw (around 0) moment (the mean) of the gradients and the moving average $g_t$ is the second raw moment (the variance) of the gradients. The hyper-parameters $\gamma_1$ (default: 0.9) and $\gamma_2$ (default: 0.999) are decay rates controlling the weighted averages. $t$ represents the iteration number starting at 1.

- $g_t$ and $v_t$ are initialized as 0-vectors ($g_1 = 0$ and $v_t = 0$). For this reason, they are biased towards 0 in the beginning of training. We correct for this with the following two equations.
- Compute bias-corrected first and second moment estimates:

$$g_t = \frac{g_t}{1 - \gamma_1^{\,t}} \tag{6}$$

$$v_t = \frac{v_t}{1 - \gamma_2^{\,t}} \tag{7}$$

In the beginning of training, these corrections will increase $g_t$ and $v_t$ (i.e., move away from 0). As $t$ increases the denominator will approach 1 and there will be no more correction. In other words, correction only happens in the beginning of training.

- Update rule:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\text{diag}(g_t) + \epsilon}} v_t$$

**Pseudo code in Python:**

```python
for t in range(num_iterations):
    d = compute_the_gradient(x, y)
    v = gamma_1 * v + (1 - gamma_1) * d                 # Momentum
    g = gamma_2 * g + (1 - gamma_2) * np.power(d, 2)    # RMSprop
    v = v / (1 - np.power(gamma_1, t))                  # Bias Correction
    g = g / (1 - np.power(gamma_2, t))                  # Bias Correction
    w = w - step_size * v / np.sqrt(g + epsilon)
```

**In TensorFlow:**

```python
tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
```

**Original paper:** <https://arxiv.org/pdf/1412.6980.pdf>

---

## Vanishing/Exploding Gradient

### Problem statement

**Preliminaries:**

- Assume that we standardized our inputs. This means that they have mean $\mu = 0$, standard deviation $\sigma = 1$, and variance $\sigma^2 = 1^2$.
- Assume that we initialized our weights $w$ from a standard normal distribution (i.e., $\mu = 0$, $\sigma = 1$, $\sigma^2 = 1^2$).
- Remember that the **Variance Product Law** in this special case states that for a given $w_i$ and $X_i$ the variance of their product is the product of their variances: $\text{Var}(X_i w_i) = 1 \ast 1 = 1$ when $w$ and $X$ are uncorrelated. If they are correlated we need to add something, but this is immaterial for our discussion.
- Remember that the **Variance Sum Law** states that the variance $\sigma^2$ of the sum equals the sum of the variances. For example, the variance of $z_{h_1} = \text{Var}\left(\sum_{i=1}^{m} X_i w_i\right) = \sum_{i=1}^{m}\text{Var}(X_i w_i)$ for $m$ inputs, and $w$ and $X$ column vectors, when the $w_i^T X_i$ are uncorrelated. If they are correlated we need to add the covariance between the $w_i^T X_i$. The latter is immaterial for our discussion.

Taking the preliminaries into account:

- If there are 200 inputs to a neuron, then $\sigma^2 = 200$ and $\sigma = \sqrt{200} = 14.14$.
- If we have a sigmoid or tanh activation function, and the unactivated weighted sum ($z$) has a value at +1 standard deviation (14.14) from the mean value then the activated value is very close to 1, say 0.9999993.
- Remember that when we backpropagate, to compute the partial derivative we use the chain rule. Imagine a one hidden layer network with two hidden nodes, and one output node. To compute the partial derivative of a weight that is positioned to the left of the hidden layer, one needs to compute the partial derivative of the activated value w.r.t. the weighted sum (highlighted):

$$\frac{\partial E}{\partial w_1} = \frac{\partial E}{\partial o}\,\frac{\partial o}{\partial h_1}\,\boxed{\frac{\partial h_1}{\partial z_{h_1}}}\,\frac{\partial z_{h_1}}{\partial w_1} \tag{8}$$

**When using the sigmoid activation function:**

$$\frac{\partial h_1}{\partial z_{h_1}} = \text{sigmoid}(z_{h_1})\left(1 - \text{sigmoid}(z_{h_1})\right) \tag{9}$$

So in our example with $z_{h_1} = 14.14$ (or $z_{h_1} = -14.14$) we have $\frac{\partial h_1}{\partial z_{h_1}} = 7.228953 \times 10^{-7}$.

**When using the tanh activation function:**

$$\frac{\partial h_1}{\partial z_{h_1}} = \left(1 - (\tanh(x))^2\right) \tag{10}$$

So in our example with $z_{h_1} = 14.14$ (or $z_{h_1} = -14.14$) we have $\frac{\partial h_1}{\partial z_{h_1}} = 2.090328 \times 10^{-12}$.

To put things in perspective: if we are using the sigmoid function, and $z = 2$, then $h$ will be approximately $1/10$. If $z = 10$ (a mere fivefold increase), then $h$ will be approximately $1/22000$. That means that in the latter case we are learning 2200 times slower.

**What are the implications for deep networks?**

- **Forward pass:** If we assume a funnel shaped network (subsequent layers each have less neurons) then more shallow layers will saturate sooner.
- **Backward pass:** weights going into more shallow layers receive increasingly small updates with every layer as we keep on having to multiply by $\frac{\partial h}{\partial z_h}$.

This is what is called the **vanishing gradient problem**. It vanishes as we do our backward pass towards more shallow layers, and the reason is that neurons in more shallow layers have increasingly saturated outputs (where the slope is close to zero). The reason why they have increasingly saturated outputs is because more elements need to be added, each having a variance, and the variance of the sum is the sum of the variances. More variance means that we are working in the limits of the tanh and sigmoid functions.

---

### Solution 1: Weight Initialization

- The goal is thus to make sure that the weighted sum remains inside a feasible range of, say, $[-4, 4]$ so that the neurons do not start out learning in a saturated state. A natural solution is therefore to change how weights are initialized, by taking into account the number of incoming weights of a layer.
- For example, a popular weight initialization, called the **normalized initialization**, is $U\!\left(\frac{-1}{\sqrt{m}}, \frac{1}{\sqrt{m}}\right)$, with $U[a,b]$ the Uniform distribution in the interval $(a,b)$, and $m$ the size of the previous layer (i.e., number of incoming weights). Biases are set to 0. We have the sqrt in the denominator because — as the derivation of the variance below shows — this cancels out $m$ from the variance. Using the normalized initialization, the uniform distribution has:
  - mean $= \frac{1}{2}(a+b) = 0$
  - variance $= m\,\text{Var}[W] = m\frac{1}{12}(b-a)^2 = m\frac{1}{12}\left(\frac{1}{\sqrt{m}} - \frac{-1}{\sqrt{m}}\right)^2 = m\frac{1}{12}\left(\frac{2}{\sqrt{m}}\right)^2 = m\frac{1}{12}\left(\frac{4}{m}\right) = \frac{1}{3}$
- In other words: neurons in a given layer are initialized from a uniform distribution that is proportional to how many weights are going to the layer. One is now much less likely to venture outside the $[-4,4]$ range, and saturate the neuron at the start of learning, as for 200 inputs $\sigma = \sqrt{1/3}$ as opposed to $\sqrt{200}$ if we would use the normal distribution.

*(Figure: a sigmoid curve plotted over $[-10, 10]$. Low variance of $z$ means we operate in the central, near-linear region; high variance of $z$ means we operate across the full range, where the extremes are included — so there is a higher chance of saturation.)*

While this has worked in many cases, the problem still persists in many others. For example, using $U\!\left(\frac{-1}{\sqrt{m}}, \frac{1}{\sqrt{m}}\right)$: 98 percentiles (markers alone) and standard deviation (solid lines with markers) of the distribution of the activation values for the hyperbolic tangent networks in the course of learning show that **more shallow layers saturate sooner**. Likewise, the activation-value histogram at the end of learning (averaged across units of the same layer and across 300 test examples, tanh activation) shows important saturation of the lower layers: **more shallow layers saturate more** (because they saturate sooner).

Source: <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>

#### Glorot and Bengio's criteria

- A breakthrough in solving this issue was presented by Glorot and Bengio.
- They posited that, to make sure the signal does not die out or explode, we need the variance of the outputs of each layer to be equal to the variance of its inputs, and we also need the gradients to have equal variance before and after flowing through a layer in the reverse direction. Formally, with $i$ denoting a layer:
  - From a **forward pass** perspective, the information would flow well if for any $h$:

$$\forall i,\ \text{Var}(h^i) = \text{Var}(h^{i+1}) \tag{11}$$

  - From a **backward pass** perspective, the information would flow well if for any $z$:

$$\forall i,\ \text{Var}\!\left(\frac{\partial E}{\partial z^i}\right) = \text{Var}\!\left(\frac{\partial E}{\partial z^{i+1}}\right) \tag{12}$$

These conditions can be rewritten as:

$$\forall i,\ m_i \,\text{Var}(W^i) = 1 \tag{13}$$

$$\forall i,\ m_{i+1}\,\text{Var}(W^i) = 1 \tag{14}$$

meaning that the variance of the sum of the incoming weights of a given layer and the variance of the sum of the incoming weights for the next layer (i.e., the outgoing weights of the previous layer) equal 1. A compromise between these constraints is the average:

$$\forall i,\ \text{Var}(W^i) = \sigma^2 = \frac{2}{m_i + m_{i+1}} = \frac{2}{m_{in} + m_{out}}, \qquad \sigma = \sqrt{\frac{2}{m_{in} + m_{out}}} \tag{15}$$

#### Xavier (Glorot) Initialization

So, the idea is to randomly initialize weights from a Gaussian distribution with mean 0 and standard deviation:

$$\sigma = \sqrt{\frac{2}{m_{in} + m_{out}}}$$

or from a Uniform distribution between $-r$ and $r$ with:

$$r = \sqrt{\frac{6}{m_{in} + m_{out}}}$$

If $m_{in} = m_{out}$ then we can just do $\sqrt{1/m_{in}}$ and $\sqrt{3/m_{in}}$. The 6 in the numerator of the last equation comes from the fact that the variance of the uniform distribution is $1/3$, and since we want to keep the variance equal to 1 on average for both incoming and outgoing weights (i.e., 2 divided by the sum of the number of incoming and outgoing weights). These two equations work for the tanh because the value of the derivative of the tanh at the central location is 1.

For the **sigmoid** function we need a correction factor. The reason is that the derivative at the central location is $1/4$:

$$\sigma = 4\sqrt{\frac{2}{m_{in} + m_{out}}}, \qquad r = 4\sqrt{\frac{6}{m_{in} + m_{out}}}$$

#### Kaiming He Initialization

For ReLU we need yet another correction factor, named after Kaiming He:

$$\sigma = \sqrt{2}\sqrt{\frac{2}{m_{in} + m_{out}}}, \qquad r = \sqrt{2}\sqrt{\frac{6}{m_{in} + m_{out}}}$$

We need to divide $m_{in}$ and $m_{out}$ by 2 inside the square root because ReLU reduces the variance (because it clips everything below 0 to 0). Variance closer to 0 leads to activations closer to 0 and therefore we want to penalize less the number of weights. Since a ReLU is 0 for half of its input, we need to double the size of weight variance to keep the signal's variance constant.

**Original paper:** *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun — <https://bit.ly/2BQNZXy>

#### Summary of Weight Initialization

| Activation Function | Uniform Distrib. $[-r, +r]$ | Normal Distrib. |
|---|---|---|
| Tanh (Glorot) | $r = \sqrt{\dfrac{6}{m_{in}+m_{out}}}$ | $\sigma = \sqrt{\dfrac{2}{m_{in}+m_{out}}}$ |
| Sigmoid (Glorot) | $r = 4\sqrt{\dfrac{6}{m_{in}+m_{out}}}$ | $\sigma = 4\sqrt{\dfrac{2}{m_{in}+m_{out}}}$ |
| ReLU (and variants) (He) | $r = \sqrt{2}\sqrt{\dfrac{6}{m_{in}+m_{out}}}$ | $\sigma = \sqrt{2}\sqrt{\dfrac{2}{m_{in}+m_{out}}}$ |

#### TensorFlow Implementation

```python
import tensorflow as tf

# Tanh
###############################
# default is glorot_uniform with uniform distribution
# uniform
keras.layers.Dense(2, activation='tanh', kernel_initializer='glorot_uniform')

# normal
keras.layers.Dense(2, activation='tanh', kernel_initializer='glorot_normal')

# Sigmoid
###############################
# uniform
glorot_sigm_init = keras.initializers.VarianceScaling(
    scale=16., mode='fan_avg', distribution='uniform')
keras.layers.Dense(2, activation='sigmoid', kernel_initializer=glorot_sigm_init)

# normal
glorot_sigm_init = keras.initializers.VarianceScaling(
    scale=16., mode='fan_avg', distribution='untruncated_normal')
keras.layers.Dense(2, activation='sigmoid', kernel_initializer=glorot_sigm_init)
```

Docs: <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling>

```python
# ReLU and Variants
###############################
# He (based on incoming weights only)
# normal
keras.layers.Dense(2, activation='relu', kernel_initializer='he_normal')

# uniform
keras.layers.Dense(2, activation='relu', kernel_initializer='he_uniform')

# He (based on average of incoming and outgoing)
# uniform
he_avg_init = keras.initializers.VarianceScaling(
    scale=2., mode='fan_avg', distribution='uniform')
keras.layers.Dense(2, activation='relu', kernel_initializer=he_avg_init)

# normal
he_avg_init = keras.initializers.VarianceScaling(
    scale=2., mode='fan_avg', distribution='untruncated_normal')
keras.layers.Dense(2, activation='relu', kernel_initializer=he_avg_init)
```

#### How well does it work?

*(Figure 7: Back-propagated gradient histograms with hyperbolic tangent activation. Top panel — $U\!\left(\frac{-1}{\sqrt m}, \frac{1}{\sqrt m}\right)$: more shallow layers have smaller gradients; the 0-peak decreases for higher layers. Bottom panel — $U\!\left(-\sqrt{\frac{6}{m_{in}+m_{out}}}, \sqrt{\frac{6}{m_{in}+m_{out}}}\right)$: the distributions for all five layers overlap closely.)*

---

### Solution 2: Nonsaturating Activation Functions

- Vanishing/Exploding Gradients problems are in part due to a poor choice of activation function.
- We went through all the activation functions in a previous class and we already know that the sigmoid function and hyperbolic tangent function are saturating quickly when the input becomes large. The function saturates at 0 or 1 for sigmoid, and -1 and 1 for tanh, with derivative 0 or extremely close to 0.
- However, other activation functions — the ReLU function and its variants — behave much better in deep neural networks. They do not saturate for positive values and also, they are quite fast to compute. In our experience: **ELU > PReLU > ReLU > tanh > sigmoid**.

---

### Solution 3: Batch Normalization

- He initialization with ELU (or variants of ReLU) can help us avoid problems at the **beginning** of learning. Batch normalization helps us stay out of trouble **during** learning.
- Batch Normalization (BN), introduced in 2015 by Ioffe and Szegedy to address the vanishing/exploding gradient problem, is a widely used technique for normalizing input.
- BN accommodates the **Internal Covariate Shift**: the distribution of each layer's input changes during training as the parameters of previous layers change. **This is a problem because a change in the distributions of layers' inputs requires the layers to continuously adapt to the new distribution.**
- BN involves adding a normalization operation right before or after the activation function of each hidden layer. This operation subtracts from each input variable the mini-batch mean $\mu_B$ of the input variable and divides by the mini-batch standard deviation $\sigma_B$ of the input, where $B$ denotes the mini-batch.

**But batch normalization does more than that:**

- Simply zero-centering and scaling by the standard deviation would change what the layer can represent. For example, if we normalize the inputs of the sigmoid, then we would always be working in the linear domain of the sigmoid (remember that the sigmoid is only non-linear at very low or very high values) and therefore the neural network model would not be able to represent non-linear functions.
- Ioffe and Szegedy's solution is to ensure that **'the transformation inserted in the network can represent the identity transform'**. In other words, the neural network needs to be given the chance to undo the normalization transformation entirely if it determines that this is the optimal thing to do. Most often this is not the optimal thing to do. What is often optimal is to change the transformation slightly.
- The latter is accomplished by applying, to the normalized inputs $\hat{x}$, a subsequent operation $\gamma\hat{x} + \beta$, where $\gamma$ and $\beta$ are a sort of standard deviation and mean (also called the **scale** and **shift** parameter), but these are learned by the model itself with backpropagation.

**Details:**

- The normalization is applied to each node $k$ independently. The usual output of the node is denoted by $x^{(k)}$. If we have a mini-batch of size $m$ instances then we have $m$ such outputs for a node $k$: $\{x_1^{(k)}, x_2^{(k)}, \ldots, x_m^{(k)}\}$. The Batch Normalized output is then denoted by $\hat{x}^{(k)}$: $\{\hat{x}_1^{(k)}, \hat{x}_2^{(k)}, \ldots, \hat{x}_m^{(k)}\}$.
- The $\gamma$ and $\beta$ are different per node $k$: $\gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}$, and are learned with the other usual weights and biases using backpropagation.
- By setting $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$ and $\beta^{(k)} = E[x^{(k)}]$ we could recover the original activations (if that were the optimal thing to do).
- If we were to be operating in a batch setting (each training step uses the whole training set) then we would use the whole set to normalize the activations. However, this is impractical when using incremental learning; therefore we use mini-batches to produce estimates of the mean and variance of each activation.
- The distribution of any $x^{(k)}$ has the expected value of 0 and variance of 1 as long as the elements of each mini-batch are sampled from the same distribution.
- Batch normalization happens before or after every layer. A scale and shift parameter needs to be learned for every node of every layer that has batch normalization.
- The whole process introduces some noise into the network, and that has been shown to **regularize** the model to some degree. However, regularization is a side effect of BN, rather than the main objective.
- **Summary:** $\gamma$ and $\beta$ are learned with backpropagation, and $\mu_B$ and $\sigma_B^2$ are computed on the mini-batch $B$ and used to normalize the mini-batch inputs.
- **Original paper:** <https://arxiv.org/pdf/1502.03167.pdf>

#### Batch Normalization — Algorithm

**Input:** Values of $x$ over a mini-batch: $B = \{x^1, x^2, \ldots, x^i, \ldots, x^m\}$, where $m$ denotes the number of instances in the mini-batch.

**Parameters to be learned with backprop:**

- $\gamma$: output scale vector containing one parameter per input
- $\beta$: output shift vector containing one parameter per input

**Output (for a given instance $x^i$):** $y^{(i)} = BN_{\gamma,\beta}(x^{(i)})$

$$\mu_B \leftarrow \frac{1}{m}\sum_{i=1}^{m} x^{(i)} \qquad (\textit{mini-batch mean})$$

$$\sigma_B^2 \leftarrow \frac{1}{m}\sum_{i=1}^{m}\left(x^{(i)} - \mu_B\right)^2 \qquad (\textit{mini-batch variance})$$

$$\hat{x}^{(i)} \leftarrow \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \qquad (\textit{normalize}) \tag{16}$$

$$y^{(i)} \leftarrow \gamma\hat{x}^{(i)} + \beta \equiv BN_{\gamma,\beta}(x^{(i)}) \qquad (\textit{scale and shift}) \tag{17}$$

**Notes:**

- $\epsilon$ is a tiny number to avoid division by zero (called smoothing term).
- $\mu_B$ is a vector of the mini-batch means, one for each input.
- $\sigma_B$ is a vector of the mini-batch standard deviations, one for each input.
- $\hat{x}^{(i)}$ is a vector of zero-centered and normalized inputs for instance $i$.
- $y^{(i)}$ is the output of the BN operation for instance $i$.

#### Batch Normalization — Making Predictions

- We now know that Batch Normalization calculates the mean and variance for each mini-batch at training time. But what to do at testing (prediction) time? It would not be proper if we are to use mean and variance of the testing mini-batch, because it may be skewed. And what do we do if we predict for a single instance?
- What we do is calculate the **population average** of the mean and variances across all the training mini-batches, and at testing time we use these two values in the normalization. This provides a more accurate value of the mean and variance.
- To calculate the population averages of the mini-batch mean and variances we keep a **moving average** so that we do not need to store all the mini-batch means and variances.

The mini-batch mean vector $\mu_B$ (one $\mu_B$ for every input) and variance vector $\sigma_B^2$ (one $\sigma_B^2$ for each input) are computed on every mini-batch and used in the exponentially weighted 'moving average':

$$\mu_{mov} = \alpha\mu_{mov} + (1-\alpha)\mu_B$$

$$\sigma_{mov}^2 = \alpha\sigma_{mov}^2 + (1-\alpha)\sigma_B^2 \tag{18}$$

Here $\alpha$ is the 'momentum' given to the previous moving statistic, around 0.99.

> **Note:** In TensorFlow, the moving mean is initialized from zero and moving variance is initialized from one.

Then, at testing time, using the population mean and variance, we normalize a given instance $x^i$ as follows:

$$\hat{x}^{(i)} = \frac{x^{(i)} - \mu_{mov}}{\sqrt{\sigma_{mov}^2 + \epsilon}}$$

$$y^{(i)} = \gamma\hat{x}^{(i)} + \beta \tag{19}$$

where

- $\epsilon$ is a tiny number to avoid division by zero (called smoothing term).
- $\mu_{mov}$ is a vector of the population means, one for each input.
- $\sigma_B$ is a vector of the population standard deviations, one for each input.
- $\hat{x}^{(i)}$ is a vector of zero-centered and normalized inputs for instance $i$.
- $y^{(i)}$ is the output of the BN operation for instance $i$.

#### Batch Normalization — Summary

**Training**

- Normalize activations according to mini-batch statistics.
- Update training population statistics via moving average of mini-batch statistics.

**Testing**

- Normalize activations according to estimated training population statistics.
- Do **not** update population statistics according to mini-batch statistics from test data.

#### Batch Normalization in TensorFlow

**Example.** Adding a `BatchNormalization` layer after every hidden layer and after the input layer:

```python
import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(2,), name='input')
BN1 = tf.keras.layers.BatchNormalization(name='bn1')(inputs)

hidden1 = tf.keras.layers.Dense(units=2, activation="elu", name='h1')(BN1)
BN2 = tf.keras.layers.BatchNormalization(name='bn2')(hidden1)

hidden2 = tf.keras.layers.Dense(units=2, activation="elu", name='h2')(BN2)
BN3 = tf.keras.layers.BatchNormalization(name='bn3')(hidden2)

output = tf.keras.layers.Dense(units=1, activation="linear", name='out')(BN3)

# Create model
model = tf.keras.Model(inputs=inputs, outputs=output)

model.summary()
```

**Output:**

```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input (InputLayer)           [(None, 2)]               0
_________________________________________________________________
bn1 (BatchNormalization)     (None, 2)                 8
_________________________________________________________________
h1 (Dense)                   (None, 2)                 6
_________________________________________________________________
bn2 (BatchNormalization)     (None, 2)                 8
_________________________________________________________________
h2 (Dense)                   (None, 2)                 6
_________________________________________________________________
bn3 (BatchNormalization)     (None, 2)                 8
_________________________________________________________________
out (Dense)                  (None, 1)                 3
=================================================================
Total params: 39
Trainable params: 27
Non-trainable params: 12
_________________________________________________________________
```

TensorFlow adds four parameters per input: $\gamma$, $\beta$, $\mu$, and $\sigma$. For example, the first batch normalization layer adds 8 parameters (four for each of the two inputs). The parameters $\mu$ and $\sigma$ are the moving averages and are not affected by backpropagation, and therefore they are called non-trainable in TensorFlow. The print screen shows that there are 12 non-trainable parameters, which is the sum of the parameters of all the batch normalization layers divided by 2: $(8 + 8 + 8)/2$.

There is debate about whether to add the Batch Normalization **after** the activation functions like we just did, or **before**. It is best to try both and see which option works best on your data. Before the activation function:

```python
import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(2,), name='input')
BN1 = tf.keras.layers.BatchNormalization(name='bn1')(inputs)

hidden1 = tf.keras.layers.Dense(units=2, name='h1')(BN1)
BN2 = tf.keras.layers.BatchNormalization(name='bn2')(hidden1)
hidden1act = tf.keras.layers.Activation("elu", name='h1act')(BN2)

hidden2 = tf.keras.layers.Dense(units=2, name='h2')(hidden1act)
BN3 = tf.keras.layers.BatchNormalization(name='bn3')(hidden2)
hidden2act = tf.keras.layers.Activation("elu", name='h2act')(BN3)

output = tf.keras.layers.Dense(units=1, activation="linear", name='out')(hidden2act)

# Create model
model = tf.keras.Model(inputs=inputs, outputs=output)

model.summary()
```

Note that above the activation is removed from the dense layers, and that we can set `use_bias = False` since batch normalization has an offset parameter per input.

#### Batch Normalization Performance

Benefits of batch normalization, summarized:

- Batch Normalization considerably improves deep neural nets by strongly reducing the vanishing gradient problem. It makes the networks much less sensitive to initialization (even saturating functions can be used).
- It regularizes the network, helping overfitting prevention.
- When you start using batch normalization you will notice that it slows down training, and that is because it needs to train extra parameters (scale and offset) for each node in a layer that is batch normalized. However, batch normalization will actually speed up learning because substantially fewer training steps will be required to reach the same model performance, because larger learning rates can be used. Each epoch will be slower but we will have to do fewer epochs.

---

### Solution 4: Gradient Clipping

- In contrast to the vanishing gradient problem there is also the **exploding gradient** problem. We will learn about the conditions that are necessary for this to happen when covering recurrent neural networks. One solution is to clip the gradients to be between a lower and upper bound.
- Choosing the threshold is the hardest part. You should use the biggest value such that you don't have an exploding gradient (whose effects can be NaNs or infinite values appearing in your tensors, constant loss/accuracy after a few training steps).
- The code below will clip every component of the gradient vector to a value between -1.0 and 1.0 and is called **clipping by value** of every component separately:

```python
optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer=optimizer)
```

For example a gradient vector with two partial derivatives `[0.8, 20.0]` will become `[0.8, 1.0]`. Before the clipping the direction was towards the second axis, and after clipping the direction is more towards the diagonal between the two axes. In other words, clipping by value **can change the direction** of the gradient vector.

- If we do not want to change the direction of the gradient then we can **clip by norm**. This involves changing all the components of the gradient vector if the L2 norm ($\sqrt{x_1^2 + x_2^2 + \ldots}$) surpasses a given threshold. The code below will rescale every component of the gradient vector if the L2 norm exceeds 1.0 so that the L2 norm equals 1.0.

```python
optimizer = tf.keras.optimizers.SGD(clipnorm=1.0)
model.compile(loss='mse', optimizer=optimizer)
```

For example, a gradient vector with two partial derivatives `[0.8, 20.0]` will become `[0.03996804, 0.99920096]`.

```python
import numpy as np
np.array([0.8, 20]) / np.sqrt(0.8**2 + 20**2)
```

This preserves the direction of the gradient: it is pointing primarily to the second axis before and after clipping.

---

## Avoiding Overfitting with Regularization

- Deep neural networks typically have tens of thousands of parameters, sometimes even millions.
- With so many parameters, the network has an incredible amount of freedom and can fit a huge variety of complex datasets.
- But this great flexibility also means that it is prone to over-fitting the training set.
- In this section, we will present some of the most popular regularization techniques for neural networks.

---

### Early Stopping

- Validation error can be used to detect when overfitting starts.
- The goal is to stop training once validation error reaches a minimum. This is called **early stopping** because we are stopping training before the training loss has converged. This is not a simple task, as we only know when we have reached a minimum once validation error starts increasing, and the error curve is not always smooth.
- One solution is to use **patience**: the number of epochs after which the validation error has remained above the minimum, before stopping training, and then rolling back the model parameters to the point where the validation error was at a minimum. The patience is often set somewhere between 10 and 20, but it really depends on the dataset and network.

*(Figure: training error decreases monotonically with epochs while validation error decreases then rises; training is stopped at the validation minimum.)*

#### Implementing Early Stopping

First, let's see how we can save and load models:

```python
# model.fit ....
# To save a model:
model.save('my_model.keras')
# To load a model:
model = tf.keras.models.load_model('my_model.keras')
```

- This will work with the Functional API, and saves the values of all model parameters and the optimizer including its hyperparameters and any state it may have.
- One typically has a script that trains the model, and then one or more other scripts to load the model and deploy it.

If we have `epochs > 1` in our fit call (i.e., we do not have a manual loop for epochs and rows), then we can also save a **checkpoint**. This is accomplished with a **callback** function. A callback function is run at specified times during the execution of the fit call (here at the end of every epoch, but it can also be after $x$ batches). In addition to running at the end of every epoch it also runs at the beginning and end of training.

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_model.keras',
    save_freq='epoch')
# filepath, if not specifically formatted to save the model
# in a new directory every time then the model will be
# overwritten each time to this filepath.
# (www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
# If save_freq = 'epoch' it saves the model after every epoch.
# save_freq can also be an integer, which denotes to save
# the model every save_freq batches.
model.fit(xtrain, ytrain, epochs=100,
          callbacks=[checkpoint_cb])
```

Provided that we are overwriting the model every time, and provided that we have a validation set during training (required for early stopping), then we can instruct TensorFlow to refrain from overwriting the **best** model based on validation performance. In other words TensorFlow will only save (overwrite) the model if it is better than before. This is accomplished with the `save_best_only = True` setting. By default it will monitor the validation loss for this purpose, but one can change this if desired to an arbitrary measure.

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_model.keras',
    save_freq='epoch',
    save_best_only=True)

model.fit(xtrain, ytrain,
          validation_data=(xval, yval),
          epochs=100,
          callbacks=[checkpoint_cb])

# roll back to the best model after 100 epochs:
model = tf.keras.models.load_model('my_model.keras')
```

With the implementation above we do not have to worry about training too long and overfitting. However, the model will run for 100 epochs unconditionally, which means we may be spending computational resources unnecessarily. We can expand the code to, in addition to saving and not overwriting the best model, have it stop training after a given number of epochs of remaining above the minimum validation loss. This number is called the **patience**.

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_model.h5',
    save_freq='epoch',
    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True)

model.fit(xtrain, ytrain,
          validation_data=(xval, yval),
          epochs=100,
          callbacks=[checkpoint_cb, early_stopping_cb])

# We do ****not*** need to run the following line if we are working
# in the same session since restore_best_weights = True in
# callbacks.EarlyStopping does that for us. Of course we would
# need this if we want to use the model in a separate session.
# model = tf.keras.models.load_model('my_model.keras')
```

- Unfortunately, all these convenience functions do not work (or are at the least inefficient) when we have a manual epoch and instance loop.
- In the bonus assignment we will implement early stopping when we are in this specific situation.

---

### L-1 and L-2 Regularization

**LASSO**, short for Least Absolute Shrinkage and Selection Operator, is a penalty added to the loss, and equal to the sum of the absolute value of all $n$ weights (i.e., the L1 norm):

$$\text{loss}(x, y, \theta) + \lambda\sum_{i=1}^{n}|\theta_i| \tag{20}$$

**Ridge (L2)** is a penalty added to the loss, and equal to the sum of all $n$ weights squared (i.e., the L2 norm):

$$\text{loss}(x, y, \theta) + \frac{\lambda}{2}\sum_{i=1}^{n}\theta_i^2 \tag{21}$$

**Elastic Net** is a mix of Lasso and Ridge, and how much each contributes is controlled by a mixing parameter $r$:

$$\text{loss}(x, y, \theta) + r\lambda\sum_{i=1}^{n}|\theta_i| + (1-r)\frac{\lambda}{2}\sum_{i=1}^{n}\theta_i^2 \tag{22}$$

**Details:**

- The Ridge (L2) loss is multiplied by $\frac{1}{2}$ because the derivative of $\theta^2$ is $2\theta$. This means in derivative space every weight counts as $\lambda\theta$, as opposed to $\lambda 2\theta$ (there is no reason to have the 2 in there).
- Penalties added to the loss are **sensitive to the scale of the data**. For example, a variable with a larger scale will have a smaller weight, and thus that variable will be penalized less. When using penalties, inputs should always be scaled.
- In L1 (LASSO), weights shrink by a constant amount, whereas in L2 (Ridge) the weights shrink by an amount proportional to the weights. When weights are large, L1 shrinks the weights a lot less than L2. When weights are small, L1 shrinks weights a lot more. For example, if an initial weight is small, 0.1, then L2 will grow by only 0.01 while L1 grows by 0.1. If the initial weight is 3, then L1 will grow by only 3, while L2 will grow by 9. Therefore, **L1 is said to be sparser than L2**. Small weights (those between -1 and 1) are penalized disproportionately little in L2 compared to weights outside that range, while in L1 they are penalized proportionally. This means that in L1 weights are sparser (they arrive at zero faster) than in L2 (it is much harder to get them to be zero).

**L1:**

```python
tf.keras.layers.Dense(10,
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l1(0.01))
```

**L2:**

```python
tf.keras.layers.Dense(10,
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

**Elastic Net** (note the lack of an explicit mixing parameter; we can set different $\lambda$ values instead):

```python
tf.keras.layers.Dense(10,
    activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))
```

---

### Dropout

- Dropout (Srivastava, Hinton, et al., 2014) is very popular because it works so well and is simple to understand.
- At every training step, every neuron (input neurons and hidden neurons, but **not** output neurons) has a probability $p$ of being temporarily **dropped out**. This means that it will be completely ignored (it outputs 0) during that training step, but it may be active during the next step.
- A typical dropout rate is between 10% and 50%.
- After training, neurons are not dropped anymore.
- Note that dropout slows down convergence, but results in better models.
- **Original paper:** <https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf>

*(Figure: (a) Standard Neural Net — fully connected; (b) After applying dropout — crossed-out units removed along with all their connections.)*

#### Why does Dropout work?

**Explanation 1: ensembling**

- A unique neural network is generated at each training step. Since each neuron can be either present or absent, there are a total of $2^n$ (2 for either present or absent) possible 'thinned' networks where $n$ is the total number of droppable neurons. These networks are not independent since they share many of their weights but they are all different.
- Therefore, while implementing dropout, we are training many different networks, each with different (but shared) parameters. And when it is time to make predictions (dropout is only active at training time), the resulting network roughly represents the combination of all these networks, which nearly always improves the performance of machine learning methods.
- Example: If we would train a network with 100,000 training iterations we have 100,000 different networks.
- This can be perceived of as an averaging **ensemble** of all these smaller networks.

**Explanation 2: resilience**

- Consider a soccer team with 11 players. Some players specialize playing in the front, some in the middle, and some in the back.
- For every game we would flip a coin for each of the players, to decide if they can play or not. This means that if we would do that many times we would have 5 to 6 players in the team every game.
- In the last game the key front player was not allowed to play, so another player had to play that position and while doing so, gathered valuable experience, which they would have never learned if all 11 players were playing. The team has therefore become much more resilient.
- It is unclear if this system would work well in soccer, but it does work really well in neural networks. Neurons cannot co-adapt with neighboring neurons, and they cannot rely excessively on a few input neurons (since any input neuron can also be dropped out). This makes neurons less sensitive to small changes in the inputs, which reduces overfitting.

#### Important Detail

- Remember that dropout is **on during training, and off during predicting**.
- If neurons are retained with probability 50% during training (the **keep probability**, or 1 − dropout probability), then every neuron will be connected to twice as many input neurons during prediction as it was on average during training.
- To compensate, during prediction after training, we need to **multiply all the incoming weights** of each neuron by 50% (the keep probability), otherwise the signal will be twice as large during prediction as what the network was trained on.
- Alternatively, we can **divide each neuron's output** by the keep probability during training. In other words, there will be fewer weights and therefore we boost the output. TensorFlow does this for us when using `tf.keras.layers.Dropout`; we do not need to do this manually.
- In sum, during prediction double (if keep probability was 50%) the weights will be active and so we need to either halve the signal by multiplying weights by 50% during prediction, or double the signal during training.

#### Practical Guidelines of Dropout

**Do not apply dropout:**

- right after the last layer, because the network has no ability to correct for dropout in subsequent layers.
- when the network is small relative to the dataset — regularization is usually unnecessary. If the model capacity is already low, lowering it further by adding regularization will hurt performance.
- when training time is limited. If we don't train until convergence (e.g., we use early stopping), dropout may give worse results. Usually dropout hurts performance at the start of training, but results in the final "converged" error being lower. Therefore, if we don't plan to train until convergence, we may not want to use dropout.

**Other things to try:**

- Increase the dropout rate when the model is overfitting, and decrease it when it is underfitting.
- Add dropout only after the last (few) hidden layers in case of underfitting.

#### TensorFlow Implementation of Dropout

- See **dropout.py** for a complete example. Each dropout layer drops some neurons (sets them to 0) and divides the remaining neurons' outputs by the keep probability.
- In 2016, Gal and Ghahramani made a modification to dropout during the prediction phase called **Monte Carlo Dropout (MC Dropout)** that was shown to improve prediction performance. They also made a profound connection to approximate Bayesian Inference and ensembling.
- Their simple suggestion was to turn dropout **on** during prediction, and deploy the model $k$ times on each record, resulting in $k$ predictions per record. These $k$ predictions can then be averaged (and we can also compute their standard deviation, for example). They showed that this average was a better prediction than making a single prediction on each record with dropout turned off.
- MC Dropout can be accomplished by tricking TensorFlow that it is in training mode, thereby activating the dropout layers. The following line will predict a different value each time for every instance:

```python
y_hat = model(X, training=True)
```

#### BatchNorm and Dropout at the Same Time?

When they are combined, they can lead to a worse performance. <https://arxiv.org/pdf/1801.05134.pdf>

---

### Data Augmentation

- One last regularization technique is data augmentation.
- It consists of generating new training instances from existing ones, artificially boosting the size of the training set.
- This will reduce overfitting. The trick is to generate **realistic** training instances. Ideally a human should not be able to tell which instances were generated and which ones were not.
- Adding simply white noise will not help because it is not learnable.
- Mostly used in Convolutional Neural Networks (see later).
- Instances can be generated on the fly during training.

---

## Depth vs Width

### Deep Neural Networks

- In recent years, neural networks have become deeper and deeper, with state-of-the-art networks going from 7 layers (AlexNet) to 1000 layers (Residual Nets) in the space of 4 years.
- However, the number of units in the fully-connected layers has remained roughly the same and is still relatively small in magnitude, despite the large increase in the number of layers.
- The advantage of multiple layers is that they attempt to model various levels of abstractions in data by using architectures composed of multiple non-linear transformations.
- The idea draws from how animals and humans learn: with simpler concepts earlier in life and higher-level abstractions later, expressed in terms of the previously learned concepts.
- Multiple layers are much better at generalizing because they learn all the intermediate features between the raw data and the high-level classification.

For example, if we train a deep convolutional neural network to classify images, we will find that the first layer will train itself to recognize very basic things such as lines and edges, the next layer will train itself to recognize collections of shapes such as eyes/noses/ears, and the next layer will learn even higher-order features such as faces.

| Low Level Features | Mid Level Features | High Level Features |
|---|---|---|
| Lines & Edges | Eyes & Nose & Ears | Facial Structure |

`Input → Shallow Layers → Middle Layers → Deeper Layers → Output`

### Wide Neural Networks

Then, how about going wider (i.e., more neurons, fewer layers)?

- The advantage of wide neural networks is that the model will be very good at **memorization**. Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data.
- Widening consistently improves performance across networks of different depth.
- Increasing width helps until the number of parameters becomes too large and stronger regularization is needed.
- Wider networks allow many multiplications to be computed in parallel, while deeper networks use more sequential computations (since the computations depend on the previous layer). Modern GPUs are better suited for parallel computing.
- Increasing depth results in better **generalization**; exploring new feature combinations that have never or rarely occurred in the past.
