### Lecture 3: Linear Regression & Optimisation

#### Linear Regression
* Assume a probabilistic model
    * $y = X\beta + \epsilon$
* Assume Gaussian noise (independent of X):
    * $\epsilon \sim N(0, \sigma^2)$
* Discriminative model
    * $p(y|\mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}(-\frac{(y-\mathbf{x}\beta)^2}{2\sigma^2})$
* Unknown param: $\beta$ (and $\sigma^2$)
* MLE: choose param values that maximise the probability of observed data (likelihood)
  * "Log trick": instead of maximising likelihood, we can maximise log-likelihood
* Under this model, MLE is equivalent to minimising SSE

#### Optimization
* Training = Fitting = Parameter estimation
* Typical formulation (minimise loss = objective)
  * $\hat{\theta} \in \argmin_{\theta \in \Theta} L(data, \theta)$
* Analytic (aka. closed form) solution
  * 1st order derivatives for optimality:
    * $\frac{\partial L}{\partial \theta_1} = ... = \frac{\partial L}{\partial \theta_p} = 0$
  * Approximate iterative solution (e.g. IRWLS)
    * Initialisation: choose starting guess $\mathbf{\theta}^{(1)}$, set $i=1$
    * Update: $\theta^{(i+1)} \leftarrow SomeRule[\theta^{(i)}]$, set $i \leftarrow i+1$
    * Termination: decide whether to stop
    * Go to step 2
    * Stop: return $\hat{\theta} \approx \theta^{(i)}$

#### Coordinate descent
* Suppose $\theta = [\theta_1, ..., \theta_K]^{T}$
  1. Choose $\theta^{(1)}$ and some $T$
  2. For $i$ from $1$ to $T$ (update all params $T$ times)
     * $\theta^{(i+1)} \leftarrow \theta^{(i)}$ (copy param values)
     * For $j$ from 1 to $K$: (update one param each time)
        * Fix components of $\theta^{(i+1)}$, except j-th component
        * Find $\hat{\theta}_j^{(i+1)}$ that minimises $L(\theta_j^{(i+1)})$
        * Update j-th component of $\theta^{(i+1)}$
  3. Return $\hat{\theta} \approx \theta^{(i)}$
* (Other stopping criteria can be used)

#### Gradient descent
* Gradient denoted as $\nabla L = [\frac{\partial L}{\partial \theta_1}, ..., \frac{\partial L}{\partial \theta_p}]^{T}$
* Algorithm: 
  1. Choose $\theta^{(1)}$ and some $T$
  2. For $i$ from $1$ to $T^*$
     * update: $\theta^{(i+1)} = \theta^{(i)} - \eta \nabla L (\theta^{(i)})$
  3. Return $\hat{\theta} \approx \theta^{(i)}$
* Note: $\eta$ (learning rate) is dynamically updated in each step
* Variants: SGD, mini batches, momentum, AdaGrad

#### Convex objective functions
* "Bowl" shaped functions
* Every local min is global min
* Informal definition: line segment between any two points on graph of function lies above or on the graph
* Gradient descent on (strictly) convex function guaranteed to find a (unique) global minimum

#### $L_1$ and $L_2$ norms
* Norm: length of vectors
* $L_2$ norm (aka. Euclidean distance)
  * $||a|| = ||a||_2 \equiv \sqrt{a_1^2 + ... + a_n^2}$
* $L_1$ norm (aka. Manhattan distance)
  * $||a||_1 \equiv |a_1| + ... + |a_n|$
* E.g. Sum of squared errors:
  * $L = \sum_{i=1}^n (y_i - \sum_{j=0}^{m}X_{ij}\beta_{j})^2 = ||y - X\beta||^2$

#### Linear Regression optimization: Least Square Method
* To find $\beta$, minimize the **sum of squared errors**:
  * $SSE = \sum_{i=1}^n (y_i - \sum_{j=0}^m X_{ij}\beta_{j})^2$
* Setting derivative to zero and solve for $\beta$: (normal equation)
  * $b = (X^TX)^{-1}X^{T}y$

---

### Lecture 4: Logistic Regression & Basis Expansion

#### Logistic Regression
* Why not linear regression for classification?
  * Predict "Yes" if $s \geq 0.5$
  * Predict "No" if $s < 0.5$
  * Reason:
    * Can be susceptible (易受影响) to outliers
    * least-squares criterion looks **unnatural** in this setting
* Problem: the probability needs to be between 0 and 1
* Logistic function
  * $f(s) = \frac{1}{1+\text{exp}(-s)}$
* Logistic regression model:
  * $P(Y=1|x) = \frac{1}{1+\text{exp}(-x^T\beta)}$
  * In GLM:
    * Mean: $\mu = P(Y=1 | x)$
    * Link function: $g(\mu) = log \frac{P(Y=1 | x)}{P(Y=0 | x)} = \eta = x^T\beta$ (log odds)
  * (Can use Cross-Validation to choose the threshold, usually just use 0.5)
* Logistic regression is a linear classifier
  * Logistic regression model:
    * $P(Y = 1 | x) = \frac{1}{1 + \text{exp}(-x^T\beta)}$
  * Classification rule:
    * Class "1" if $P(Y=1 | x) > \frac{1}{2}$, else class "0"
    * Decision boundary (line): $p = \frac{1}{1 + \text{exp}(-x^T\beta)} = \frac{1}{2}$
    * (In higher dimensional problems, the decision boundary is a plane or hyperplane, vector $\beta$ is perpendicular to the decision boundary)

#### Linear v.s. logistic probabilistic models
* Linear regression
  * Assume $\epsilon \sim N(0, \sigma^2)$
  * Therefore assume $y \sim N(X\beta, \sigma^2)$
* Logistic regression
  * Assume $y \sim Bernoulli(p)$

#### Logistic MLE
* Doesn't have closed form solution (cannot solve $\frac{\partial L}{\partial \beta} = 0 $ directly)
* Therefore use iterative optimisation
  * E.g. Newton-Raphson, IRWLS, or gradient descent
* Good news: it's a convex problem $\Rightarrow$ guaranteed to get global minimum

#### Cross entropy
* To compare models with different numbers of parameters in an all-subsets search, one may use an information theoretic criterion (e.g. AIC, BIC, ...) which estimates information divergence between true model and a given candidate model (working model)
* Best model: **smallest** criterion value
* 