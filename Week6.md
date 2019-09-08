### Lecture 10: Soft-Margin SVM, Lagrangian Duality

#### Soft-Margin SVM
* One of the three approach to fit non-linearly separable data
  1. Transform the data (kernel)
  2. Relex the constraints (Soft-Margin)
  3. Combination of (1) and (2)
* Relax constraints to allow points to be:
  * **Inside the margin**
  * Or on the **wrong side of the boundary**
* Penalise boundaries by the **extent of "violation"** (distance from margin to wrong points)

#### Hinge loss: soft-margin SVM loss
* **Hard-margin SVM loss:**
  * $l_\infty = 0$ if prediction correct
  * $l_\infty = \infty$ if prediction wrong
* **Soft-margin SVM loss: (hinge loss)**
  * $l_h = 0$ if prediction correct
  * $l_h = 1 - y(w'x + b) = 1 - y\hat{y}$ if prediction wrong (penalty)

#### Soft-Margin SVM Objective
* $\argmin_{\mathbf{w}, b} (\sum_{i = 1}^n l_h(x_i, y, w, b) + \lambda ||w||^2)$
  * Like ridge regression
* Reformulate objective:
  * Define **slack variables** as upper bound on loss
    * Allow you to relax the constraint
    * $\xi_i \geq l_h = \max(0, 1 - y_i (w'x_i + b))$
    * Non-zero means there is some violation
    * Don't like function like this in optimisation (no derivative)
  * **Then, new objective:**
    * $\argmin_{w,b,\xi} (\frac{1}{2} ||w||^2 + C \sum_{i = 1}^n \xi_i)$ 
    * Constraints:
      * $\xi \geq 1 - y_i (w'x_i + b)$ for $i = 1, ..., n$
      * $\xi \geq 0$ for $i = 1, ..., n$
      * (Penalise based on the size of $\xi$, like having loss function in objective)
      * **$\xi$ gets pushed down to be equal to $l_h$**
    * C: hyperparameter (have to tune by gridSearch)

#### Two variations of SVM
* **Hard-margin SVM objective:**
  * $\argmin_{w,b} \frac{1}{2}||w||^2$
  * s.t. $y_i(w'x_i + b) \geq 1$ for $i = 1, ..., n$
* **Soft-margin SVM objective:**
  * $\argmin_{w,b} \frac{1}{2}||w||^2$
  * s.t. $y_i(w'x_i + b) \geq 1 - \xi_i$ for $i = 1, ..., n$ and $\xi \geq 0$ for $i = 1, ..., n$ 
  * The constraints are **relaxed** by allowing violation by $\xi_i$

#### Constraint optimisation
* **Canonical form:**
  * minimise $f(x)$
  * s.t. $g_i(x) \leq 0, i = 1, ..., n$
  * and $h_j(x) = 0, j = 1, ..., m$
* Training SVM is also a constrained optimisation problem
* Method of **Lagrange multipliers**
  * Transform to unconstrained optimisation
  * Transform **primal** program to a related **dual** program
  * Analyze necessary & sufficient conditions for solutions of both program

#### Lagrangian and duality
* **Dual** objective function:
  * $L(x, \lambda, v) = f(x) + \sum_{i = 1}^n \lambda_i g_i(x) + \sum_{j = 1}^m v_j h_j(x)$
  * Primal constraints became penalties
  * Called **Lagrangian** function
  * New $\lambda$ and $v$ are called the **Lagrange multipliers** or **dual variables**
* Primal program: $\min_x \max_{\lambda \geq 0} L(x, \lambda, v)$
* Dual program: $\max_{\lambda \geq 0, v} \min_{x} L(x, \lambda, v)$
* Duality
  * Weak duality: dual optimum $\leq$ primal optimum
  * For convex problem, we have strong duality: optima coincide (same optima for primal and dual)
    * Including SVM

#### Dual program for hard-margin SVM
* Minimise Lagrangian w.r.t to primal variables <=> maximise w.r.t dual variables yields the **dual program:**
  * $\argmax_\lambda \sum_{i = 1}^m \lambda_i - \frac{1}{2}\sum_{i = 1}^n \sum_{j = 1}^n \lambda_i \lambda_j y_i y_j x'_i x_j$
  * s.t. $\lambda_i \geq 0$ and $\sum_i^n \lambda_i y_i = 0$
* According to strong duality, solve dual <=> solve primal

#### Making predictions with dual solution
* **Recovering primal variables**
  * From stationarity: get $w_j^*$
    * $w_j^* = \sum_{i = 1}^n \lambda_i y_i (x_i)_j = 0$
  * From dual solution: (get $b^*$)
    * $y_j(b^* + \sum_{i = 1}^n \lambda_i^* y_i x'_i x_j) = 1$
    * For any example $j$ with $\lambda_i^* > 0$ **(support vectors)**
* Make predictions (testing)
  * Classify new instance $x$ based on sign of
    * $s = b^* + \sum_{i = 1}^n \lambda_i^* y_i x'_i x$
    * ($s = w'x + b$)

#### Optimisation for Soft-margin SVM
* Training: find $\lambda$ that solves (dual)
  * $\argmax_\lambda \sum_{i = 1}^m \lambda_i - \frac{1}{2}\sum_{i = 1}^n \sum_{j = 1}^n \lambda_i \lambda_j y_i y_j x'_i x_j$
  * s.t. $C \geq \lambda_i \geq 0$ and $\sum_i^n \lambda_i y_i = 0$
  * Where $C$ is a box constraints **(only difference between soft and hard SVM)**
    * Vector $\lambda$ is inside a box of side length $C$
    * Big $C$: penalise more training data, let training data has more influence
    * Small $C$: don't care about training data, want big margins
* Make predictions: (same as hard margin)
  * Classify new instance $x$ based on sign of
    * $s = b^* + \sum_{i = 1}^n \lambda_i^* y_i x'_i x$

#### Complementary slackness
* One of the KKT conditions:
  * $\lambda_i^* (y_i((w^*)'x_i + b^*) - 1) = 0$
* Remember:
  * $y_i (w'x_i + b) - 1 > 0$ means that $x_i$ is outside the margin (classified correctly)
* Points outside the margin must have $\lambda_i^* = 0$
* Points with non-zero $\lambda^*$ are **support vectors**
  * $w^* = \sum_{i = 1}^n \lambda_i y_i x_i$
  * Other points has no influence on $w^*$ (orientation of hyperplane)

#### Training SVM
* Inefficient
* Many $\lambda$s will be zero (sparsity)