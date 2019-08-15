### Lecture 5: Regularisation

#### Irrelevant / Multicolinear features
* Co-linearity between features
* Features not linearly independent
  * E.g. If $x_1$ and $x_2$ are the same $\Rightarrow$ perfectly correlated
* For linear model, feature $X_j$ is irrelevant if
  * $X_j$ is a linear combination of other columns
  $$
    X_{.j} = \Sigma_{l \neq j} \alpha_{l}X_{.l}
  $$
  for some scalars $\alpha_l$
  * Equivalently: Some eigenvalue of $X'X$ is zero
* Problems
  1. The solution is not **unique**
     * Infinite number of solutions
  2. Lack of interpretability
     * cannot interpret the weights
  3. Optimising to learn parameter is **ill-posed problem**


#### Ill-posed problems
* Well-posed problem
  1. a solution exists
  2. the solution is unique
  3. the solution's behavior changes continuously with the initial condition
* If ill-posed, there is **no closed form solution**
  * Closed form solution $\hat{w} = (X'X)^{-1}X'y$
  * But if irrelevant, $X'X$ has no inverse (singular)

#### Re-conditioning the problem (Ridge Regression)
* Make it a well-posed solution and also prevent fitting noise / overfitting
* Original problem: minimise squared error
$$
    || y - Xw ||_2^2
$$ 
* Regularised problem (L2, **Ridge regression**): minimise
$$
    || y - Xw ||_2^2 + \lambda||w||_2^2 \text{ for } \lambda > 0
$$
  * Turns the ridge into a peak ($\Rightarrow$ unique solution)
  * Adds $\lambda$ to eigenvalues of $X'X$: makes invertible

#### Regulariser as a prior (Bayesian intepretation of ridge regression)
* Let prior distribution be:
$$
  W \sim N(0, 1/\lambda)
$$
* Higher $\lambda$, more confidence at prior, therefore ignore data more
* Computing posterior and take MAP
$$
  \text{log}(posterior) = \text{log}(likelihood) + \text{log}(prior) - \text{log}(marg)
$$
  * can just ignore $\text{log}(marg)$, since this term doesn't affect optimisation
* Arrive at the problem of minimising:
$$
  || y - Xw ||_2^2 + \lambda||w||_2^2
$$
* Become equivalent problem: Ridge Regression

---

#### Regularisation in non-linear models
* There is trade-off between **overfitting** and **underfitting**
* Right model class $\Theta$ will sacrifice some traininig error, for test error
* Choosing model complexity (2 Methods)
  1. Explicit model selection
  2. Regularisation

#### Explicit model selection
* Using hold-out or CV to select the model
1. Split data into $D_{train}$ and $D_{validate}$ sets
2. For each degree d (# of parameters), we have model $f_d$
   1. Train $f_d$ on $D_{train}$
   2. Test (evaluate) $f_d$ on $D_{validate}$
3. Pick degree $\hat{d}$ that gives the best test score
4. Re-train model $f_{\hat{d}}$ using all data (return this final model)


#### Regularisation
* Solving the problem:
$$
  \hat{\theta} \in \argmin_{\theta \in \Theta} (L(data, \theta) + \lambda R(\theta))
$$
  * E.g. Ridge regression
  $$ 
    \hat{w} \in \argmin_{w \in W} ||y - Xw||_2^2 + \lambda||w||_2^2
  $$
* Note: regulariser $R(\theta)$ doesn't depend on data
* Use held-out validation / cross validation to choose $\lambda$

