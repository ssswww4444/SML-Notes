### Lecture 1: Introduction Probability Theory

#### Terminologies
* **Instance**: measurements about individual entities/objects
* **Attributes**: component of the instances
* **Label**: an outcome that is categorical, numerical, etc.
* **Examples**: instance coupled with label
* **Models**: discovered relationship between attributes and/or label

#### Supervised v.s. Unsupervised
* **Supervised**: 
  * Labelled data
  * Predict labels on new instances
* **Unsupervised**: 
  * Unlabelled data
  * Cluster related instances; project to fewer dimensions; understand attribute relationships

#### Evaluation
1. Pick an evaluation metric comparing label v.s. prediction
2. Procure an independent, labelled test set
3. "Average" the evaluation metric over the test set
(When data poor, use cross-validation)

Probability相关的部分就不写了

---

### Lecture 2: Statistical Schools of Thoughts

#### Frequentist statistics
* Unknown params are treated as having fixed but unknown values
* Parameter estimation:
  * Classes of models indexed by parameters
  * Point estimate: a function (or statistic) of data (samples)
* If T is an estimator for $\theta$
  * Bias: $Bias_{\theta}(T) = E_{\theta}[T] - \theta$
  * Variance: $Var_{\theta}(T) = E_{\theta}[(T - E_{\theta}[T])^2]$
* Asymptotic properties:
  * Consistency: $T \rightarrow \theta$ (converges in probability) as $n \rightarrow \infty$
  * Efficiency: asymptotic variance is as small as possible
* Maximum-Likelihood Estimation (MLE)
  * General principle for designing estimators
  * Involves optimisation
  * $T \in \argmax_{\theta \in \Theta} \prod_{i=1}^{n} p_{\theta}(x_i)$
  * MLE estimators are consistent (but usually biased)
  * "Algorithm":
    1. Given data $X_1, ..., X_n$
    2. Likelihood: $L(\theta) = \prod_{i=1}^{n} p_{\theta}(X_i)$
    3. Optimise to find best params
        * Take partial derivatives of log likelihood: $l'(\theta)$
        * Solve $l'(\theta) = 0$

#### Decision Theory
