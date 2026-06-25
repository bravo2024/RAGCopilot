# ML Basics

## Supervised learning

Supervised learning uses labeled training data where each example has an input and a known output label. The model learns to map inputs to outputs by minimizing a loss function on the training set.

Common supervised tasks include classification (predict a discrete label) and regression (predict a continuous value). Algorithms include logistic regression, decision trees, support vector machines, and gradient boosting.

## Unsupervised learning

Unsupervised learning finds hidden patterns or structure in unlabeled data. Common tasks include clustering, dimensionality reduction, and density estimation.

Examples include k-means clustering, DBSCAN, principal component analysis (PCA), and autoencoders.

## Reinforcement learning

Reinforcement learning trains an agent to make decisions by interacting with an environment. The agent receives rewards or penalties for actions and learns a policy that maximizes cumulative reward.

Classic algorithms include Q-learning, SARSA, and policy gradient methods. Modern approaches use deep neural networks for the value function or policy.

## Overfitting

Overfitting occurs when a model learns noise in the training data instead of the underlying signal. Symptoms include high training accuracy but poor test accuracy.

Regularisation techniques (L1, L2, dropout), early stopping, and data augmentation help prevent overfitting.

## Cross-validation

Cross-validation splits data into k folds, trains on k-1 folds, and evaluates on the held-out fold. It gives a more reliable estimate of model performance than a single train-test split.

K-fold, stratified k-fold, and leave-one-out are the most common variants.

## Gradient descent

Gradient descent iteratively updates model parameters in the direction of the negative gradient of the loss function. The learning rate controls step size; too large can diverge, too small converges slowly.

Variants include SGD, Adam, RMSProp, and L-BFGS.

## Confusion matrix

A confusion matrix tabulates true positives, true negatives, false positives, and false negatives. From these you can derive accuracy, precision, recall, specificity, and F1-score.

## Bias-variance tradeoff

The bias-variance tradeoff captures the tension between underfitting (high bias) and overfitting (high variance). Simple models have high bias; complex models have high variance.

## Feature engineering

Feature engineering transforms raw data into informative predictors. Techniques include scaling, one-hot encoding, binning, interaction terms, and domain-specific aggregates.

## Ensemble methods

Ensemble methods combine multiple weak learners to produce a strong learner. Bagging (e.g., Random Forest) reduces variance; boosting (e.g., XGBoost) reduces bias sequentially.
