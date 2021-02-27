
# Data Science NOTES from Ng Course - Oreilly

There are a lot of statistical tests and information. Mostly for the purpose of statistical analysis. You do not need all of these for data science.

Data science focus is on prediction and having models that work on predicting real data. It is not concerned as much with correct specifications of statistical problems.

# 2 . Regresssion

**Regressions**
- Robust reggression 
- Multiple regression with statsmodels, Multiple Regression and Feature Importance
- OLS and Gradient Descent 
- Regularized Regression( restrict overfitting) - Ridge, Lasso, Elastic Net
- Polynomial Regression. FOr models that are not Linear

**Performance Evaluation**
- How does it perform when it is out of sample? 

**Dealing with Non-linear Relationships**
- Decision Tree, Random Forest, AdaBoost

**Feature Importance**

**Data Pre-processing**
- Exploratory Data Analysis (EDA)
- standardization / Mean removal / Variance Scaling
- Min-Max or Scaling Features to a Range
- Normalization
- Binarization
- Encoding categorical features
   - labelEncoder
   - One Hot / One-of-K encodeing
        
**Variance Bias trade off**
- Validation Curve 
- Learning Curve

**Cross Validation** 
- Hold out CV
- k-fold CV
- Stratified k-fold


### Regression

Regression deals with Continuous variables. You use the explanatory variable to predict the outcome.

X = predictor/explanatory variable/independent variable.

Y = dependent variable/outcome
eg use crime rate and room size to predict house value

### Reinforcement learning
Trying to develop a system or agent that improves performance based on feedback or interactions with the environment. This was used by Google on the game of GO.
Its neither supervised ( not given labeled data, not left by itself either) 
- given an agent to learn what the utility function, the agent/system( learn the action function and the value function).
Learning ..to every action there is a reaction. Does it get a reward or an punishment?

JPMorgan has a reinforcment agent on the stock market. Apric ??

### Unsupervised 
Exploratory data analysis. Feed raw data and the algorithm will cluster to meaningful subgroups.

# Simple Linear Regression - Supervised Learning example: 
### When x is _, y is _
Linear Regression aims to establish if there is a **statistically significant** relationship between two variables. Eg Income and spending, location and price, sugar and health. We may also want to **forcast** new observations. It is called Linear because if we plot it on a bi-dimensional plot it will create a straight line.

- Dependent variable: the value we want to forecast/explain. Denoted as `Y`.
- Independent variable: the value that explains Dependent. Denoted as `X`.
- Errors/Difference between the real data points and the Line of the regression model represented as $\epsilon$ 

The Linear equation model: $y$ = $\beta_0$ + $\beta_1$ $x$ + $\epsilon$

Where 
- $\beta_0$ the constant/intercept of x
- $\beta_1$ the coeficient/slope of x
- $\epsilon$ is the error term we are trying to minimise

[More Theory on Youtube](https://www.youtube.com/watch?v=owI7zxCqNY0)




## Model Statistical Outputs:

**Dep. Variable**: The dependent variable or target variable

**Model**: Highlight the model used to obtain this output. It is OLS here. Ordinary least squares / Linear regression

**Method**: The method used to fit the data to the model. Least squares

**No. Observations**: The number of observations

**DF Residuals**: The degrees of freedom of the residuals. Calculated by taking the number of observations less the number of parameters

**DF Model**: The number of estimated parameters in the model. In this case 13. The constant term is not included.

**R-squared**: This is the coefficient of determination. Measure of goodness of fit. Tells you how much your model can explain the variability of the data. The more parameters you add to your model , the more this value will go up. 

$$R^2=1-\frac{SS_{res}}{SS_{tot}}$$

From [wiki](https://en.wikipedia.org/wiki/Coefficient_of_determination),

The total sum of squares, $SS_{tot}=\sum_i(y_i-\bar{y})^2$

The regression sum of squares (explained sum of squares), $SS_{reg}=\sum_i(f_i-\bar{y})^2$

The sum of squares of residuals (residual sum of squares), $SS_{res}=\sum_i(y_i-f_i)^2 = \sum_ie^2_i$

**Adj. R-squared**: This is the adjusted R-squared. It is the coefficient of determination adjusted by sample size and the number of parameters used. It is not normally used on Linear Regression. IT is trying to penalise the additional paramaters you add to your model. 
$$\bar{R}^2=1-(1-R^2)\frac{n-1}{n-p-1}$$

$p$ = The total number of explanatory variables not including the constant term

$n$ = The sample size

**F-statistic**: A measure that tells you if your model is different from a simple average or not. 

**p-value - Prob (F-statistic)**: This measures the significance of your F-statistic. Also called **p-value of F-statistic**. In statistics and before computeres got so powerful, a p-value equal or lower than 0.05 is considered significant, now it could be as low as 0.005.

**AIC**: This is the Akaike Information Criterion. It evaluatess the model based on the model complexity and number of observations. The lower the better. 

**BIC**: This is the Bayesian Information Criterion. Similar to AIC, except it pushishes models with more parameters.Again, The lower the better. 


## Parameters Estimates and the Associated Statistical Tests

**coef**: The estimated coefficient. Note that this is just a point estimate. By itself , it doesn't mean much. The best way to read the significance of a number is to compare it against the p-value.

**P > |t|**: The p-value. A measure of the probability that the coefficient is different from zero. the closer to zero, the more significant.

**std err**: The standard error of the estimate of the coefficient. Another term for standard deviation

**t**: The t-statistic score. This should be compared against a t-table.


**[95.0% Conf. Interval]**: The 95% confidence interval of the coefficient. Shown here as [0.025, 0.975], the lower and upper bound.


## Residual Tests
The residual has substantial significance. The reason is that the residual, after the model has explained the variability of the data, should have left a residual that is random , with normal distribution and not have any pattern. IF this is not the case the model has missed out on some signals. 

**Skewness**: This is a measure of the symmetry of the residuals around the mean. Zero if symmetrical. A positive value indicates a long tail to the right; a negative value a long tail to the left.

**Kurtosis**: This is a measure of the shape of the distribution of the residuals, the peakness. A normal distribution has a zero measure. A negative value points to a flatter than normal distribution; a positive one has a higher peak than normal distribution. 

**Omnibus D'Angostino's test**: This is a combined statistical test for skewness and kurtosis.

**Both Skewness and Kurtosis indiacte problems with the model**

**Prob(Omnibus)**: p-value of Omnibus test.

**Jarque-Bera**: This is a combined statistical test of skewness and kurtosis.

**Prob (JB)**: p-value of Jarque-Bera.

**Durbin-Watson**: This is a test for the presence of correlation among the residuals. This is especially important for time series modelling.

**Cond. No**: This is a test for multicollinearity. > 30 indicates unstable results. Meaning within your data some of explanitory factor share the same variance (they are correlated(.

# Feature Importance and Extractions
Check
1. The Direction of the coefficient. Is it going Up or Down. Positive or negative

2. Impact of the variable/ factor on the model. How significant is it to the scheme of things ( the model) 
- In order to measure the impact we need to standardise the variable. As an example, price of a house could range between 100k to 700k , compared to age with is just 0-100. We would want everything in the same range so nothing skews the model in to thinking it is more imporant than it actually is.

## Detecting collinearity with Eigenvectors
Use the numpy linear algebra eigen decomposition of the correlation matrix

## Use $R^2$ to identify key Features
- Compare $R^2$ of the model **VS** $R^2$ of the model **without the feature**
- A significance change in $R^2$ signifies the importance of the feature.


# Gradient Descent

Inspired by [Chris McCormick on Gradient Descent Derivation](http://mccormickml.com/2014/03/04/gradient-descent-derivation/)

# Background

$h(x) = \theta_0 + \theta_1X$

Find the values of $\theta_0$ and $\theta_1$ which provide the best fit of our hypothesis to a training set. 

The training set examples are labeled $x$, $y$, 

$x$ is the input value and $y$ is the output. 

The $i$th training example is labeled as $x^{(i)}$, $y^{(i)}$.

## MSE Cost Function

The cost function $J$ for a particular choice of parameters $\theta$ is the mean squared error (MSE):

$$J(\theta)=\frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$$

$m$ The number of training examples

$x^{(i)}$ The input vector for the $i^{th}$ training example

$y^{(i)}$ The class label for the $i^{th}$ training example

$\theta$ The chosen parameter values of weights ($\theta_0, \theta_1, \theta_2$)

$h_{\theta}(x^{(i)})$ The algorithm's prediction for the $i^{th}$ training example using the parameters $\theta$

The MSE measures the mean amount that the model's predictions deviate from the correct values.

It is a measure of the model's performance on the training set. 

The cost is higher when the model is performing poorly on the training set. 

The objective of the learning algorithm is to find the parameters $\theta$ which give the minimum possible cost $J$.


This minimization objective is expressed using the following notation, which simply states that we want to find the $\theta$ which minimizes the cost $J(\theta)$.

$$\min_{\theta}J(\theta)$$

## Regularised Method for Regression


These examples below each have their own way of regularising the coefficient:

* [Ridge Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
* [Least Absolute Shrinkage and Selection Operator (LASSO)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
* [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

## Ridge Regression
Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

Ridge regression addresses some of the problems of **Ordinary Least Squares** by imposing a penalty on the size of coefficients. Especially those problems caused by outliers wich shift the coefficient quite substantially. The ridge coefficients minimize a penalized residual sum of squares,

$$\min_{w}\big|\big|Xw-y\big|\big|^2_2+\alpha\big|\big|w\big|\big|^2_2$$

where by 
$$\min_{w}\big|\big|Xw-y\big|\big|^2_2$$
is the minimum of your essitmates minus your y; squared.

and 
$$\alpha\big|\big|w\big|\big|^2_2$$ is your penalty term



$\alpha>=0$ is a complexity parameter that controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity .

Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.

Shows the effect of collinearity in the coefficients of an estimator.

Ridge Regression is the estimator used in this example. Each color represents a different feature of the coefficient vector, and this is displayed as a function of the regularization parameter.

This example also shows the usefulness of applying Ridge regression to highly ill-conditioned matrices. For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).

# Summary

[Question in StackExchange](https://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge)

**When should I use Lasso, Ridge or Elastic Net?**

* **Ridge regression** can't zero out coefficients; You either end up including all the coefficients in the model, or none of them. 

* **LASSO** does both parameter shrinkage and variable selection automatically. 

* If some of your covariates are highly correlated, you may want to look at the **Elastic Net** instead of the LASSO.

# Other References

1. [The Lasso Page](http://statweb.stanford.edu/~tibs/lasso.html)

2. [A simple explanation of the Lasso and Least Angle Regression](http://statweb.stanford.edu/~tibs/lasso/simple.html)

3. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)


# Data Pre-Processing 

When it comes to continuous nominal data, There are 4 main types of preprocessing
- Standardization / Mean Removal
- Min-Max or Scaling Features to a Range
- Normalization
- Binarization

**Assumptions**:
* Implicit/explicit assumption of machine learning algorithms: The features follow a normal distribution (Bell-curve , +- 3SD).
* Most method are based on linear assumptions
* Most machine learning requires the data to be standard normally distributed. Gaussian with zero mean and unit variance (SD/var = 1). IF we don't have variance and SD of one , it will be hard for the ML model to converge to a reasonable solution.

[scikit-learn:](http://scikit-learn.org/stable/modules/preprocessing.html) In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.

For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) **assume that all features are centered around zero and have variance in the same order**. Variance referring to multiple features. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

# Standardization / Mean Removal / Variance Scaling

[scikit Scale](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

Mean is removed. Data is centered on zero. This is to remove bias.

Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance. standard normal random variable with mean 0 and standard deviation 1.

$$X'=\frac{X-\bar{X}}{\sigma}$$

Where 

${X-\bar{X}}$

Keeping in mind that if you have scaled your training data, you must do likewise with your test data as well. However, your assumption is that the mean and variance must be invariant between your train and test data. `scikit-learn` assists with a built-in utility function `StandardScaler`.



# Min-Max or Scaling Features to a Range

Scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size.

The motivation to use this scaling include robustness to very small standard deviations of features and preserving zero entries in sparse data.


doc:

Init signature: preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

Transforms features by scaling each feature to a given range.

This estimator scales and translates each feature individually such
that it is in the given range on the training set, i.e. between
zero and one.

The transformation is given by::

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    
$$X_{std}=\frac{X-X_{min}}{X_{max}-X_{min}}$$

$$X'=X_{std} (\text{max} - \text{min}) + \text{min}$$

## MaxAbsScaler

Works in a very similar fashion, but scales in a way that the training data lies within the range `[-1, 1]` by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data.

## Scaling sparse data

Centering sparse data would destroy the sparseness structure in the data, and thus rarely is a sensible thing to do. 

However, it can make sense to scale sparse inputs, especially if features are on different scales.

`MaxAbsScaler` and `maxabs_scale` were specifically designed for scaling sparse data
[Compare the effect of different scalers on data with outliers](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)


## Scaling vs Whitening

It is sometimes not enough to center and scale the features independently, since a downstream model can further make some assumption on the linear independence of the features.

To address this issue you can use `sklearn.decomposition.PCA` or `sklearn.decomposition.RandomizedPCA` with `whiten=True` to further remove the linear correlation across features.


# Normalization
Normalization is the process of scaling individual samples to have unit norm. 

This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
What you do is take the mean away from each value and divide it by the range. 

$$X'=\frac{X-X_{mean}}{X_{max}-X_{min}}$$

This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.

There are two types of Normalization

  1. **L1 normalization**, Least Absolute Deviations
ensure the sum of absolute values is 1 in each row. 

  2. **L2 normalization**, Least squares, 
Ensure that the sum of squares is 1.

# Binarization


$$f(x)={0,1}$$

Feature binarization is the process of thresholding numerical features to get boolean values. You make a value either 0 or 1 depending on a threshold.  This can be useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution. Bernoulli is either zero or one. 


It is also common among the text processing community to use binary feature values (probably to simplify the probabilistic reasoning) even if normalized counts (a.k.a. term frequencies) or TF-IDF valued features often perform slightly better in practice.

# Encoding categorical features

There are two main ways to do this
## Label encodeing  
- The value is coded to a numerical value for example country to code mapping:
  - Australia 	 0
  - Hong Kong 	 1
  - New Zealand  2
  - Singapore 	 3

The problem with this is a Machine might think Singapore is more important that Hong Kong. To get around this they may use One Hot / One-of-K Encoding.

## One Hot / One-of-K Encoding

* Useful for dealing with sparse matrix
* uses [one-of-k scheme](http://code-factor.blogspot.sg/2012/10/one-hotone-of-k-data-encoder-for.html)

The process of turning a series of categorical responses into a set of binary result (0 or 1)
For example if we use One Hot encodeing on the four countries again we would get a matrix as follows:
[[1. 0. 0. 0.]  = Australia
 [0. 0. 0. 1.]  = Hong Kong
 [0. 0. 1. 0.]  = New Zealand
 [0. 1. 0. 0.]] = Singapore

These values can be inverted so we could call [3, :] and get Singapore returned to us. 

# Data Pre-Processing References

* [Section - Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)
* [Colleen Farrelly - Machine Learning by Analogy](https://www.slideshare.net/ColleenFarrelly/machine-learning-by-analogy-59094152)
* [Lior Rokach - Introduction to Machine Learning](https://www.slideshare.net/liorrokach/introduction-to-machine-learning-13809045)
* [Ritchie Ng](http://www.ritchieng.com/machinelearning-one-hot-encoding/)


# Bias Variance trade off

Every estimator ( Linear Regression, SVM etc ) has its advantages and drawbacks. Its generalization error can be decomposed in terms of bias, variance and noise. 
- The **bias** of an estimator is its average error for different training sets. These are the errors.
- The **variance** of an estimator indicates how sensitive it is to varying training sets. 
- Noise is a property of the data.

Bias and variance are inherent properties of estimators and we usually have to select learning algorithms and hyperparameters so that both bias and variance are as low as possible. Another way to reduce the variance of a model is to use more training data. However, you should only collect more training data if the true function is too complex to be approximated by an estimator with a lower variance. 

# Validation Curve

* The purpose of the validation curve is for the identification of over- and under-fitting
* Plotting training and validation scores vs model parameters.

# Learning Curve

* Shows the validation and training score of an estimator for varying numbers of training samples. This is VS the training sample. Where as the Validation Curve shows the training and validation scores vs model (hyper)parameters.
* A tool to find out how much we benefit from adding more training data and whether the actual estimator suffers more from a variance error or a bias error. 
* If both the validation score and the training score converge to a value that is too low with increasing size of the training set, we will not benefit much from more training data. 


# Cross Validation (CV)

* Hold out Cross Validation
* k-fold Cross Validation

A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. 
    
In the basic approach, called **k-fold CV**, the training set is split into k smaller sets. The following procedure is followed for each of the k “folds”:
* A model is trained using k-1 of the folds as training data - eg 8/9 folds
* the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
    
The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. 
* Split initial dataset into a separate training and test dataset
* Training dataset - model training
* Test dataset - estimate its generalisation performance

## Holdout Method

 - Split initial dataset into a separate training and test dataset. eg 70/30, 80/20
 - Training dataset - used for model training
 - Test dataset - used to estimate its generalisation performance

A variation is to split the training set to two, training set and validation set.
Training set : For tuning and comparing different parameter setting to further improve the performance for making prediction on unseen data and also for model selection.
This process is called model selection. We want to select to optimal values to tuning parameters (aka hyperparameteres) 

## k-fold Cross Validation
- Randomly split the training dataset into k-folds without replacemet ( without replacemet meaning you dont put data back in when its been used). 
- k-1 folds are used for the model training. 
- The one fold is used for performance evaluation. 

The procedure is repeated k times. 
Final outcomes: - k models and performance estimates. 

- calculate the average performance of the models based on the different, independent folds to obtain a performance estimate that is less sensitive to the sub-partitionaing of the trained data compared to the holdout method.
- k-fold cross-validation is used for model tuning , Finding the optimal hyperparameter values that yields a satisfying generalization performance.
- Once we have found satisfactory hyperparameter values, we can retain the model on the complete training set and obtain a final performance estimate useing the independent test set. The rationale behind fitting a model to the whole training dataset after k-fold cross-validation is that providing more training samples to a learning algorithm usually results in a more accurate and robust model. 

### Stratified k-fold cross-validation
- variation of k-fold that can yield better bias and variance estimates, especially in cases of unequal class proportions.

See The [The scoring parameter: defining model evaluation rules](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) for details. In the case of the Iris dataset, the samples are balanced across target classes hence the accuracy and the F1-score are almost equal.

# 4. CLASSIFICATION 

* Logistic regression 
* Makeing Prediction with Logistic Regression
* Learning with Stochastic Gradient Descent 
* Using Scikit Learn to Estimate Coefficients
* MINST - Famous dataset of hand written digits

* Performance Measures
  * Stratified k-fold
  * Confusion Matrix
  * Precision
  * Recall
  * F1 Score
  * PResision/ recall trade-off
  * ROC 

# Logistic Regression
## Logistic Regression Resources:

[Logistic Regression Tutorial for Machine Learning](http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/)

[Logistic Regression for Machine Learning](http://machinelearningmastery.com/logistic-regression-for-machine-learning/)

[How To Implement Logistic Regression With Stochastic Gradient Descent From Scratch With Python](http://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)

https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

[A comparison of numerical optimizers for logistic regression](https://tminka.github.io/papers/logreg/)

[PDF: A comparison of numerical optimizers for logistic regression](https://tminka.github.io/papers/logreg/minka-logreg.pdf)

Logistic regression is the go-to linear classification algorithm for two-class(binary,yes/no,0/1) problems. It is easy to implement, easy to understand and gets great results on a wide variety of problems, even when the expectations the method has for your data are violated. It is often used in he market place for things like default predictions or fraud detection.

Logistic regression is named for the function used at the core of the method, the [logistic function](https://en.wikipedia.org/wiki/Logistic_function).
The logistic function, also called the **Sigmoid function** was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

$$\frac{1}{1 + e^{-x}}$$

$e$ is the base of the natural logarithms and $x$ is value that you want to transform via the logistic function.

The logistic regression equation has a very similar representation like linear regression. The difference is that the output value being modelled is binary in nature.

$$\hat{y}=\frac{e^{\beta_0+\beta_1x_1}}{1+\beta_0+\beta_1x_1}$$

or

$$\hat{y}=\frac{1.0}{1.0+e^{-\beta_0-\beta_1x_1}}$$

$\beta_0$ is the intecept term

$\beta_1$ is the coefficient for $x_1$

$\hat{y}$ is the predicted output with real value between 0 and 1. To convert this to binary output of 0 or 1, this would either need to be rounded to an integer value or a cutoff point be provided to specify the class segregation point.

To run the predction you would do as follows. You have some data and you have been luckily been given the coefficients. A simple model would be as below. 
```py
coef = [-0.806605464, 0.2573316]
dataset = [[-2.0011, 0],
           [-1.4654, 0],
           [0.0965, 0],
           [1.3881, 0],
           [3.0641, 0],
           [7.6275, 1],
           [5.3324, 1],
           [6.9225, 1],
           [8.6754, 1],
           [7.6737, 1]]

for row in dataset:
    yhat = 1.0 / (1.0 + np.exp(- coef[0] - coef[1] * row[0]))
    print("yhat value {0:.4f}, yhat class {1}".format(yhat, round(yhat)))
```
# Learning the Logistic Regression Model

The coefficients (Beta values b) of the logistic regression algorithm must be estimated from your training data. 

* Generally done using [maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).
* Maximum-likelihood estimation is a common learning algorithm
* Note the underlying assumptions about the distribution of your data
* The best coefficients would result in a model that would predict a value very close to 1 (e.g. male) for the default class and a value very close to 0 (e.g. female) for the other class. 
* The intuition for maximum-likelihood for logistic regression is that a search procedure seeks values for the coefficients (Beta values) that minimize the error in the probabilities predicted by the model to those in the data.

# Learning with Stochastic Gradient Descent
Logistic Regression uses gradient descent to update the coefficients.
Each gradient descent iteration, the coefficients are updated using the equation:
$$ \beta=\beta+\textrm{learning rate}\times (y-\hat{y}) \times \hat{y} \times (1-\hat{y}) \times x $$

# Classification Based Machine Learning Algorithm
[An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction)
This notebook is inspired by Geron [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)

## Scikit-learn Definition:

**Supervised learning**, in which the data comes with additional attributes that we want to predict. This problem can be either:

* **Classification**: samples belong to two or more **classes** and we want to learn from already labeled data how to predict the class of unlabeled data. An example of classification problem would be the handwritten digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories. Another way to think of classification is as a discrete (as opposed to continuous) form of supervised learning where one has a limited number of categories and for each of the n samples provided, one is to try to label them with the correct category or class.

* **Regression**: if the desired output consists of one or more **continuous variables**, then the task is called regression. An example of a regression problem would be the prediction of the length of a salmon as a function of its age and weight.

MNIST dataset - a set of 70,000 small images of digits handwritten. You can read more via [The MNIST Database](http://yann.lecun.com/exdb/mnist/)

# Performance Measures
## Measuring Accuracy Using Cross-Validation
## Stratified-K-Fold

Stratified-K-Fold utilised the Stratified sampling concept

* The population is divided into homogeneous subgroups called strata
* The right number of instances is sampled from each stratum 
* To guarantee that the test set is representative of the population

Bare this in mind when you are dealing with **skewed datasets**. Because of this, accuracy is generally not the preferred performance measure for classifiers.

# Confusion Matrix
|  |  | PREDICTED |  |  |
|-|-|-|-|-|
|  |  | Negative | Positive |  |
| ACTUAL | Negative | true Negative | false Positive |  |
|  | Positive | false Negative | true Positive |  |

```
array([[53360,   717],
       [  395,  5528]])
```
Each row: actual class
Each column: predicted class

First row: Non-zero images, the negative class:
* 53360 were correctly classified as non-zeros. **True negatives**. 
* Remaining 717 were wrongly classified as 0s. **False positive**

Second row: The images of zeros, the positive class:
* 395 were incorrectly classified as 0s. **False negatives**
* 5528 were correctly classified as 0s. **True positives**

## Precision

**Precision** measures the accuracy of positive predictions of your classifier. Also called the `precision` of the classifier. Focus is on the second row of the matrix.

$$\textrm{precision} = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Positives}}$$

## Recall
`Precision` is typically used with `recall` (`Sensitivity` or `True Positive Rate`). The ratio of positive instances that are correctly detected by the classifier. This will be interested in the bottom row of the Confusion Matrix. 

$$\textrm{recall} = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Negatives}}$$

## F1 Score

$F_1$ score is the harmonic mean of precision and recall. Regular mean gives equal weight to all values which we don't always want. Harmonic mean gives more weight to low values.

$$F_1=\frac{2}{\frac{1}{\textrm{precision}}+\frac{1}{\textrm{recall}}}=2\times \frac{\textrm{precision}\times \textrm{recall}}{\textrm{precision}+ \textrm{recall}}=\frac{TP}{TP+\frac{FN+FP}{2}}$$

The $F_1$ score favours classifiers that have similar precision and recall.

## Accuracy 
= (TP+TN)/(TP+TN+FN+FP) 
$$\textrm{Accuracy} = \frac{\textrm{True Positives + True Negatives}}{\textrm{True Positives + False Positives} + \textrm{False Negatives + True Negatives}}$$

# Precision / Recall Tradeoff

Increasing precision will reduce recall and vice versa.


# 5. Support Vector Machines
- Linear Classification
- Polynomial Kernel
- Radial Basis Function(RBF) / Gaussian Kernel - Draws a curve
- Support Vector Regression
- Grid Search 
  - HyperParameter Tuning

Invented in [1963](https://en.wikipedia.org/wiki/Support_vector_machine#History) by [Vladimir N. Vapnik](https://en.wikipedia.org/wiki/Vladimir_Vapnik) and Alexey Ya. Chervonenkis while working at AT&T Bell Labs. Vladimir N. Vapnik joined Facebook AI Research in Nov 2014. In 1992, Bernhard E. Boser, Isabelle M. Guyon and Vladimir N. Vapnik suggested a way to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard incarnation (soft margin) was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.

References:
1. [Support Vector Machine in Javascript Demo by Karpathy](http://cs.stanford.edu/people/karpathy/svmjs/demo/)
2. [SVM](http://www.svms.org/tutorials/)
3. [Statsoft](http://www.statsoft.com/Textbook/Support-Vector-Machines)
4. [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)
5. [Scikit-Learn](http://scikit-learn.org/stable/modules/svm.html)
* [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
  Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
* [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
  C-Support Vector Classification.
  The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than 10000 samples.

## Introduction
**Note:**
* SVM are sensitive to feature scaling so it is very important to scale features before we fit a model.

Supervised learning methods used for classification, regression and outliers detection.
Let's assume we have two classes here - black and purple. In classification, we are interested in the best way to separate the two classes. 
However, there are infinite lines (in 2-dimensional space) or hyperplanes (in 3-dimensional space) that can be used to separate the two classes as the example below illustrates. 

The term hyperplane essentially means it is a subspace of one dimension less than its ambient space. If a space is 3-dimensional then its hyperplanes are the 2-dimensional planes, while if the space is 2-dimensional, its hyperplanes are the 1-dimensional lines. ~ [Wikipedia](https://en.wikipedia.org/wiki/Hyperplane)
In SVM, the **separating line**, the solid brown line, is the line that allows for largest margin between the two classes. 
SVM would place the separating line in the middle of the margin, also called maximum margin. SVM will optimise and locate the hyperplane that maximises the margin of the two classes. The samples that are closest to the hyperplane are called **support vectors**, circled in red. 

## Linear SVM Classification
* Support Vectors
* Separate with a straight line (linearly separable)
* Margin
  * Hard margin classification ( old school) 
      * Strictly based on those that are at the margin between the two classes
      * However, this is sensitive to outliers when Support Vectors cross over that line.
  * Soft margin classification ( new school) 
      * Widen the margin and allows for violation
      * With Python Scikit-Learn, you control the width of the margin, 
      * Control with `C` hyperparameter. (Cost)
        * Smaller `C` -  leads to a wider street but more margin violations
        * High `C` - fewer margin violations but ends up with a smaller margin


# Gaussian Radial Basis Function (rbf)
The kernel function can be any of the following:
* **linear**: $\langle x, x'\rangle$.
* **polynomial**: $(\gamma \langle x, x'\rangle + r)^d$. The degree makes it poly  -.
  $d$ is specified by keyword `degree`
  $r$ by `coef0`.
* **rbf**: $\exp(-\gamma \|x-x'\|^2)$. 
  $\gamma$ is specified by keyword `gamma` must be greater than 0.
* **sigmoid** $(\tanh(\gamma \langle x,x'\rangle + r))$
  where $r$ is specified by `coef0`.  
[scikit-learn documentation](http://scikit-learn.org/stable/modules/svm.html#svm)

## Grid search

Grid search is a way you can search through permutations of hyperparameters in order to validate the best combination of hyperparameters with our model. 
```py
from sklearn.model_selection import train_test_split, GridSearchCV 
pipeline = Pipeline([('clf', svm.SVC(kernel='rbf', C=1, gamma=0.1))]) 
params = {'clf__C':(0.1, 0.5, 1, 2, 5, 10, 20), 
          'clf__gamma':(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1)} 
svm_grid_rbf = GridSearchCV(pipeline, params, n_jobs=-1,
                            cv=3, verbose=1, scoring='accuracy') 

svm_grid_rbf.fit(X_train, y_train) 
# OUTPUT 
Fitting 3 folds for each of 49 candidates, totalling 147 fits
[Parallel(n_jobs=-1)]: Done 147 out of 147 | elapsed:    9.1s finished
GridSearchCV(cv=3, error_score='raise',
       estimator=Pipeline(steps=[('clf', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))]),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'clf__gamma': (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1), 'clf__C': (0.1, 0.5, 1, 2, 5, 10, 20)},
       pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
       scoring='accuracy', verbose=1)
```


## Advantages and Disadvantages of SVM

The **advantages** of support vector machines are:
* Effective in high dimensional spaces.
* Uses only a subset of training points (Support Vectors) in the decision function.
* Many different Kernel functions can be specified for the decision function.
    * Linear
    * Polynomial
    * RBF
    * Sigmoid
    * Custom

The **disadvantages** of support vector machines include:
* Beware of overfitting when num_features > num_samples.
* Choice of Kernel and Regularization can have a large impact on performance
* No probability estimates

## SVM Summary
SciKit learn has three types of SVM:

| Class |  Out-of-core support | Kernel Trick |
| :- |  :- | :- | :- |
| `SGDClassifier` |  Yes | No |
| `LinearSVC` |  No | No |
| `SVC` |  No | Yes |

**Note:** All require features scaling

Support Vector Machine algorithms are not scale invariant(ie; highly dependent on the scale), so it is highly recommended to scale your data. For example, scale each attribute on the input vector X to [0,1] or [-1,+1], or standardize it to have mean 0 and variance 1. Note that the same scaling must be applied to the test vector to obtain meaningful results. See section Preprocessing data for more details on scaling and normalization. ~ [scikit-learn documentation](http://scikit-learn.org/stable/modules/svm.html#svm)

# Where to From Here

* [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
* [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/)
* [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/ch05.html#svm_chapter)
* [Python Data Science Handbook](https://www.safaribooksonline.com/library/view/python-data-science/9781491912126/ch05.html#in-depth-support-vector-machines)
* [Python Machine Learning, 2E](https://www.safaribooksonline.com/library/view/python-machine-learning/9781787125933/ch03s04.html)
* [Statistics for Machine Learning](https://www.safaribooksonline.com/library/view/statistics-for-machine/9781788295758/f2c95085-6676-41c6-876e-ab6802666ea2.xhtml)
* [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/)

**Q: What is Kernel, Loss , gamma, cost in SVM exactly?**

# 7. Decisions Tree

Aka:CART (Classification  and Regression Tree)
* Supervised Learning
* Works for both classification and regression
* Foundation of Random Forests
* Attractive because of interpretability
Decision Tree works by:
* Split based on set impurity criteria
* Stopping criteria (eg; Depth)

Source: [Scikit-Learn](http://scikit-learn.org/stable/modules/tree.html#tree)
Some **advantages** of decision trees are:
* Simple to understand and to interpret. Trees can be visualised.
* Requires little data preparation. Unlike other algorithms, there is not much preprocessing required. 
* Able to handle both numerical and categorical data.
* Possible to validate a model using statistical tests. 
* Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

The **disadvantages** of decision trees include:
* Overfitting. Mechanisms such as pruning (not currently supported), **setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.**
* Decision trees can be unstable. Mitigate: Use decision trees within an ensemble.
* Cannot guarantee to return the globally optimal decision tree. Mitigate: Training multiple trees in an ensemble learner
* Decision tree learners create biased trees if some classes dominate. Recommendation: Balance the dataset prior to fitting

## Decision Tree Learning

* [ID3](https://en.wikipedia.org/wiki/ID3_algorithm) (Iterative Dichotomiser 3)
* [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm) (successor of ID3)
* CART (Classification And Regression Tree)
* [CHAID](http://www.statisticssolutions.com/non-parametric-analysis-chaid/) (Chi-squared Automatic Interaction Detector). by [Gordon Kass](https://en.wikipedia.org/wiki/Chi-square_automatic_interaction_detection). 

## Tree algorithms: ID3, C4.5, C5.0 and CART

* ID3 (Iterative Dichotomiser 3) was developed in 1986 by Ross Quinlan. The algorithm creates a multiway tree, finding for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the ability of the tree to generalise to unseen data.

* C4.5 is the successor to ID3 and removed the restriction that features must be categorical by dynamically defining a discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set of intervals. Ie : it takes all the data and chops it up into n catagories. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules. These accuracy of each rule is then evaluated to determine the order in which they should be applied. Pruning is done by removing a rule’s precondition if the accuracy of the rule improves without it.

* C5.0 is Quinlan’s latest version release under a proprietary license. It uses less memory and builds smaller rulesets than C4.5 while being more accurate.

* CART (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.

* CHAID (Chi-squared Automatic Interaction Detector). by Gordon Kass. Performs multi-level splits when computing classification trees. Non-parametric so it does not require the data to be normally distributed. Often used on Ordinal Data.

scikit-learn uses an optimised version of the CART algorithm.

## Gini Impurity

scikit-learn default

[Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

If all data are of the same class then the gini will be 0.0. If 25% are of one class, that gini will be 0.25. The largest gini value will be under the root node and as you split , the gini will get smaller. 0 is pure, 1 is full diversity.

A measure of purity / variability of categorical data. As a side note on the difference between [Gini Impurity and Gini Coefficient](https://datascience.stackexchange.com/questions/1095/gini-coefficient-vs-gini-impurity-decision-trees). 

* No, despite their names they are not equivalent or even that similar.
* **Gini impurity** is a measure of misclassification, which applies in a multiclass classifier context.
* **Gini coefficient** applies to binary classification and requires a classifier that can in some way rank examples according to the likelihood of being in a positive class.
* Both could be applied in some cases, but they are different measures for different things. Impurity is what is commonly used in decision trees.

Developed by [Corrado Gini](https://en.wikipedia.org/wiki/Corrado_Gini) in 1912.

Key Points:
* A pure node (homogeneous contents or samples with the same class) will have a Gini coefficient of zero
* As the variation increases (heterogeneneous classes or increase diversity), Gini coefficient increases and approaches 1.

$$Gini=1-\sum^r_j p^2_j$$

$p$ is the probability (often based on the frequency table)

## Entropy
[Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory))
The entropy can explicitly be written as
$${\displaystyle \mathrm {H} (X)=\sum _{i=1}^{n}{\mathrm {P} (x_{i})\,\mathrm {I} (x_{i})}=-\sum _{i=1}^{n}{\mathrm {P} (x_{i})\log _{b}\mathrm {P} (x_{i})},}$$
where `b` is the base of the logarithm used. Common values of `b` are 2, Euler's number `e`, and 10.

Note: The probability of the individual observation multiplied by the log of the individual observation

## Which should I use? Entropy or Gini
[Sebastian Raschka](https://sebastianraschka.com/faq/docs/decision-tree-binary.html)
* They tend to generate similar tree
* Gini tends to be faster to compute

## Information Gain
* Expected reduction in entropy caused by splitting 
* Keep splitting until you obtain a as close to homogeneous class as possible


## Trees - Where to From Here
### Tips on practical use
* Decision trees tend to overfit on data with a large number of features. Check ratio of samples to number of features
* Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand
* Visualise your tree as you are training by using the export function. Use max_depth=3 as an initial tree depth.
* Use max_depth to control the size of the tree to prevent overfitting.
* Tune `min_samples_split` or `min_samples_leaf` to control the number of samples at a leaf node. 
* Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. 
  * By sampling an equal number of samples from each class
  * By normalizing the sum of the sample weights (sample_weight) for each class to the same value. 

# References:
1. [Wikipedia - Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
2. [Decision Tree - Classification](http://www.saedsayad.com/decision_tree.htm)
3. [Data Aspirant](http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/)
4. [Scikit-learn](http://scikit-learn.org/stable/modules/tree.html)
5. https://en.wikipedia.org/wiki/Predictive_analytics
6. L. Breiman, J. Friedman, R. Olshen, and C. Stone. Classification and Regression Trees. Wadsworth, Belmont, CA, 1984.
7. J.R. Quinlan. C4. 5: programs for machine learning. Morgan Kaufmann, 1993.
8. T. Hastie, R. Tibshirani and J. Friedman. Elements of Statistical Learning, Springer, 2009.


# 8. Ensemble Methods: Combineing models. 

**Note: Ensemble methods** It is still supervised.
* Work best with indepedent predictors
* Best to utilise different algorithms


### **B**ootstrap **Agg**regat**ing** or [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
* [Scikit- Learn Reference](http://scikit-learn.org/stable/modules/ensemble.html#bagging)
* Bootstrap sampling: Sampling with replacement - (put the sample back in the pool to possibly take the same one again.)
* Combine by averaging the output (regression)
* Combine by voting (classification) - 
* Can be applied to many classifiers which includes ANN(Artificial Neural Network), CART, etc.


### [Pasting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html))
* Similar to bagging apart from **Sampling without replacement**.

### [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning))
You start with an equal weight and then once you have iterated, you increase the weighting of the weak classifiers and go round again. It iis seriel and learns fr omthe past becasue you have to wait for the previous iteration to finish. So it can take longer. 
* Train weak classifiers.
* Add them to a final strong classifier by weighting. Weighting by accuracy (typically)
* Once added, the data are reweighted
  * **Misclassified** samples **gain weight** 
  * **Correctly** classified samples **lose weight** (Exception: Boost by majority and BrownBoost - decrease the weight of repeatedly misclassified examples). 
  * Algo are forced to learn more from misclassified samples
  
    
### [Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
* Also known as Stacked generalization
* [From Kaggle:](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) Combine information from multiple predictive models to generate a new model. Often times the stacked model (also called 2nd-level model) will outperform each of the individual models due its smoothing nature and ability to highlight each base model where it performs best and discredit each base model where it performs poorly. For this reason, stacking is most effective when the base models are significantly different. 
* Training a learning algorithm to combine the predictions of several other learning algorithms. 
  * Step 1: Train learning algo
  * Step 2: Combiner algo is trained using algo predictions from step 1.  
  
### Other Ensemble Methods:

[Wikipedia](https://en.wikipedia.org/wiki/Ensemble_learning)
* Bayes optimal classifier
  * An ensemble of all the hypotheses in the hypothesis space. 
  * Each hypothesis is given a vote proportional to the likelihood that the training dataset would be sampled from a system if that hypothesis were true. 
  * To facilitate training data of finite size, the vote of each hypothesis is also multiplied by the prior probability of that hypothesis. 
* Bayesian parameter averaging
  * an ensemble technique that seeks to approximate the Bayes Optimal Classifier by sampling hypotheses from the hypothesis space, and combining them using Bayes' law.
  * Unlike the Bayes optimal classifier, Bayesian model averaging (BMA) can be practically implemented. 
  * Hypotheses are typically sampled using a Monte Carlo sampling technique such as MCMC. 
* Bayesian model combination
  * Instead of sampling each model in the ensemble individually, it samples from the space of possible ensembles (with model weightings drawn randomly from a Dirichlet distribution having uniform parameters). 
  * This modification overcomes the tendency of BMA to converge toward giving all of the weight to a single model. 
  * Although BMC is somewhat more computationally expensive than BMA, it tends to yield dramatically better results. The results from BMC have been shown to be better on average (with statistical significance) than BMA, and bagging.
* Bucket of models
  * An ensemble technique in which a model selection algorithm is used to choose the best model for each problem. 
  * When tested with only one problem, a bucket of models can produce no better results than the best model in the set, but when evaluated across many problems, it will typically produce much better results, on average, than any model in the set.

R released
* BMS (an acronym for Bayesian Model Selection) package
* BAS (an acronym for Bayesian Adaptive Sampling) package
* BMA package

## Random Forest

[Original paper of Random Forest](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)

* Random Forest is basically an Ensemble of Decision Trees

* Training via the bagging method (Repeated sampling with replacement)
  * Bagging: Sample from samples - 
  * RF: Sample from predictors. $m=sqrt(p)$ for classification and $m=p/3$ for regression problems.

* Utilise uncorrelated trees


Random Forest
* Sample both **observations and features** of training data.
It will ignore other feature when it is doing sampleing. 

Bagging
* Samples **only observations at random**
* Decision Tree select best feature when splitting a node. It will always go for the best feature to split a node.
* Focus on the training data and leave the features by inclueing them all??. 

Running on the Titanic Dataset we didnt get doo results. Probably because the dataset was too small.

## Extra-Trees (Extremely Randomized Trees) Ensemble

[scikit-learn](http://scikit-learn.org/stable/modules/ensemble.html#bagging)

* Random Forest is build upon Decision Tree
* Decision Tree node splitting is based on gini or entropy or some other algorithms
* Extra-Trees make use of random thresholds for each feature unlike Decision Tree

When it comes to Random Forest, the sample is drawn by bootstrap sampleing. The splitting of node is used to construct the tree, but on the second time it will randomly sample different features rather than the previously best fetures. This adds an extra source of randomness. This does increase the bias slightly but becasue of averageing , the variace decreases. Typically this impves the SD and Variance at the slight sacrifice of bias .

Caompareing with  Extremely Randomized Trees the randomness is introduced in the way the note splittings are computed. In random forest, a random subset of the features are used but in Extremely Randomized Trees the threashold is randonly generated which introdues another layer of variance. 


# Boosting (Aka Hypothesis Boosting)
* Combine several weak learners into a strong learner. 
* Train predictors sequentially

# AdaBoost / Adaptive Boosting

Getting on the cutting edge.
[Creator: Robert Schapire](http://rob.schapire.net/papers/explaining-adaboost.pdf)
[Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
[Chris McCormick](http://mccormickml.com/2013/12/13/adaboost-tutorial/)
[Scikit Learn AdaBoost](http://scikit-learn.org/stable/modules/ensemble.html#adaboost)

1995

As above for Boosting:
* Similar to human learning, the algo learns from past mistakes by focusing more on difficult problems it did not get right in prior learning. You make a mistake, learn from them, spend more time on them. 
* In machine learning speak, **it pays more attention to training instances that previously underfitted.**

### PsuedoCode
Source: Scikit-Learn:

* Fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. 
* The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.
* Then, The data modifications at each so-called boosting iteration consist of applying weights $w_1, w_2, …, w_N$ to each of the training samples. 
* Initially, these weights are all set to $w_i = 1/N$, so that the first step simply trains a weak learner on the original data. 
* For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data. Here theweights are shifted.
* At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly. 
* **As iterations proceed, examples that are difficult to predict receive ever-increasing influence.** Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.

## Adaboost basic params
```
{'algorithm': 'SAMME.R',
 'base_estimator': None,
 'learning_rate': 1.0,
 'n_estimators': 50,
 'random_state': None}
 ```
[SAMME16](https://web.stanford.edu/~hastie/Papers/samme.pdf) (Stagewise Additive Modeling using a Multiclass Exponential loss function).

R stands for real


# Gradient Boosting / Gradient Boosting Machine (GBM)

Works for both regression and classification. IT learns from the mistakes and continuously tries to improve from there.

[Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)

* Sequentially adding predictors, so a little differnt from AdaBoost. These are added bit by bit. Not all predictors are being exposed.
* Each one correcting its predecessor
* Fit new predictor to the residual errors.

Compare this to AdaBoost: 
* Alter instance weights at every iteration. Meaning the area of mistakes increases the weight. 

So you apply a model () predictors to predict the target outcome) which will explain some of the variance. The residuals which are left and are not explained, you apply another predictor to it until you get to the final predictors. 

**Step 1.** Basically simple linear regression
  $$Y = F(x) + \epsilon$$
**Step 2.** With the error term ($\epsilon$) It is optimised and run it through another predictor $G(x_2)$ , rather than starting from scratch. 
  $$\epsilon = G(x_2) + \epsilon_2$$
  Substituting step (2) into step (1), we get :  
  $$Y = F(x) + G(x) + \epsilon_2$$    
**Step 3.** We create another function $H(x)$
  $$\epsilon_2 = H(x)  + \epsilon_3$$
We can keep on continuesin until we have used all of the predictors:  
  $$Y = F(x) + G(x) + H(x)  + \epsilon_3$$  
Finally, by adding weighting eg with $\alpha,  \beta , \gamma$ alpha,beta and gamma.   
  $$Y = \alpha F(x) + \beta G(x) + \gamma H(x)  + \epsilon_4$$

The key thing with Gradient boosting is that it involves three elements:

* **Loss function to be optimized**: Loss function depends on the type of problem being solved. In the case of regression problems, mean squared error is used, and in classification problems, logarithmic loss will be used. In boosting, at each stage, unexplained loss from prior iterations will be optimized rather than starting from scratch.

* **Weak learner to make predictions**: Decision trees are used as a weak learner in gradient boosting. 

* **Additive model to add weak learners to minimize the loss function**: Trees are added one at a time and existing trees in the model are not changed. The gradient descent procedure is used to minimize the loss when adding trees, hence the term gradient.

# XGBoost (Extreme Gradient Boosting)
...it might be more suitable to be called as **regularized gradient boosting....**

[Documentation](http://xgboost.readthedocs.io/en/latest/)
[tqchen github](https://github.com/tqchen/xgboost/tree/master/demo/guide-python)
[dmlc github](https://github.com/dmlc/xgboost)
* “Gradient Boosting” is proposed in the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman. 
* XGBoost is based on this original model. 
* Supervised Learning

Like Gradient Boosting Machines (GBM)
* Sequentially adding predictors, so a little differnt from AdaBoost. These are added bit by bit. Not all predictors are being exposed.
* Each one correcting its predecessor
* Fit new predictor to the residual errors.

## Objective Function : Training Loss + Regularization
$$Obj(Θ)=L(θ)+Ω(Θ)$$
* $L$ is the training loss function. A measure of how predictive our model is on the training data. 
* $Ω$ is the regularization term. The complexity of the model, which helps us to inform and avoid overfitting.

### Training Loss
The training loss measures how predictive our model is on training data.
Example 1, Mean Squared Error for Linear Regression:
$$L(θ)= \sum_i(y_i-\hat{y}_i)^2$$
Example 2, Logistic Loss for Logistic Regression:

$$ L(θ) = \sum_i \large[ y_i ln(1 + e^{-\hat{y}_i}) + (1-y_i) ln(1 + e^{\hat{y}_i}) \large] $$

Hoever when we only focus on this objective function. Enter the Regularization Term.

### Regularization Term
What sets XGBoost appart from GBM's is this regularisation term. 
The regularization term controls the complexity of the model, which helps us to avoid overfitting. 
[XGBoost vs GBM](https://www.quora.com/What-is-the-difference-between-the-R-gbm-gradient-boosting-machine-and-xgboost-extreme-gradient-boosting/answer/Tianqi-Chen-1)
* Specifically,  xgboost used a more regularized model formalization to control over-fitting, which gives it better performance.
* For model, it might be more suitable to be called as regularized gradient boosting.

# 9. Ensemble of ensembles - model stacking
* **Ensemble with different types of classifiers**: 
  * Different types of classifiers (E.g., logistic regression, decision trees, random forest, etc.) are fitted on the same training data
  * Results are combined based on either 
    * majority voting (classification) or 
    * average (regression)
  

* **Ensemble with a single type of classifier**: 
  * Bootstrap samples are drawn from training data 
  * With each bootstrap sample, seperate models (E.g., Individual model may be decision trees, random forest, etc.) will be fitted 
  * All the results are combined to create an ensemble. 
  * Suitabe for highly flexible models that are prone to overfitting / high variance. 

***

## Combining Method - 3 different way of combining outcomes

Meta Classifier means layering a classifier ontop of another classifier. Eg; Logistic regression ontop of a base layer of your classifier. 

* **Majority voting or average**: 
  * Classification: Largest number of votes (mode) 
  * Regression problems: Average (mean).
  
  
* **Method of application of meta-classifiers on outcomes**: 
  * Binary outcomes: 0 / 1 from individual classifiers
  * Meta-classifier is applied on top of the individual classifiers. 
  
  
* **Method of application of meta-classifiers on probabilities**: 
  * Probabilities are obtained from individual classifiers. 
  * Applying meta-classifier
  

# 10. k-Nearest Neighbor (KNN)

[wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

It is both a Classification and Regression based supervised algorithm. K is how many neighbours do you want your data sample in the space to be compared to it, in order to clasify it?  
It is a very simple classifier to impliment that you can use quickly and get a certain amount of accuracy with. 

1. Lazy learner as it is [Instance Based](https://en.wikipedia.org/wiki/Instance-based_learning)
  * It memorise the pattern from the dataset
  * Lazy because it does not try to learn a function from the training data. 
  
2. It is a [Nonparametric model](http://blog.minitab.com/blog/adventures-in-statistics-2/choosing-between-a-nonparametric-test-and-a-parametric-test)
  * distribution-free tests because no assumption of the data needing to follow a specific distribution. Eg Normal distibution etc.
  * [wikipedia](https://en.wikipedia.org/wiki/Nonparametric_statistics)
  * Other examples - Decision Tree, Random Forest


Used for:
* Predict cancer is malignant or benign
* Pattern recognition
* Recommender Systems
* Computer Vision
* Gene Expression
* Protein-Protein Interaction and 3D Structure Prediction

## Disadvantages

* Not efficient on big data, this is becasue it is mempry based.
* Very susceptible to overfitting from the **Curse of dimensionality.** The Curse of dimensionality describes the phenomenon where the feature space becomes increasingly sparse for an increasing number of dimensions of a fixed-size training dataset. Intuitivly, we can think of even the closest neighbours being too far away in a high-dimensional space to give a good esitmate.

References:

* [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* [Scikit-Learn Nearest Neighbours](http://scikit-learn.org/stable/modules/neighbors.html)
* [Introduction to k-nearest neighbors : Simplified](https://www.analyticsvidhya.com/blog/2014/10/introduction-k-neighbours-algorithm-clustering/)
* [Quora](https://www.quora.com/What-are-industry-applications-of-the-K-nearest-neighbor-algorithm)
* [Kevin Zakka](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)
* [CS231n CNN](http://cs231n.github.io/classification/#nn)
* [A Detailed Introduction to K-Nearest Neighbor (KNN) Algorithm](https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/)
* [Chris Albon](https://chrisalbon.com/)
* [K-Nearest Neighbors for Machine Learning](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)
* [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php)

## Steps:
* Choose the number of $k$ , the neighbors.
* Select a distance metric
* Find the k nearest neighbors of the sample
* Assign the class label by majority vote

## DistanceMetric class documentation
[scikit-learn]()

Metrics intended for real-valued vector spaces:

$|x+y| \leq |x| + |y|$

| identifier	| class name	| args	| distance function |
|:- |:- |:- |:- |
|"euclidean" | 	EuclideanDistance  |   | $\sqrt{\sum(x - y)^2)}$  |  
|"manhattan" | ManhattanDistance |  | $\sum (x - y)   | | 
|"chebyshev" | ChebyshevDistance |  | max${\big|x - y\big|}$ |  
|"minkowski" | MinkowskiDistance	 | p	 | $\sum(\big|x - y\big|^p)^{\frac{1}{p}}$     | 
|"wminkowski" | WMinkowskiDistance	 | p, w	 | $\sum(w\big|x - y\big|^p)^{\frac{1}{p}}$     |
|"seuclidean" | SEuclideanDistance	 | V	 | $\sqrt{\sum\frac{(x - y)^2}{V})}$     | 



Refer to documentation for more on 
* Metrics intended for two-dimensional vector spaces
* Metrics intended for integer-valued vector spaces
* Metrics intended for boolean-valued vector spaces

* User-defined distance

Source: [Rorasa's blog](https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)

* Mathematically a norm is a total size or length of all vectors in a vector space or matrices. 
* For simplicity, we can say that the higher the norm is, the bigger the (value in) matrix or vector is. 
* Norm may come in many forms and many names, including these popular name: Euclidean distance, Mean-squared Error, etc.
* Most of the time you will see the norm appears in a equation like this:

$\left \| x \right \|$ where $x$ can be a vector or a matrix.

* Euclidean distance - Strightline connecting two points
  * Most common
  * The Euclidean distance between points (1,2) and (3,3) can be computed $\sqrt{(1-3)^2+(2-3)^2}$, which results in a distance of about 2.236.
  * L2 norm of two vectors. 
  * In a bidimensional plane, the Euclidean distance refigures as the straight line connecting two points, and you calculate it as the square root of the sum of the squared difference between the elements of two vectors. 
  
* Manhattan distance ( TAxi cab, city block) 
  * Another useful measure is the Manhattan distance
  * For instance, the Manhattan distance between points (1,2) and (3,3) is abs(1–3) and abs(2–3), which results in 3.
  * L1 norm of two vectors
  * Summing the absolute value of the difference between the elements of the vectors. 
  * If the Euclidean distance marks the shortest route, **the Manhattan distance marks the longest route**, resembling the directions of a taxi moving in a city. (The distance is also known as taxicab or city-block distance.) 

* Chebyshev distance
  * Takes the maximum of the absolute difference between the elements of the vectors. 
  * In the example used in previous sections, the distance is simply 2, the max between abs(1–3) and abs(2–3).
  * It is a distance measure that can represent how a king moves in the game of chess or, in warehouse logistics, the operations required by an overhead crane to move a crate from one place to another. 
  * In machine learning, the Chebyshev distance can prove useful when you have many dimensions to consider and most of them are just irrelevant or redundant (in Chebyshev, you just pick the one whose absolute difference is the largest). 
  
### P - argument ( Where is minkowski ? )
- minkowski uses with a P value = $1/p$.
- minkowski with a P value = 1 is the same as the distance as **manhattan**.
- minkowski with a P value = 2 is the same as the distance as **euclidean**
- minkowski with a P value = $\infty$ is the same as the distance as **chebyshev**
  


# 11. Dimensionality Reduction
Trying to discover a pattern, consistency and relationship with the data itself. 
**Why?** - Often we are faced with large data sets with many features. 200-1000 features.
 - Really hard to train an ML algorithm with so many features
 - It takes a very long time to train
So we do some DR before we start to train our model. Both these methods will preserve the key Features/Components so there is not much contens lost.

## Comparison between LDA and PCA
**PCA** - maximiseing the componant axes for **varience in the data itself.** 
  - Good for data without classes.
**LDA** - maximiseing the componant axes for **varience between class sepreation.**
  - Good for data with classes in mind
[Scikit-Learn](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)

* Principal Component Analysis (PCA) 
  * Identifies the combination of attributes (principal components) that account for the most variance in the data. 
  
* Linear Discriminant Analysis (LDA) tries to identify attributes ( features, predictors) that account for the most variance between classes. 
  * LDA is a supervised method, using known class labels.
 

## Principal Component Analysis (PCA)
1901 by Karl Pearson (Also known for "Pearson Correlation")
Used in exploratory data analysis (EDA). 

* Unsupervised Machine Learning
[scikit-learn Doc](http://scikit-learn.org/stable/modules/decomposition.html#pca)
[scikit-learn Parameters](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
  * Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
  * [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
  * Statistical procedure that utilise [orthogonal transformation](https://en.wikipedia.org/wiki/Orthogonal_transformation) technology
  * Convert possible correlated features (predictors) into linearly uncorrelated features (predictors) called **principal components**
  * \# of principal components <= number of features (predictors)
  * First principal component explains the largest possible variance
  * Each subsequent component has the next highest variance, subject to the restriction: that it must be orthogonal to the preceding components. 
    * Orthogonal - of or involving right angles; at right angles. In geometry, two Euclidean vectors are orthogonal if they are perpendicular.
  * A collection of the components are called vectors. EigenVectors + EigenValues = EigenPairs
  * Sensitive to scaling
  * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html): Component axes that maximise the variance
  * The transformation from data axes to principal axes is as an affine transformation, which basically means it is composed of a translation, rotation, and uniform scaling.
**Note:**
* Used in exploratory data analysis (EDA) 
* Visualize genetic distance and relatedness between populations. How Correlated are they?

* Method:
  * Eigenvalue decomposition of a data covariance (or correlation) matrix
  * Singular value decomposition of a data matrix (After mean centering / normalizing ) the data matrix for each attribute.
* Output
  * Component scores, sometimes called **factor scores** (Aka The transformed variable values, Principle Components), **loadings** (the weight)

- Other uses of PCA
  * Data compression and information preservation 
  * Visualization
  * Noise filtering
  * Feature extraction and engineering


## Kernel PCA
* Non-linear dimensionality reduction through the use of kernels

[Scikit Learn Documentation](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)

[Scikit Learn Reference](http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html)

This is inspired by this [scikit notebook by Mathieu Blondel and Andreas Mueller](http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py)

## Linear Discriminant Analysis (LDA) 
  * [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
  * Most commonly used as dimensionality reduction technique in the pre-processing step for pattern-classification and machine learning applications. 
  * Goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs.
  * Locate the 'boundaries' around clusters of classes.  
  * Projects data points on a line. Trying ot keep these as aseperate as possible. 
  * A centroid will be allocated to each cluster or have a centroid nearby.
  * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html): Maximising the component axes for class-separation
* Supervised dimensionality reduction
* Project the input data to a linear subspace consisting of the directions which maximize the separation between classes. Takes into the account the classes using the training data.
* Most useful in a multiclass setting.
* Commonly used in Finance

[scikit learn](http://scikit-learn.org/stable/modules/lda_qda.html#lda-qda)

[scikit learn](http://scikit-learn.org/stable/modules/lda_qda.html#lda-qda)

### Other Dimensionality Reduction Techniques
* [Multidimensional Scaling (MDS) ](http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds)
  * Seeks a low-dimensional representation of the data in which the distances respect well the distances in the original high-dimensional space.
* [Isomap (Isometric Mapping)](http://scikit-learn.org/stable/modules/manifold.html#isomap)
  * Seeks a lower-dimensional embedding which maintains geodesic distances between all points.
* [t-distributed Stochastic Neighbor Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
  * Nonlinear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. 
  * Models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points. dimensional space (e.g., to visualize the MNIST images in 2D).

### Matrix Multiplication revision:
$$A=\begin{bmatrix} 1. & 2. \\ 10. & 20. \end{bmatrix}$$

$$B=\begin{bmatrix} 1. & 2. \\ 100. & 200. \end{bmatrix}$$

$$A \times B  = \begin{bmatrix} 1. & 2. \\ 10. & 20. \end{bmatrix}

\times \begin{bmatrix} 1. & 2. \\ 100. & 200. \end{bmatrix} = \begin{bmatrix} 201. & 402. \\ 2010. & 4020. \end{bmatrix} $$

By parts:
$$A \times B = \begin{bmatrix} 1. \times 1. + 2.  \times 100. &  1. \times 2. + 2. \times 200. \\ 
1.  \times 1. + 20. \times 100. & 10. \times 2. + 20. \times 200. \end{bmatrix}$$
$$A \times B  = \begin{bmatrix} a & b \\ c & d \end{bmatrix}

\times \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} (a\times1)+ (1\times3) & (a\times1)+ (2\times4) \\ (c\times d)+ (1\times3) & (c\times d)+ (2\times4) \end{bmatrix} $$

In python 
```py
import numpy as np
A = [[1., 2.], [10., 20.]]
B = [[1., 2.], [100., 200.]]
np.dot(A, B)
```
Returns 
```py
array([[ 201.,  402.],
       [2010., 4020.]])
```


# 12 . Unsupervised Learning Cluster. 
[Wikipedia](https://en.wikipedia.org/wiki/Cluster_analysis)

* The task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters). 

* With Supervised learning we do not have a target that we can learn from. With Supervised,  There are features that go along with the class. ML is powerful here as it uncovers patterns in the data.

Examples where its used that have no target:
* Natural Language Processing (NLP) - Getting the sentiment.
* Computer Vision
* Stock markets
* Customer / Market Segmentation
* Customer Churn
* fraud detection

## how do you cluster? 
### 4 of the many types :

### Connectivity-based clustering
* Distance based. How close are they to each other
* E.g., Hierarchical clustering - based on an object related to another object that is close by. 
 * if you live in this neighbourhood you are more like to be like the people here than in another neighborhood. 
* Distances will be represented by Dendrogram

### Centroid-based clustering
* Represents each cluster by a single mean vector. Trying to find the average in a cluster.
* E.g., k-means Clustering algorithm
* 1 catch with k-means: You do need to specify the number of clusters. So is it really supervised? 

### Distribution-based clustering
* Modeled using statistical distributions
* E.g., Multivariate normal distributions used by the expectation-maximization algorithm.

### Density-based clustering
* Defines clusters as connected dense regions in the data space.
* E.g., DBSCAN

[MLXTEND](http://rasbt.github.io/mlxtend/) is an extra python library supported by Sebastien Raschka. 

# Ward’s Agglomerative Hierarchical Clustering
Agglomeration: a large group of many different things collected or brought together: 

[Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_clustering)

Wards' can work in two different ways:
* Agglomerative: 
  * Bottom up
  * Each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.

* Divisive: 
  * Top down
  * All observations start in one cluster, and splits ( like a Decision Tree) are performed recursively as one moves down the hierarchy.

So how does it actually do it ? IT uses a distance matrix. Similar (but not exactly the same) to K NEarest Neighbour.
- euclidean, Manahattan, Mahalanobis

  
[Stackexchange](https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering)
[CMU Notes](http://www.stat.cmu.edu/~ryantibs/datamining/lectures/05-clus2-marked.pdf)
[PSE Stat505 Linkage Methods](https://onlinecourses.science.psu.edu/stat505/node/143):

Linkage Criteria 

1. Single Linkage: shortest distance. Distance between two clusters to be the **minimum distance between any single data point in the first cluster and any single data point in the second cluster**. 

2. Complete Linkage: Furthest distance. Distance between two clusters to be the **maximum distance between any single data point in the first cluster and any single data point in the second cluster**.

3. Average Linkage: Average of all pairwise links.

4. Centroid Method: Distance between two clusters is the **distance between the two mean vectors of the clusters**. These may not exist in the data but are calculated.

5. Ward’s Method: ANOVA based approach. 
    * Iterative process
    * Minimises the total within cluster variance. So it will calculate within a clusters variance. 
    * At each step, the pair of clusters with minimum between cluster distance are merged

### Retrieve the Clusters
By Retrieve we mean labeling the observations with the class somehow. The scipy library fcluster is the recomendation. 

* Utilise the [fcluster](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html) function.
* Retrieve by distance or number of clusters

***

Becasue we are wrking with data that has no target or label, we beed multiple methods to confirm eahc other, and investigate those that say otherwise. If so then we may need to do further explorations.

# k-Means Clustering
* Unsupervised method that Analyses and find patterns / clusters within data
* Distance measures

[scikit learn](http://scikit-learn.org/stable/modules/clustering.html#k-means)

* Clusters data by trying to separate samples in n groups of equal variance eg $Sd^2$ 
* Minimizing a criterion known as the "inertia" or "within-cluster sum-of-squares". Trying to minimize the seperate groups of equal variance.
* Requires the number of clusters to be specified. 
* Scales well

How does it work?
* Divides a set of samples into disjoint clusters. Before hand you neeed to give it a target of clusters to work towards.
* Each described by the mean of the samples in the cluster. 
* The means are commonly called the cluster “centroids”
* Note that the centroids are not, in general, points from, although they live in the same space. 
* The K-means algorithm aims to choose centroids that minimise the inertia, or **within-cluster sum of squared criterion**. Once it has identified the clusters, it tries to find a point which minimises the Sums of Square( deviation from the centroid; like Linear regression)

Some Challenges k-Means Clustering :
* The globally optimal result may not be achieved
* The number of clusters must be selected beforehand
* k-means is limited to linear cluster boundaries. To resolve this you could use PCA to get down to 2 PCA
* k-means can be slow for large numbers of samples.

## Elbow Method

Becasue we are workign with an Unsupervised methods we don't have an answer , a Y. 
* Use intrinsic metrics - another name for "within-cluster Sums of Squared Error"
* An example fo this is the **within-cluster Sums of Squared Error** 
* scikit learn has already provided it via the kmeans `inertia_` attribute. So it will do the heavy lifting.

When we plot this on a line graph, there will be an elbow where the line bends sharply. This indicates where we can stop clustering because the **within-cluster Sums of Squared Error** is not giving us any more reduction.

# Silhouette Analysis Method
This is an alternative to the Elbow method. 

$$\text{silhouette score}=\frac{p-q}{max(p,q)}$$

$p$ is the mean distance to the points in the nearest cluster that the data point is not a part of.The middle of one centroid to another.

$q$ is the mean intra-cluster distance to all the points in its own cluster. Intra-cluster means within the cluster, so the mean distance of all the distances within one cluster.

So $p-q$ will contrast the `p` inter cluster and the `q` intra-cluster. You want to minimsie p and maximise q so the top value of the fraction is as big as possible.
The $max(p,q)$ is to standardise it and get it closest to 1. 

* The value of the silhouette score range lies between -1 to 1. 
* A score closer to 1 indicates that the data point is very similar to other data points in the cluster. 
* A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

example 
```py
from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k).fit(X)
    sse_.append([k, silhouette_score(X, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);
```
When you plot the silhouette scores for different kluster sizes, it will give you arch that presents a kind of ranking. This is more useful than the elbow method. 

****

## Mean Shift


[wikipedia](https://en.wikipedia.org/wiki/Mean_shift)

* The basic intuition is [Non-parametric](https://en.wikipedia.org/wiki/Nonparametric_statistics)

* Identify centroids location

  * For each data point, it identifies a window around it
  * it then Computes centroid
  * Updates centroid location
  * Continue to update windows
  * Keep shifting the centroids, the means, towards the peaks of each cluster. Hence the term **Means Shift**
  * Continues until centroids no longer move

* Often used for object tracking

This Algorithm has a good ability to identify and locate the centroid of clusters.
************************
# NOTES 
************************

Read up on  Linear Discriminant Analysis (LDA) 
 * [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
 * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html)


### Q: What is reinforcement learning? 
Its neither supervised ( not given labeled data, not left by itself either) 
- given ana agent to learn what the utility function. the agent(system( learn the action function and the value function.
Learning  To every action there is areaction

- **NOTE**: Data science requires some understanding of the domain that your modeling in. 


Some toolkits in which these algorithms are available: 
 Python: 
  NLTK: https://www.nltk.org/ 
 Scikit Learn: https://scikit-learn.org/stable/ 

Steps of a project

(EDA) Exploritory Data Analysis
PReprocessing 
Fit Model
Cross validation - to get an average accuracy

### Regressions  
Q: What is the RANdom SAmple Consensus (RANSAC) Algorithm exactly?

Q. What are ensemble methods?
Lectures on Natural Language understanding. - https://www.youtube.com/watch?v=tZ_Jrc_nRJY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20
 
(stats without the tears : symbols)[https://brownmath.com/swt/symbol.htm]

q: is an estimator another name for model?
Q: What is, RSS, SSE,  R ^2 exactly?
Q: What is SGDClassifier exactly? Stochastic Gradient Decent
Q: What is a Kernel functions re SVM?
[Orange data mining visulisation software](https://orange.biolab.si/) - Looks really good 