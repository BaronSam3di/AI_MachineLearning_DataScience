  
#  Data Science NOTES from Ng Course - Oreilly
  
  
There are a lot of statistical tests and information. Mostly for the purpose of statistical analysis. You do not need all of these for data science.
  
Data science focus is on prediction and having models that work on predicting real data. It is not concerned as much with correct specifications of statistical problems.
  
#  2 . Regresssion
  
  
**Regressions**
- Robust reggression 
- Multiple regression with statsmodels, Multiple Regression and Feature Importance
- OLS and Gradient Descent 
- Regularized Regression( restrict overfitting) - Ridge, Lasso, Elastic Net
- Polynomial Regression. FOr models that are not Linear
  
**Performance Evaluation**
- How does it perform when it is put of sample? 
  
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
  
  
###  Regression
  
  
Regression deals with Continuous variables. You use the explanatory variable to predict the outcome.
x = is the predictor/explanatory variable/independent variable
Y = dependent variable/outcome
eg use crime rate and room size to predict house value
  
###  Reinforcement learning
  
Trying to develop a system or agent that improves performance based on feedback or interactions with the environment. This was used by Google on the game of GO.
Its neither supervised ( not given labeled data, not left by itself either) 
- given ana agent to learn what the utility function. the agent(system( learn the action function and the value function.
Learning ..to every action there is a reaction. Does it get a reward or an punishment.
  
JPMorgan has a reinforcment agent on the stock market. Apric
  
###  Unsupervised 
  
Exploratory data analysis. Feed raw data and the algorithm will cluster to meaningful subgroups.
  
##  Supervised Learning example: Simple Linear Regression
  
###  When x is _, y is _
  
Linear Regression aims to establish if there is a **statistically significant** relationship between two variables. Eg Income and spending, location and price, sugar and health. We may also want to **forcast** new observations. IT is called Linear because if we plot it on a bi-dimensional plot it will create a straight line.
  
- Dependent variable: the value we want to forecast/explain. Denoted as `Y`.
- Independent variable: the value that explains Dependent. Denoted as `X`.
- Errors/Difference between the real data points and the Line of the regression model represented as <img src="https://latex.codecogs.com/gif.latex?&#x5C;epsilon"/> 
  
The Linear equation model: <img src="https://latex.codecogs.com/gif.latex?y"/> = <img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_0"/> + <img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_1"/> <img src="https://latex.codecogs.com/gif.latex?x"/> + <img src="https://latex.codecogs.com/gif.latex?&#x5C;epsilon"/>
  
Where 
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_0"/> the constant/intercept of x
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_1"/> the coeficient/slope of x
- <img src="https://latex.codecogs.com/gif.latex?&#x5C;epsilon"/> is the error term we are trying to minimise
  
[More Theory on Youtube](https://www.youtube.com/watch?v=owI7zxCqNY0 )
  
  
  
  
##  Model Statistical Outputs:
  
  
**Dep. Variable**: The dependent variable or target variable
  
**Model**: Highlight the model used to obtain this output. It is OLS here. Ordinary least squares / Linear regression
  
**Method**: The method used to fit the data to the model. Least squares
  
**No. Observations**: The number of observations
  
**DF Residuals**: The degrees of freedom of the residuals. Calculated by taking the number of observations less the number of parameters
  
**DF Model**: The number of estimated parameters in the model. In this case 13. The constant term is not included.
  
**R-squared**: This is the coefficient of determination. Measure of goodness of fit. Tells you how much your model can explain the variability of the data. The more parameters you add to your model , the more this value will go up. 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?R^2=1-&#x5C;frac{SS_{res}}{SS_{tot}}"/></p>  
  
  
From [wiki](https://en.wikipedia.org/wiki/Coefficient_of_determination ),
  
The total sum of squares, <img src="https://latex.codecogs.com/gif.latex?SS_{tot}=&#x5C;sum_i(y_i-&#x5C;bar{y})^2"/>
  
The regression sum of squares (explained sum of squares), <img src="https://latex.codecogs.com/gif.latex?SS_{reg}=&#x5C;sum_i(f_i-&#x5C;bar{y})^2"/>
  
The sum of squares of residuals (residual sum of squares), <img src="https://latex.codecogs.com/gif.latex?SS_{res}=&#x5C;sum_i(y_i-f_i)^2%20=%20&#x5C;sum_ie^2_i"/>
  
**Adj. R-squared**: This is the adjusted R-squared. It is the coefficient of determination adjusted by sample size and the number of parameters used. It is not normally used on Linear Regression. IT is trying to penalise the additional paramaters you add to your model. 
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;bar{R}^2=1-(1-R^2)&#x5C;frac{n-1}{n-p-1}"/></p>  
  
  
<img src="https://latex.codecogs.com/gif.latex?p"/> = The total number of explanatory variables not including the constant term
  
<img src="https://latex.codecogs.com/gif.latex?n"/> = The sample size
  
**F-statistic**: A measure that tells you if you model is different from a simple average or not. 
  
**p-value - Prob (F-statistic)**: This measures the significance of your F-statistic. Also called p-value of F-statistic. In statistics and before computeres got so powerful, a p-value equal or lower than 0.05 is considered significant, now it could be as low as 0.005.
  
**AIC**: This is the Akaike Information Criterion. It evaluatess the model based on the model complexity and number of observations. The lower the better. 
  
**BIC**: This is the Bayesian Information Criterion. Similar to AIC, except it pushishes models with more parameters.Again, The lower the better. 
  
  
##  Parameters Estimates and the Associated Statistical Tests
  
  
**coef**: The estimated coefficient. Note that this is just a point estimate. By itself , it doesn't mean much. the best way to read the significance of a number is to compare it against the p-value.
  
**P > |t|**: The p-value. A measure of the probability that the coefficient is different from zero. the closer to zero, the more significant.
  
**std err**: The standard error of the estimate of the coefficient. Another term for standard deviation
  
**t**: The t-statistic score. This should be compared against a t-table.
  
  
**[95.0% Conf. Interval]**: The 95% confidence interval of the coefficient. Shown here as [0.025, 0.975], the lower and upper bound.
  
  
##  Residual Tests
  
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
  
#  Feature Importance and Extractions
  
Check
1. The Direction of the coefficient. Is it going Up or Down. Positive or negative
  
2. Impact of the variable/ factor on the model. How significant is it to the scheme of things ( the model) 
- In order to measure the impact we need to standardise the variable. As an example, price of a house could range between 100k to 700k , compared to age with is just 0-100. We would want everything in the same range so nothing skews the model in to thinking it is more imporant than it actually is.
  
##  Detecting collinearity with Eigenvectors
  
Use the numpy linear algebra eigen decomposition of the correlation matrix
  
##  Use <img src="https://latex.codecogs.com/gif.latex?R^2"/> to identify key Features
  
- Compare <img src="https://latex.codecogs.com/gif.latex?R^2"/> of the model **VS** <img src="https://latex.codecogs.com/gif.latex?R^2"/> of the model **without the feature**
- A significance change in <img src="https://latex.codecogs.com/gif.latex?R^2"/> signifies the importance of the feature.
  
  
#  Gradient Descent
  
  
Inspired by [Chris McCormick on Gradient Descent Derivation](http://mccormickml.com/2014/03/04/gradient-descent-derivation/ )
  
#  Background
  
  
<img src="https://latex.codecogs.com/gif.latex?h(x)%20=%20&#x5C;theta_0%20+%20&#x5C;theta_1X"/>
  
Find the values of <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta_0"/> and <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta_1"/> which provide the best fit of our hypothesis to a training set. 
  
The training set examples are labeled <img src="https://latex.codecogs.com/gif.latex?x"/>, <img src="https://latex.codecogs.com/gif.latex?y"/>, 
  
<img src="https://latex.codecogs.com/gif.latex?x"/> is the input value and <img src="https://latex.codecogs.com/gif.latex?y"/> is the output. 
  
The <img src="https://latex.codecogs.com/gif.latex?i"/>th training example is labeled as <img src="https://latex.codecogs.com/gif.latex?x^{(i)}"/>, <img src="https://latex.codecogs.com/gif.latex?y^{(i)}"/>.
  
##  MSE Cost Function
  
  
The cost function <img src="https://latex.codecogs.com/gif.latex?J"/> for a particular choice of parameters <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> is the mean squared error (MSE):
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?J(&#x5C;theta)=&#x5C;frac{1}{m}&#x5C;sum_{i=1}^m(h_{&#x5C;theta}(x^{(i)})-y^{(i)})^2"/></p>  
  
  
<img src="https://latex.codecogs.com/gif.latex?m"/> The number of training examples
  
<img src="https://latex.codecogs.com/gif.latex?x^{(i)}"/> The input vector for the <img src="https://latex.codecogs.com/gif.latex?i^{th}"/> training example
  
<img src="https://latex.codecogs.com/gif.latex?y^{(i)}"/> The class label for the <img src="https://latex.codecogs.com/gif.latex?i^{th}"/> training example
  
<img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> The chosen parameter values of weights (<img src="https://latex.codecogs.com/gif.latex?&#x5C;theta_0,%20&#x5C;theta_1,%20&#x5C;theta_2"/>)
  
<img src="https://latex.codecogs.com/gif.latex?h_{&#x5C;theta}(x^{(i)})"/> The algorithm's prediction for the <img src="https://latex.codecogs.com/gif.latex?i^{th}"/> training example using the parameters <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/>
  
The MSE measures the mean amount that the model's predictions deviate from the correct values.
  
It is a measure of the model's performance on the training set. 
  
The cost is higher when the model is performing poorly on the training set. 
  
The objective of the learning algorithm is to find the parameters <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> which give the minimum possible cost <img src="https://latex.codecogs.com/gif.latex?J"/>.
  
  
This minimization objective is expressed using the following notation, which simply states that we want to find the <img src="https://latex.codecogs.com/gif.latex?&#x5C;theta"/> which minimizes the cost <img src="https://latex.codecogs.com/gif.latex?J(&#x5C;theta)"/>.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;min_{&#x5C;theta}J(&#x5C;theta)"/></p>  
  
  
##  Regularised Method for Regression
  
  
  
These examples below each have their own way of regularising the coefficient:
  
* [Ridge Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html )
* [Least Absolute Shrinkage and Selection Operator (LASSO)](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html )
* [Elastic Net](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html )
  
##  Ridge Regression
  
Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression )
  
Ridge regression addresses some of the problems of **Ordinary Least Squares** by imposing a penalty on the size of coefficients. Especially those problems caused by outliers wich shift the coefficient quite substantially. The ridge coefficients minimize a penalized residual sum of squares,
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;min_{w}&#x5C;big|&#x5C;big|Xw-y&#x5C;big|&#x5C;big|^2_2+&#x5C;alpha&#x5C;big|&#x5C;big|w&#x5C;big|&#x5C;big|^2_2"/></p>  
  
  
where by 
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;min_{w}&#x5C;big|&#x5C;big|Xw-y&#x5C;big|&#x5C;big|^2_2"/></p>  
  
is the minimum of your essitmates minus your y; squared.
  
and 
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha&#x5C;big|&#x5C;big|w&#x5C;big|&#x5C;big|^2_2"/></p>  
 is your penalty term
  
  
  
<img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha&gt;=0"/> is a complexity parameter that controls the amount of shrinkage: the larger the value of <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha"/>, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity .
  
Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.
  
Shows the effect of collinearity in the coefficients of an estimator.
  
Ridge Regression is the estimator used in this example. Each color represents a different feature of the coefficient vector, and this is displayed as a function of the regularization parameter.
  
This example also shows the usefulness of applying Ridge regression to highly ill-conditioned matrices. For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).
  
#  Summary
  
  
[Question in StackExchange](https://stats.stackexchange.com/questions/866/when-should-i-use-lasso-vs-ridge )
  
**When should I use Lasso, Ridge or Elastic Net?**
  
* **Ridge regression** can't zero out coefficients; You either end up including all the coefficients in the model, or none of them. 
  
* **LASSO** does both parameter shrinkage and variable selection automatically. 
  
* If some of your covariates are highly correlated, you may want to look at the **Elastic Net** instead of the LASSO.
  
#  Other References
  
  
1. [The Lasso Page](http://statweb.stanford.edu/~tibs/lasso.html )
  
2. [A simple explanation of the Lasso and Least Angle Regression](http://statweb.stanford.edu/~tibs/lasso/simple.html )
  
3. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf )
  
  
#  Data Pre-Processing 
  
  
When it comes to continuous nominal data, There are 4 main types of preprocessing
- Standardization / Mean Removal
- Min-Max or Scaling Features to a Range
- Normalization
- Binarization
  
**Assumptions**:
* Implicit/explicit assumption of machine learning algorithms: The features follow a normal distribution (Bell-curve , +- 3SD).
* Most method are based on linear assumptions
* Most machine learning requires the data to be standard normally distributed. Gaussian with zero mean and unit variance (SD/var = 1). IF we don't have variance and SD of one , it will be hard for the ML model to converge to a reasonable solution.
  
[scikit-learn:](http://scikit-learn.org/stable/modules/preprocessing.html ) In practice we often ignore the shape of the distribution and just transform the data to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
  
For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) **assume that all features are centered around zero and have variance in the same order**. Variance referring to multiple features. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
  
#  Standardization / Mean Removal / Variance Scaling
  
  
[scikit Scale](http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling )
  
Mean is removed. Data is centered on zero. This is to remove bias.
  
Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance. standard normal random variable with mean 0 and standard deviation 1.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X&#x27;=&#x5C;frac{X-&#x5C;bar{X}}{&#x5C;sigma}"/></p>  
  
  
Where 
  
<img src="https://latex.codecogs.com/gif.latex?{X-&#x5C;bar{X}}"/>
  
Keeping in mind that if you have scaled your training data, you must do likewise with your test data as well. However, your assumption is that the mean and variance must be invariant between your train and test data. `scikit-learn` assists with a built-in utility function `StandardScaler`.
  
  
  
#  Min-Max or Scaling Features to a Range
  
  
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
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X_{std}=&#x5C;frac{X-X_{min}}{X_{max}-X_{min}}"/></p>  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X&#x27;=X_{std}%20(&#x5C;text{max}%20-%20&#x5C;text{min})%20+%20&#x5C;text{min}"/></p>  
  
  
##  MaxAbsScaler
  
  
Works in a very similar fashion, but scales in a way that the training data lies within the range `[-1, 1]` by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data.
  
##  Scaling sparse data
  
  
Centering sparse data would destroy the sparseness structure in the data, and thus rarely is a sensible thing to do. 
  
However, it can make sense to scale sparse inputs, especially if features are on different scales.
  
`MaxAbsScaler` and `maxabs_scale` were specifically designed for scaling sparse data
[Compare the effect of different scalers on data with outliers](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py )
  
  
##  Scaling vs Whitening
  
  
It is sometimes not enough to center and scale the features independently, since a downstream model can further make some assumption on the linear independence of the features.
  
To address this issue you can use `sklearn.decomposition.PCA` or `sklearn.decomposition.RandomizedPCA` with `whiten=True` to further remove the linear correlation across features.
  
  
#  Normalization
  
Normalization is the process of scaling individual samples to have unit norm. 
  
This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
What you do is take the the mean away from each value and divide it by the range. 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?X&#x27;=&#x5C;frac{X-X_{mean}}{X_{max}-X_{min}}"/></p>  
  
  
This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
  
There are two types of Normalization
  
  1. **L1 normalization**, Least Absolute Deviations
Ensure the sum of absolute values is 1 in each row. 
  
  2. **L2 normalization**, Least squares, 
Ensure that the sum of squares is 1.
  
#  Binarization
  
  
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?f(x)={0,1}"/></p>  
  
  
Feature binarization is the process of thresholding numerical features to get boolean values. You make a value either 0 or 1 depending on a threshold.  This can be useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution. Bernoulli is either zero or one. 
  
  
It is also common among the text processing community to use binary feature values (probably to simplify the probabilistic reasoning) even if normalized counts (a.k.a. term frequencies) or TF-IDF valued features often perform slightly better in practice.
  
#  Encoding categorical features
  
  
There are two main ways to do this
- label encodeing where the value is coded to a numerical value for example country to code mapping:
  - Australia 	 0
  - Hong Kong 	 1
  - New Zealand  2
  - Singapore 	 3
The problem with this is a Machine might think Singapore is more important that Hong Kong. to get around this they may use One Hot / One-of-K Encoding.
  
##  One Hot / One-of-K Encoding
  
  
* Useful for dealing with sparse matrix
* uses [one-of-k scheme](http://code-factor.blogspot.sg/2012/10/one-hotone-of-k-data-encoder-for.html )
  
The process of turning a series of categorical responses into a set of binary result (0 or 1)
For example if we use One Hot encodeing on the four countries again we would get a matrix as follows:
[[1. 0. 0. 0.]  = Australia
 [0. 0. 0. 1.]  = Hong Kong
 [0. 0. 1. 0.]  = New Zealand
 [0. 1. 0. 0.]] = Singapore
  
These values can be inverted so we could call [3, :] and get Singapore returned to us. 
  
#  Data Pre-Processing References
  
  
* [Section - Should I normalize/standardize/rescale the data?](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html )
* [Colleen Farrelly - Machine Learning by Analogy](https://www.slideshare.net/ColleenFarrelly/machine-learning-by-analogy-59094152 )
* [Lior Rokach - Introduction to Machine Learning](https://www.slideshare.net/liorrokach/introduction-to-machine-learning-13809045 )
* [Ritchie Ng](http://www.ritchieng.com/machinelearning-one-hot-encoding/ )
  
  
#  Bias Variance trade off
  
  
Every estimator ( Linear Regression, SVM etc ) has its advantages and drawbacks. Its generalization error can be decomposed in terms of bias, variance and noise. 
- The **bias** of an estimator is its average error for different training sets. These are the errors.
- The **variance** of an estimator indicates how sensitive it is to varying training sets. 
- Noise is a property of the data.
  
Bias and variance are inherent properties of estimators and we usually have to select learning algorithms and hyperparameters so that both bias and variance are as low as possible. Another way to reduce the variance of a model is to use more training data. However, you should only collect more training data if the true function is too complex to be approximated by an estimator with a lower variance. 
  
#  Validation Curve
  
  
* The purpose of the validation curve is for the identification of over- and under-fitting
* Plotting training and validation scores vs model parameters.
  
#  Learning Curve
  
  
* Shows the validation and training score of an estimator for varying numbers of training samples. This is VS the training sample. Where as the Validation Curve shows the training and validation scores vs model (hyper)parameters.
* A tool to find out how much we benefit from adding more training data and whether the actual estimator suffers more from a variance error or a bias error. 
* If both the validation score and the training score converge to a value that is too low with increasing size of the training set, we will not benefit much from more training data. 
  
  
#  Cross Validation (CV)
  
  
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
  
##  Holdout Method
  
  
 - Split initial dataset into a separate traininn and test dataset. eg 70/30, 80/20
 - Training dataset - used for model training
 - Test dataset - used to estimate its generalisation performance
  
A variation is to split the training set to two :- training set and validation set.
Training set :- For tuning and comparing different parameter setting to further improve the performance for making prediction on unseen data and also for model selection.
This process is called model selection. We want to select to optimal values to tuning parameters (aka hyperparameteres) 
  
##  k-fold Cross Validation
  
- Randomly split the training dataset into k-folds without replacemet ( meaning you dont put data back in when its been used????). 
- k-1 folds are use for the model training. 
- the one fold is used for performance evaluation. 
  
The procedure is repeated k times. 
Final outcomes: - k models and performance estimates. 
  
- calculate the average performance of the models based on the different, independent folds to obtain a performance estimate that is less sensitive to the sub-partitionaing of the trained data compared to the holdout method.
- k-fold cross-validation is used for model tuning , Finding the optimal hyperparameter values that yields a satisfying generalization performance.
- Once we have found satisfactory hyperparameter values, we can retain the model on the complete training set and obtain a final performance estimate useing the independent test set. The rationale behind fitting a model to the whole training dataset after k-fold cross-validation is that providing more training samples to a learning algorithm usually results in a more accurate and robust model. 
  
###  Stratified k-fold cross-validation
  
- variation of k-fold that can yield better bias and variance estimates, especially in cases of unequal class proportions.
  
See The [The scoring parameter: defining model evaluation rules](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter ) for details. In the case of the Iris dataset, the samples are balanced across target classes hence the accuracy and the F1-score are almost equal.
  
#  4. CLASSIFICATION 
  
  
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
  
#  Logistic Regression
  
##  Logistic Regression Resources:
  
  
[Logistic Regression Tutorial for Machine Learning](http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/ )
[Logistic Regression for Machine Learning](http://machinelearningmastery.com/logistic-regression-for-machine-learning/ )
[How To Implement Logistic Regression With Stochastic Gradient Descent From Scratch With Python](http://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/ )
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
[A comparison of numerical optimizers for logistic regression](https://tminka.github.io/papers/logreg/ )
[PDF: A comparison of numerical optimizers for logistic regression](https://tminka.github.io/papers/logreg/minka-logreg.pdf )
  
Logistic regression is the go-to linear classification algorithm for two-class(binary,yes/no,0/1) problems. It is easy to implement, easy to understand and gets great results on a wide variety of problems, even when the expectations the method has for your data are violated. IT is often used in he market place for things like default predictions or fraud detection.
  
Logistic regression is named for the function used at the core of the method, the [logistic function](https://en.wikipedia.org/wiki/Logistic_function ).
The logistic function, also called the **Sigmoid function** was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{1}{1%20+%20e^{-x}}"/></p>  
  
  
<img src="https://latex.codecogs.com/gif.latex?e"/> is the base of the natural logarithms and <img src="https://latex.codecogs.com/gif.latex?x"/> is value that you want to transform via the logistic function.
  
The logistic regression equation has a very similar representation like linear regression. The difference is that the output value being modelled is binary in nature.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;hat{y}=&#x5C;frac{e^{&#x5C;beta_0+&#x5C;beta_1x_1}}{1+&#x5C;beta_0+&#x5C;beta_1x_1}"/></p>  
  
  
or
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;hat{y}=&#x5C;frac{1.0}{1.0+e^{-&#x5C;beta_0-&#x5C;beta_1x_1}}"/></p>  
  
  
<img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_0"/> is the intecept term
  
<img src="https://latex.codecogs.com/gif.latex?&#x5C;beta_1"/> is the coefficient for <img src="https://latex.codecogs.com/gif.latex?x_1"/>
  
<img src="https://latex.codecogs.com/gif.latex?&#x5C;hat{y}"/> is the predicted output with real value between 0 and 1. To convert this to binary output of 0 or 1, this would either need to be rounded to an integer value or a cutoff point be provided to specify the class segregation point.
  
TO run the predction you would do as follows. You have some data and you have been luckily been given the coefficients. A simple model would be as below. 
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
#  Learning the Logistic Regression Model
  
  
The coefficients (Beta values b) of the logistic regression algorithm must be estimated from your training data. 
  
* Generally done using [maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation ).
* Maximum-likelihood estimation is a common learning algorithm
* Note the underlying assumptions about the distribution of your data
* The best coefficients would result in a model that would predict a value very close to 1 (e.g. male) for the default class and a value very close to 0 (e.g. female) for the other class. 
* The intuition for maximum-likelihood for logistic regression is that a search procedure seeks values for the coefficients (Beta values) that minimize the error in the probabilities predicted by the model to those in the data.
  
#  Learning with Stochastic Gradient Descent
  
Logistic Regression uses gradient descent to update the coefficients.
Each gradient descent iteration, the coefficients are updated using the equation:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;beta=&#x5C;beta+&#x5C;textrm{learning%20rate}&#x5C;times%20(y-&#x5C;hat{y})%20&#x5C;times%20&#x5C;hat{y}%20&#x5C;times%20(1-&#x5C;hat{y})%20&#x5C;times%20x"/></p>  
  
  
#  Classification Based Machine Learning Algorithm
  
[An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction )
This notebook is inspired by Geron [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do )
  
##  Scikit-learn Definition:
  
  
**Supervised learning**, in which the data comes with additional attributes that we want to predict. This problem can be either:
  
* **Classification**: samples belong to two or more **classes** and we want to learn from already labeled data how to predict the class of unlabeled data. An example of classification problem would be the handwritten digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories. Another way to think of classification is as a discrete (as opposed to continuous) form of supervised learning where one has a limited number of categories and for each of the n samples provided, one is to try to label them with the correct category or class.
  
* **Regression**: if the desired output consists of one or more **continuous variables**, then the task is called regression. An example of a regression problem would be the prediction of the length of a salmon as a function of its age and weight.
  
MNIST dataset - a set of 70,000 small images of digits handwritten. You can read more via [The MNIST Database](http://yann.lecun.com/exdb/mnist/ )
  
#  Performance Measures
  
##  Measuring Accuracy Using Cross-Validation
  
##  Stratified-K-Fold
  
  
Stratified-K-Fold utilised the Stratified sampling concept
  
* The population is divided into homogeneous subgroups called strata
* The right number of instances is sampled from each stratum 
* To guarantee that the test set is representative of the population
  
Bare this in mind when you are dealing with **skewed datasets**. Because of this, accuracy is generally not the preferred performance measure for classifiers.
  
#  Confusion Matrix
  
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
  
##  Precision
  
  
**Precision** measures the accuracy of positive predictions of your classifier. Also called the `precision` of the classifier. Focus is on the second row of the matrix.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;textrm{precision}%20=%20&#x5C;frac{&#x5C;textrm{True%20Positives}}{&#x5C;textrm{True%20Positives}%20+%20&#x5C;textrm{False%20Positives}}"/></p>  
  
  
##  Recall
  
`Precision` is typically used with `recall` (`Sensitivity` or `True Positive Rate`). The ratio of positive instances that are correctly detected by the classifier. This will be interested in the bottom row of the Confusion Matrix. 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;textrm{recall}%20=%20&#x5C;frac{&#x5C;textrm{True%20Positives}}{&#x5C;textrm{True%20Positives}%20+%20&#x5C;textrm{False%20Negatives}}"/></p>  
  
  
##  F1 Score
  
  
<img src="https://latex.codecogs.com/gif.latex?F_1"/> score is the harmonic mean of precision and recall. Regular mean gives equal weight to all values which we don't always want. Harmonic mean gives more weight to low values.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?F_1=&#x5C;frac{2}{&#x5C;frac{1}{&#x5C;textrm{precision}}+&#x5C;frac{1}{&#x5C;textrm{recall}}}=2&#x5C;times%20&#x5C;frac{&#x5C;textrm{precision}&#x5C;times%20&#x5C;textrm{recall}}{&#x5C;textrm{precision}+%20&#x5C;textrm{recall}}=&#x5C;frac{TP}{TP+&#x5C;frac{FN+FP}{2}}"/></p>  
  
  
The <img src="https://latex.codecogs.com/gif.latex?F_1"/> score favours classifiers that have similar precision and recall.
  
##  Accuracy 
  
= (TP+TN)/(TP+TN+FN+FP) 
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;textrm{Accuracy}%20=%20&#x5C;frac{&#x5C;textrm{True%20Positives%20+%20True%20Negatives}}{&#x5C;textrm{True%20Positives%20+%20False%20Positives}%20+%20&#x5C;textrm{False%20Negatives%20+%20True%20Negatives}}"/></p>  
  
  
##  Precision / Recall Tradeoff
  
  
Increasing precision will reduce recall and vice versa.
  
  
#  5. Support Vector Machines
  
- Linear Classification
- Polynomial Kernel
- Radial Basis Function(RBF) / Gaussian Kernel - Draws a curve
- Support Vector Regression
- Grid Search 
  - HyperParameter Tuning
  
Invented in [1963](https://en.wikipedia.org/wiki/Support_vector_machine#History ) by [Vladimir N. Vapnik](https://en.wikipedia.org/wiki/Vladimir_Vapnik ) and Alexey Ya. Chervonenkis while working at AT&T Bell Labs. Vladimir N. Vapnik joined Facebook AI Research in Nov 2014. In 1992, Bernhard E. Boser, Isabelle M. Guyon and Vladimir N. Vapnik suggested a way to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard incarnation (soft margin) was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.
  
References:
1. [Support Vector Machine in Javascript Demo by Karpathy](http://cs.stanford.edu/people/karpathy/svmjs/demo/ )
2. [SVM](http://www.svms.org/tutorials/ )
3. [Statsoft](http://www.statsoft.com/Textbook/Support-Vector-Machines )
4. [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine )
5. [Scikit-Learn](http://scikit-learn.org/stable/modules/svm.html )
* [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC )
  Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
* [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC )
  C-Support Vector Classification.
  The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than 10000 samples.
  
##  Introduction
  
**Note:**
* SVM are sensitive to feature scaling. SO it is very important to scale features before we fit a model.
  
Supervised learning methods used for classification, regression and outliers detection.
Let's assume we have two classes here - black and purple. In classification, we are interested in the best way to separate the two classes. 
However, there are infinite lines (in 2-dimensional space) or hyperplanes (in 3-dimensional space) that can be used to separate the two classes as the example below illustrates. 
  
The term hyperplane essentially means it is a subspace of one dimension less than its ambient space. If a space is 3-dimensional then its hyperplanes are the 2-dimensional planes, while if the space is 2-dimensional, its hyperplanes are the 1-dimensional lines. ~ [Wikipedia](https://en.wikipedia.org/wiki/Hyperplane )
In SVM, the **separating line**, the solid brown line, is the line that allows for largest margin between the two classes. 
SVM would place the separating line in the middle of the margin, also called maximum margin. SVM will optimise and locate the hyperplane that maximises the margin of the two classes. The samples that are closest to the hyperplane are called **support vectors**, circled in red. 
  
##  Linear SVM Classification
  
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
        * smaller `C` leads to a wider street but more margin violations
        * High `C` - fewer margin violations but ends up with a smaller margin
  
  
#  Gaussian Radial Basis Function (rbf)
  
The kernel function can be any of the following:
* **linear**: <img src="https://latex.codecogs.com/gif.latex?&#x5C;langle%20x,%20x&#x27;&#x5C;rangle"/>.
* **polynomial**: <img src="https://latex.codecogs.com/gif.latex?(&#x5C;gamma%20&#x5C;langle%20x,%20x&#x27;&#x5C;rangle%20+%20r)^d"/>. The deggree makes it poly-.
  <img src="https://latex.codecogs.com/gif.latex?d"/> is specified by keyword `degree`
  <img src="https://latex.codecogs.com/gif.latex?r"/> by `coef0`.
* **rbf**: <img src="https://latex.codecogs.com/gif.latex?&#x5C;exp(-&#x5C;gamma%20&#x5C;|x-x&#x27;&#x5C;|^2)"/>. 
  <img src="https://latex.codecogs.com/gif.latex?&#x5C;gamma"/> is specified by keyword `gamma` must be greater than 0.
* **sigmoid** <img src="https://latex.codecogs.com/gif.latex?(&#x5C;tanh(&#x5C;gamma%20&#x5C;langle%20x,x&#x27;&#x5C;rangle%20+%20r))"/>
  where <img src="https://latex.codecogs.com/gif.latex?r"/> is specified by `coef0`.  
[scikit-learn documentation](http://scikit-learn.org/stable/modules/svm.html#svm )
  
##  Grid search
  
  
Grid search is a way you can search through permutations of hyperparameters in order to validate the the best combination of hyperparameters with our model. 
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
  
  
##  Advantages and Disadvantages of SVM
  
  
The **advantages** of support vector machines are:
* Effective in high dimensional spaces.
* Uses only a subset of training points (Support Vvectors) in the decision function.
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
  
##  SVM Summary
  
SciKit learn has three types of SVM:
  
| Class |  Out-of-core support | Kernel Trick |
| :- |  :- | :- | :- |
| `SGDClassifier` |  Yes | No |
| `LinearSVC` |  No | No |
| `SVC` |  No | Yes |
  
**Note:** All require features scaling
  
Support Vector Machine algorithms are not scale invariant(ie; highly dependent on the scale), so it is highly recommended to scale your data. For example, scale each attribute on the input vector X to [0,1] or [-1,+1], or standardize it to have mean 0 and variance 1. Note that the same scaling must be applied to the test vector to obtain meaningful results. See section Preprocessing data for more details on scaling and normalization. ~ [scikit-learn documentation](http://scikit-learn.org/stable/modules/svm.html#svm )
  
#  Where to From Here
  
  
* [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/ )
* [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/ )
* [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.safaribooksonline.com/library/view/hands-on-machine-learning/9781491962282/ch05.html#svm_chapter )
* [Python Data Science Handbook](https://www.safaribooksonline.com/library/view/python-data-science/9781491912126/ch05.html#in-depth-support-vector-machines )
* [Python Machine Learning, 2E](https://www.safaribooksonline.com/library/view/python-machine-learning/9781787125933/ch03s04.html )
* [Statistics for Machine Learning](https://www.safaribooksonline.com/library/view/statistics-for-machine/9781788295758/f2c95085-6676-41c6-876e-ab6802666ea2.xhtml )
* [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/ )
  
**Q: What is Kernel, Loss , gamma, cost in SVM exactly?**
  
#  7. Decisions Tree
  
  
Aka:CART (Classification  and Regression Tree)
* Supervised Learning
* Works for both classification and regression
* Foundation of Random Forests
* Attractive because of interpretability
Decision Tree works by:
* Split based on set impurity criteria
* Stopping criteria (eg; Depth)
  
Source: [Scikit-Learn](http://scikit-learn.org/stable/modules/tree.html#tree )
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
  
##  Decision Tree Learning
  
  
* [ID3](https://en.wikipedia.org/wiki/ID3_algorithm ) (Iterative Dichotomiser 3)
* [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm ) (successor of ID3)
* CART (Classification And Regression Tree)
* [CHAID](http://www.statisticssolutions.com/non-parametric-analysis-chaid/ ) (Chi-squared Automatic Interaction Detector). by [Gordon Kass](https://en.wikipedia.org/wiki/Chi-square_automatic_interaction_detection ). 
  
##  Tree algorithms: ID3, C4.5, C5.0 and CART
  
  
* ID3 (Iterative Dichotomiser 3) was developed in 1986 by Ross Quinlan. The algorithm creates a multiway tree, finding for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the ability of the tree to generalise to unseen data.
  
* C4.5 is the successor to ID3 and removed the restriction that features must be categorical by dynamically defining a discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set of intervals. Ie : it takes all the data and chops it up into n catagories. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules. These accuracy of each rule is then evaluated to determine the order in which they should be applied. Pruning is done by removing a rule’s precondition if the accuracy of the rule improves without it.
  
* C5.0 is Quinlan’s latest version release under a proprietary license. It uses less memory and builds smaller rulesets than C4.5 while being more accurate.
  
* CART (Classification and Regression Trees) is very similar to C4.5, but it differs in that it supports numerical target variables (regression) and does not compute rule sets. CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.
  
* CHAID (Chi-squared Automatic Interaction Detector). by Gordon Kass. Performs multi-level splits when computing classification trees. Non-parametric so it does not require the data to be normally distributed. Often used on Ordinal Data.
  
scikit-learn uses an optimised version of the CART algorithm.
  
##  Gini Impurity
  
  
scikit-learn default
  
[Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity )
  
If all data are of the same class then the gini will be 0.0. If 25% are of one class, that gini will be 0.25. The largest gini value will be under the root node and as you split , the gini will get smaller. 0 is pure, 1 is full diversity.
  
A measure of purity / variability of categorical data. As a side note on the difference between [Gini Impurity and Gini Coefficient](https://datascience.stackexchange.com/questions/1095/gini-coefficient-vs-gini-impurity-decision-trees ). 
  
* No, despite their names they are not equivalent or even that similar.
* **Gini impurity** is a measure of misclassification, which applies in a multiclass classifier context.
* **Gini coefficient** applies to binary classification and requires a classifier that can in some way rank examples according to the likelihood of being in a positive class.
* Both could be applied in some cases, but they are different measures for different things. Impurity is what is commonly used in decision trees.
  
Developed by [Corrado Gini](https://en.wikipedia.org/wiki/Corrado_Gini ) in 1912.
  
Key Points:
* A pure node (homogeneous contents or samples with the same class) will have a Gini coefficient of zero
* As the variation increases (heterogeneneous classes or increase diversity), Gini coefficient increases and approaches 1.
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?Gini=1-&#x5C;sum^r_j%20p^2_j"/></p>  
  
  
<img src="https://latex.codecogs.com/gif.latex?p"/> is the probability (often based on the frequency table)
  
##  Entropy
  
[Wikipedia](https://en.wikipedia.org/wiki/Entropy_(information_theory ))
The entropy can explicitly be written as
<p align="center"><img src="https://latex.codecogs.com/gif.latex?{&#x5C;displaystyle%20&#x5C;mathrm%20{H}%20(X)=&#x5C;sum%20_{i=1}^{n}{&#x5C;mathrm%20{P}%20(x_{i})&#x5C;,&#x5C;mathrm%20{I}%20(x_{i})}=-&#x5C;sum%20_{i=1}^{n}{&#x5C;mathrm%20{P}%20(x_{i})&#x5C;log%20_{b}&#x5C;mathrm%20{P}%20(x_{i})},}"/></p>  
  
where `b` is the base of the logarithm used. Common values of `b` are 2, Euler's number `e`, and 10.
  
Note: The probability of the individual observation multiplied by the log of the individual observation
  
##  Which should I use? Entropy or Gini
  
[Sebastian Raschka](https://sebastianraschka.com/faq/docs/decision-tree-binary.html )
* They tend to generate similar tree
* Gini tends to be faster to compute
  
##  Information Gain
  
* Expected reduction in entropy caused by splitting 
* Keep splitting until you obtain a as close to homogeneous class as possible
  
  
##  Trees - Where to From Here
  
###  Tips on practical use
  
* Decision trees tend to overfit on data with a large number of features. Check ratio of samples to number of features
* Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand
* Visualise your tree as you are training by using the export function. Use max_depth=3 as an initial tree depth.
* Use max_depth to control the size of the tree to prevent overfitting.
* Tune `min_samples_split` or `min_samples_leaf` to control the number of samples at a leaf node. 
* Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. 
  * By sampling an equal number of samples from each class
  * By normalizing the sum of the sample weights (sample_weight) for each class to the same value. 
  
#  References:
  
1. [Wikipedia - Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning )
2. [Decision Tree - Classification](http://www.saedsayad.com/decision_tree.htm )
3. [Data Aspirant](http://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/ )
4. [Scikit-learn](http://scikit-learn.org/stable/modules/tree.html )
5. https://en.wikipedia.org/wiki/Predictive_analytics
6. L. Breiman, J. Friedman, R. Olshen, and C. Stone. Classification and Regression Trees. Wadsworth, Belmont, CA, 1984.
7. J.R. Quinlan. C4. 5: programs for machine learning. Morgan Kaufmann, 1993.
8. T. Hastie, R. Tibshirani and J. Friedman. Elements of Statistical Learning, Springer, 2009.
  
  
#  8. Ensemble Methods: Combineing models 
  
  
**Note: Ensemble methods** It is still supervised.
* Work best with indepedent predictors
* Best to utilise different algorithms
  
  
###  **B**ootstrap **Agg**regat**ing** or [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating )
  
* [Scikit- Learn Reference](http://scikit-learn.org/stable/modules/ensemble.html#bagging )
* Bootstrap sampling: Sampling with replacement - (put the sample back in the pool to possibly take the same one again.)
* Combine by averaging the output (regression)
* Combine by voting (classification) - 
* Can be applied to many classifiers which includes ANN(Artificial Neural Network), CART, etc.
  
  
###  [Pasting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html ))
  
* Similar to bagging apart from **Sampling without replacement**.
  
###  [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning ))
  
You start with an equal weight and then once you have iterated, you increase the weighting of the weak classifiers and go round again. It iis seriel and learns fr omthe past becasue you have to wait for the previous iteration to finish. So it can take longer. 
* Train weak classifiers.
* Add them to a final strong classifier by weighting. Weighting by accuracy (typically)
* Once added, the data are reweighted
  * **Misclassified** samples **gain weight** 
  * **Correctly** classified samples **lose weight** (Exception: Boost by majority and BrownBoost - decrease the weight of repeatedly misclassified examples). 
  * Algo are forced to learn more from misclassified samples
  
  
###  [Stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/ )
  
* Also known as Stacked generalization
* [From Kaggle:](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/ ) Combine information from multiple predictive models to generate a new model. Often times the stacked model (also called 2nd-level model) will outperform each of the individual models due its smoothing nature and ability to highlight each base model where it performs best and discredit each base model where it performs poorly. For this reason, stacking is most effective when the base models are significantly different. 
* Training a learning algorithm to combine the predictions of several other learning algorithms. 
  * Step 1: Train learning algo
  * Step 2: Combiner algo is trained using algo predictions from step 1.  
  
###  Other Ensemble Methods:
  
  
[Wikipedia](https://en.wikipedia.org/wiki/Ensemble_learning )
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
  
##  Random Forest
  
  
[Original paper of Random Forest](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf )
  
* Random Forest is basically an Ensemble of Decision Trees
  
* Training via the bagging method (Repeated sampling with replacement)
  * Bagging: Sample from samples - 
  * RF: Sample from predictors. <img src="https://latex.codecogs.com/gif.latex?m=sqrt(p)"/> for classification and <img src="https://latex.codecogs.com/gif.latex?m=p&#x2F;3"/> for regression problems.
  
* Utilise uncorrelated trees
  
  
Random Forest
* Sample both **observations and features** of training data.
It will ignore other feature when it is doing sampleing. 
  
Bagging
* Samples **only observations at random**
* Decision Tree select best feature when splitting a node. It will always go for the best feature to split a node.
* Focus on the training data and leave the features by inclueing them all??. 
  
Running on the Titanic Dataset we didnt get doo results. Probably because the dataset was too small.
  
##  Extra-Trees (Extremely Randomized Trees) Ensemble
  
  
[scikit-learn](http://scikit-learn.org/stable/modules/ensemble.html#bagging )
  
* Random Forest is build upon Decision Tree
* Decision Tree node splitting is based on gini or entropy or some other algorithms
* Extra-Trees make use of random thresholds for each feature unlike Decision Tree
  
When it comes to Random Forest, the sample is drawn by bootstrap sampleing. The splitting of node is used to construct the tree, but on the second time it will randomly sample different features rather than the previously best fetures. This adds an extra source of randomness. This does increase the bias slightly but becasue of averageing , the variace decreases. Typically this impves the SD and Variance at the slight sacrifice of bias .
  
Caompareing with  Extremely Randomized Trees the randomness is introduced in the way the note splittings are computed. In random forest, a random subset of the features are used but in Extremely Randomized Trees the threashold is randonly generated which introdues another layer of variance. 
  
  
#  Boosting (Aka Hypothesis Boosting)
  
* Combine several weak learners into a strong learner. 
* Train predictors sequentially
  
#  AdaBoost / Adaptive Boosting
  
  
Getting on the cutting edge.
[Creator: Robert Schapire](http://rob.schapire.net/papers/explaining-adaboost.pdf )
[Wikipedia](https://en.wikipedia.org/wiki/AdaBoost )
[Chris McCormick](http://mccormickml.com/2013/12/13/adaboost-tutorial/ )
[Scikit Learn AdaBoost](http://scikit-learn.org/stable/modules/ensemble.html#adaboost )
  
1995
  
As above for Boosting:
* Similar to human learning, the algo learns from past mistakes by focusing more on difficult problems it did not get right in prior learning. You make a mistake, learn from them, spend more time on them. 
* In machine learning speak, **it pays more attention to training instances that previously underfitted.**
  
###  PsuedoCode
  
Source: Scikit-Learn:
  
* Fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. 
* The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.
* Then, The data modifications at each so-called boosting iteration consist of applying weights <img src="https://latex.codecogs.com/gif.latex?w_1,%20w_2,%20…,%20w_N"/> to each of the training samples. 
* Initially, these weights are all set to <img src="https://latex.codecogs.com/gif.latex?w_i%20=%201&#x2F;N"/>, so that the first step simply trains a weak learner on the original data. 
* For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data. Here theweights are shifted.
* At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly. 
* **As iterations proceed, examples that are difficult to predict receive ever-increasing influence.** Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.
  
##  Adaboost basic params
  
```
{'algorithm': 'SAMME.R',
 'base_estimator': None,
 'learning_rate': 1.0,
 'n_estimators': 50,
 'random_state': None}
 ```
[SAMME16](https://web.stanford.edu/~hastie/Papers/samme.pdf ) (Stagewise Additive Modeling using a Multiclass Exponential loss function).
  
R stands for real
  
  
# Gradient Boosting / Gradient Boosting Machine (GBM)
  
Works for both regression and classification. IT learns from the mistakes and continuously tries to improve from there.
  
[Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting )
  
* Sequentially adding predictors, so a little differnt from AdaBoost. These are added bit by bit. Not all predictors are being exposed.
* Each one correcting its predecessor
* Fit new predictor to the residual errors.
  
Compare this to AdaBoost: 
* Alter instance weights at every iteration. Meaning the area of mistakes increases the weight. 
  
So you apply a model () predictors to predict the target outcome) which will explain some of the variance. The residuals which are left and are not explained, you apply another predictor to it until you get to the final predictors. 
  
**Step 1.** Basically simple linear regression
  <p align="center"><img src="https://latex.codecogs.com/gif.latex?Y%20=%20F(x)%20+%20&#x5C;epsilon"/></p>  
  
**Step 2.** With the error term (<img src="https://latex.codecogs.com/gif.latex?&#x5C;epsilon"/>) It is optimised and run it through another predictor <img src="https://latex.codecogs.com/gif.latex?G(x_2)"/> , rather than starting from scratch. 
  <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;epsilon%20=%20G(x_2)%20+%20&#x5C;epsilon_2"/></p>  
  
  Substituting step (2) into step (1), we get :  
  <p align="center"><img src="https://latex.codecogs.com/gif.latex?Y%20=%20F(x)%20+%20G(x)%20+%20&#x5C;epsilon_2"/></p>  
  
**Step 3.** We create another function <img src="https://latex.codecogs.com/gif.latex?H(x)"/>
  <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;epsilon_2%20=%20H(x)%20%20+%20&#x5C;epsilon_3"/></p>  
  
We can keep on continuesin until we have used all of the predictors:  
  <p align="center"><img src="https://latex.codecogs.com/gif.latex?Y%20=%20F(x)%20+%20G(x)%20+%20H(x)%20%20+%20&#x5C;epsilon_3"/></p>  
  
Finally, by adding weighting eg with <img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha,%20%20&#x5C;beta%20,%20&#x5C;gamma"/> alpha,beta and gamma.   
  <p align="center"><img src="https://latex.codecogs.com/gif.latex?Y%20=%20&#x5C;alpha%20F(x)%20+%20&#x5C;beta%20G(x)%20+%20&#x5C;gamma%20H(x)%20%20+%20&#x5C;epsilon_4"/></p>  
  
  
The key thing with Gradient boosting is that it involves three elements:
  
* **Loss function to be optimized**: Loss function depends on the type of problem being solved. In the case of regression problems, mean squared error is used, and in classification problems, logarithmic loss will be used. In boosting, at each stage, unexplained loss from prior iterations will be optimized rather than starting from scratch.
  
* **Weak learner to make predictions**: Decision trees are used as a weak learner in gradient boosting. 
  
* **Additive model to add weak learners to minimize the loss function**: Trees are added one at a time and existing trees in the model are not changed. The gradient descent procedure is used to minimize the loss when adding trees, hence the term gradient.
  
# XGBoost (Extreme Gradient Boosting)
...it might be more suitable to be called as **regularized gradient boosting....**
  
[Documentation](http://xgboost.readthedocs.io/en/latest/ )
[tqchen github](https://github.com/tqchen/xgboost/tree/master/demo/guide-python )
[dmlc github](https://github.com/dmlc/xgboost )
* “Gradient Boosting” is proposed in the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman. 
* XGBoost is based on this original model. 
* Supervised Learning
  
Like Gradient Boosting Machines (GBM)
* Sequentially adding predictors, so a little differnt from AdaBoost. These are added bit by bit. Not all predictors are being exposed.
* Each one correcting its predecessor
* Fit new predictor to the residual errors.
  
## Objective Function : Training Loss + Regularization
<p align="center"><img src="https://latex.codecogs.com/gif.latex?Obj(Θ)=L(θ)+Ω(Θ)"/></p>  
  
* <img src="https://latex.codecogs.com/gif.latex?L"/> is the training loss function. A measure of how predictive our model is on the training data. 
* <img src="https://latex.codecogs.com/gif.latex?Ω"/> is the regularization term. The complexity of the model, which helps us to inform and avoid overfitting.
  
### Training Loss
The training loss measures how predictive our model is on training data.
Example 1, Mean Squared Error for Linear Regression:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?L(θ)=%20&#x5C;sum_i(y_i-&#x5C;hat{y}_i)^2"/></p>  
  
Example 2, Logistic Loss for Logistic Regression:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?L(θ)%20=%20&#x5C;sum_i%20&#x5C;large[%20y_i%20ln(1%20+%20e^{-&#x5C;hat{y}_i})%20+%20(1-y_i)%20ln(1%20+%20e^{&#x5C;hat{y}_i})%20&#x5C;large]"/></p>  
  
  
Hoever when we only focus on this objective function. Enter the Regularization Term.
  
### Regularization Term
What sets XGBoost appart from GBM's is this regularisation term. 
The regularization term controls the complexity of the model, which helps us to avoid overfitting. 
[XGBoost vs GBM](https://www.quora.com/What-is-the-difference-between-the-R-gbm-gradient-boosting-machine-and-xgboost-extreme-gradient-boosting/answer/Tianqi-Chen-1 )
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
  
[wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm )
  
It is both a Classification and Regression based supervised algorithm. K is how many neighbours do you want your data sample in the space to be compared to it, in order to clasify it?  
It is a very simple classifier to impliment that you can use quickly and get a certain amount of accuracy with. 
  
1. Lazy learner as it is [Instance Based](https://en.wikipedia.org/wiki/Instance-based_learning )
  * It memorise the pattern from the dataset
  * Lazy because it does not try to learn a function from the training data. 
  
2. It is a [Nonparametric model](http://blog.minitab.com/blog/adventures-in-statistics-2/choosing-between-a-nonparametric-test-and-a-parametric-test )
  * distribution-free tests because no assumption of the data needing to follow a specific distribution. Eg Normal distibution etc.
  * [wikipedia](https://en.wikipedia.org/wiki/Nonparametric_statistics )
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
  
* [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm )
* [Scikit-Learn Nearest Neighbours](http://scikit-learn.org/stable/modules/neighbors.html )
* [Introduction to k-nearest neighbors : Simplified](https://www.analyticsvidhya.com/blog/2014/10/introduction-k-neighbours-algorithm-clustering/ )
* [Quora](https://www.quora.com/What-are-industry-applications-of-the-K-nearest-neighbor-algorithm )
* [Kevin Zakka](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/ )
* [CS231n CNN](http://cs231n.github.io/classification/#nn )
* [A Detailed Introduction to K-Nearest Neighbor (KNN) Algorithm](https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/ )
* [Chris Albon](https://chrisalbon.com/ )
* [K-Nearest Neighbors for Machine Learning](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/ )
* [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php )
  
## Steps:
* Choose the number of <img src="https://latex.codecogs.com/gif.latex?k"/> , the neighbors.
* Select a distance metric
* Find the k nearest neighbors of the sample
* Assign the class label by majority vote
  
## DistanceMetric class documentation
[scikit-learn]( )
  
Metrics intended for real-valued vector spaces:
  
<img src="https://latex.codecogs.com/gif.latex?|x+y|%20&#x5C;leq%20|x|%20+%20|y|"/>
  
| identifier	| class name	| args	| distance function |
|:- |:- |:- |:- |
|"euclidean" | 	EuclideanDistance  |   | <img src="https://latex.codecogs.com/gif.latex?&#x5C;sqrt{&#x5C;sum(x%20-%20y)^2)}"/>  |  
|"manhattan" | ManhattanDistance |  | <img src="https://latex.codecogs.com/gif.latex?&#x5C;sum%20(x%20-%20y)%20%20%20|%20|%20|&quot;chebyshev&quot;%20|%20ChebyshevDistance%20|%20%20|%20max"/>{\big|x - y\big|}<img src="https://latex.codecogs.com/gif.latex?|%20%20|&quot;minkowski&quot;%20|%20MinkowskiDistance	%20|%20p	%20|"/>\sum(\big|x - y\big|^p)^{\frac{1}{p}}<img src="https://latex.codecogs.com/gif.latex?|%20|&quot;wminkowski&quot;%20|%20WMinkowskiDistance	%20|%20p,%20w	%20|"/>\sum(w\big|x - y\big|^p)^{\frac{1}{p}}<img src="https://latex.codecogs.com/gif.latex?||&quot;seuclidean&quot;%20|%20SEuclideanDistance	%20|%20V	%20|"/>\sqrt{\sum\frac{(x - y)^2}{V})}<img src="https://latex.codecogs.com/gif.latex?|%20Refer%20to%20documentation%20for%20more%20on%20*%20Metrics%20intended%20for%20two-dimensional%20vector%20spaces*%20Metrics%20intended%20for%20integer-valued%20vector%20spaces*%20Metrics%20intended%20for%20boolean-valued%20vector%20spaces*%20User-defined%20distanceSource:%20[Rorasa&#x27;s%20blog](https:&#x2F;&#x2F;rorasa.wordpress.com&#x2F;2012&#x2F;05&#x2F;13&#x2F;l0-norm-l1-norm-l2-norm-l-infinity-norm&#x2F;%20)*%20Mathematically%20a%20norm%20is%20a%20total%20size%20or%20length%20of%20all%20vectors%20in%20a%20vector%20space%20or%20matrices.%20*%20For%20simplicity,%20we%20can%20say%20that%20the%20higher%20the%20norm%20is,%20the%20bigger%20the%20(value%20in)%20matrix%20or%20vector%20is.%20*%20Norm%20may%20come%20in%20many%20forms%20and%20many%20names,%20including%20these%20popular%20name:%20Euclidean%20distance,%20Mean-squared%20Error,%20etc.*%20Most%20of%20the%20time%20you%20will%20see%20the%20norm%20appears%20in%20a%20equation%20like%20this:"/>\left \| x \right \|<img src="https://latex.codecogs.com/gif.latex?where"/>x<img src="https://latex.codecogs.com/gif.latex?can%20be%20a%20vector%20or%20a%20matrix.*%20Euclidean%20distance%20-%20Strightline%20connecting%20two%20points%20%20*%20Most%20common%20%20*%20The%20Euclidean%20distance%20between%20points%20(1,2)%20and%20(3,3)%20can%20be%20computed"/>\sqrt{(1-3)^2+(2-3)^2}<img src="https://latex.codecogs.com/gif.latex?,%20which%20results%20in%20a%20distance%20of%20about%202.236.%20%20*%20L2%20norm%20of%20two%20vectors.%20%20%20*%20In%20a%20bidimensional%20plane,%20the%20Euclidean%20distance%20refigures%20as%20the%20straight%20line%20connecting%20two%20points,%20and%20you%20calculate%20it%20as%20the%20square%20root%20of%20the%20sum%20of%20the%20squared%20difference%20between%20the%20elements%20of%20two%20vectors.%20%20%20*%20Manhattan%20distance%20(%20TAxi%20cab,%20city%20block)%20%20%20*%20Another%20useful%20measure%20is%20the%20Manhattan%20distance%20%20*%20For%20instance,%20the%20Manhattan%20distance%20between%20points%20(1,2)%20and%20(3,3)%20is%20abs(1–3)%20and%20abs(2–3),%20which%20results%20in%203.%20%20*%20L1%20norm%20of%20two%20vectors%20%20*%20Summing%20the%20absolute%20value%20of%20the%20difference%20between%20the%20elements%20of%20the%20vectors.%20%20%20*%20If%20the%20Euclidean%20distance%20marks%20the%20shortest%20route,%20**the%20Manhattan%20distance%20marks%20the%20longest%20route**,%20resembling%20the%20directions%20of%20a%20taxi%20moving%20in%20a%20city.%20(The%20distance%20is%20also%20known%20as%20taxicab%20or%20city-block%20distance.)%20*%20Chebyshev%20distance%20%20*%20Takes%20the%20maximum%20of%20the%20absolute%20difference%20between%20the%20elements%20of%20the%20vectors.%20%20%20*%20In%20the%20example%20used%20in%20previous%20sections,%20the%20distance%20is%20simply%202,%20the%20max%20between%20abs(1–3)%20and%20abs(2–3).%20%20*%20It%20is%20a%20distance%20measure%20that%20can%20represent%20how%20a%20king%20moves%20in%20the%20game%20of%20chess%20or,%20in%20warehouse%20logistics,%20the%20operations%20required%20by%20an%20overhead%20crane%20to%20move%20a%20crate%20from%20one%20place%20to%20another.%20%20%20*%20In%20machine%20learning,%20the%20Chebyshev%20distance%20can%20prove%20useful%20when%20you%20have%20many%20dimensions%20to%20consider%20and%20most%20of%20them%20are%20just%20irrelevant%20or%20redundant%20(in%20Chebyshev,%20you%20just%20pick%20the%20one%20whose%20absolute%20difference%20is%20the%20largest).%20%20%20###%20P%20-%20argument%20(%20Where%20is%20minkowski%20?%20)-%20minkowski%20uses%20with%20a%20P%20value%20="/>1/p<img src="https://latex.codecogs.com/gif.latex?.-%20minkowski%20with%20a%20P%20value%20=%201%20is%20the%20same%20as%20the%20distance%20as%20**manhattan**.-%20minkowski%20with%20a%20P%20value%20=%202%20is%20the%20same%20as%20the%20distance%20as%20**euclidean**-%20minkowski%20with%20a%20P%20value%20="/>\infty<img src="https://latex.codecogs.com/gif.latex?is%20the%20same%20as%20the%20distance%20as%20**chebyshev**%20%20#%2011.%20Dimensionality%20ReductionTrying%20to%20discover%20a%20pattern,%20consistency%20and%20relationship%20with%20the%20data%20itself.%20**Why?**%20-%20Often%20we%20are%20faced%20with%20large%20data%20sets%20with%20many%20features.%20200-1000%20features.%20-%20Really%20hard%20to%20train%20an%20ML%20algorithm%20with%20so%20many%20features%20-%20It%20takes%20a%20very%20long%20time%20to%20trainSo%20we%20do%20some%20DR%20before%20we%20start%20to%20train%20our%20model.%20Both%20these%20methods%20will%20preserve%20the%20key%20Features&#x2F;Components%20so%20there%20is%20not%20much%20contens%20lost.##%20Comparison%20between%20LDA%20and%20PCA**PCA**%20-%20maximiseing%20the%20componant%20axes%20for%20**varience%20in%20the%20data%20itself.**%20%20%20-%20Good%20for%20data%20without%20classes.**LDA**%20-%20maximiseing%20the%20componant%20axes%20for%20**varience%20between%20class%20sepreation.**%20%20-%20Good%20for%20data%20with%20classes%20in%20mind[Scikit-Learn](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;auto_examples&#x2F;decomposition&#x2F;plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py%20)*%20Principal%20Component%20Analysis%20(PCA)%20%20%20*%20Identifies%20the%20combination%20of%20attributes%20(principal%20components)%20that%20account%20for%20the%20most%20variance%20in%20the%20data.%20%20%20*%20Linear%20Discriminant%20Analysis%20(LDA)%20tries%20to%20identify%20attributes%20(%20features,%20predictors)%20that%20account%20for%20the%20most%20variance%20between%20classes.%20%20%20*%20LDA%20is%20a%20supervised%20method,%20using%20known%20class%20labels.%20##%20Principal%20Component%20Analysis%20(PCA)1901%20by%20Karl%20Pearson%20(Also%20known%20for%20&quot;Pearson%20Correlation&quot;)Used%20in%20exploratory%20data%20analysis%20(EDA).%20*%20Unsupervised%20Machine%20Learning[scikit-learn%20Doc](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;decomposition.html#pca%20)[scikit-learn%20Parameters](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;generated&#x2F;sklearn.decomposition.PCA.html#sklearn.decomposition.PCA%20)%20%20*%20Linear%20dimensionality%20reduction%20using%20Singular%20Value%20Decomposition%20of%20the%20data%20to%20project%20it%20to%20a%20lower%20dimensional%20space.%20%20*%20[Wikipedia](https:&#x2F;&#x2F;en.wikipedia.org&#x2F;wiki&#x2F;Principal_component_analysis%20)%20%20*%20Statistical%20procedure%20that%20utilise%20[orthogonal%20transformation](https:&#x2F;&#x2F;en.wikipedia.org&#x2F;wiki&#x2F;Orthogonal_transformation%20)%20technology%20%20*%20Convert%20possible%20correlated%20features%20(predictors)%20into%20linearly%20uncorrelated%20features%20(predictors)%20called%20**principal%20components**%20%20*%20&#x5C;#%20of%20principal%20components%20&lt;=%20number%20of%20features%20(predictors)%20%20*%20First%20principal%20component%20explains%20the%20largest%20possible%20variance%20%20*%20Each%20subsequent%20component%20has%20the%20next%20highest%20variance,%20subject%20to%20the%20restriction:%20that%20it%20must%20be%20orthogonal%20to%20the%20preceding%20components.%20%20%20%20%20*%20Orthogonal%20-%20of%20or%20involving%20right%20angles;%20at%20right%20angles.%20In%20geometry,%20two%20Euclidean%20vectors%20are%20orthogonal%20if%20they%20are%20perpendicular.%20%20*%20A%20collection%20of%20the%20components%20are%20called%20vectors.%20EigenVectors%20+%20EigenValues%20=%20EigenPairs%20%20*%20Sensitive%20to%20scaling%20%20*%20[Sebastian%20Raschka](http:&#x2F;&#x2F;sebastianraschka.com&#x2F;Articles&#x2F;2014_python_lda.html%20):%20Component%20axes%20that%20maximise%20the%20variance%20%20*%20The%20transformation%20from%20data%20axes%20to%20principal%20axes%20is%20as%20an%20affine%20transformation,%20which%20basically%20means%20it%20is%20composed%20of%20a%20translation,%20rotation,%20and%20uniform%20scaling.**Note:***%20Used%20in%20exploratory%20data%20analysis%20(EDA)%20*%20Visualize%20genetic%20distance%20and%20relatedness%20between%20populations.%20How%20Correlated%20are%20they?*%20Method:%20%20*%20Eigenvalue%20decomposition%20of%20a%20data%20covariance%20(or%20correlation)%20matrix%20%20*%20Singular%20value%20decomposition%20of%20a%20data%20matrix%20(After%20mean%20centering%20&#x2F;%20normalizing%20)%20the%20data%20matrix%20for%20each%20attribute.*%20Output%20%20*%20Component%20scores,%20sometimes%20called%20**factor%20scores**%20(Aka%20The%20transformed%20variable%20values,%20Principle%20Components),%20**loadings**%20(the%20weight)-%20Other%20uses%20of%20PCA%20%20*%20Data%20compression%20and%20information%20preservation%20%20%20*%20Visualization%20%20*%20Noise%20filtering%20%20*%20Feature%20extraction%20and%20engineering##%20Kernel%20PCA*%20Non-linear%20dimensionality%20reduction%20through%20the%20use%20of%20kernels[Scikit%20Learn%20Documentation](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;generated&#x2F;sklearn.decomposition.KernelPCA.html%20)[Scikit%20Learn%20Reference](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;auto_examples&#x2F;decomposition&#x2F;plot_kernel_pca.html%20)This%20is%20inspired%20by%20this%20[scikit%20notebook%20by%20Mathieu%20Blondel%20and%20Andreas%20Mueller](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;auto_examples&#x2F;decomposition&#x2F;plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py%20)##%20Linear%20Discriminant%20Analysis%20(LDA)%20%20%20*%20[Wikipedia](https:&#x2F;&#x2F;en.wikipedia.org&#x2F;wiki&#x2F;Linear_discriminant_analysis%20)%20%20*%20Most%20commonly%20used%20as%20dimensionality%20reduction%20technique%20in%20the%20pre-processing%20step%20for%20pattern-classification%20and%20machine%20learning%20applications.%20%20%20*%20Goal%20is%20to%20project%20a%20dataset%20onto%20a%20lower-dimensional%20space%20with%20good%20class-separability%20in%20order%20avoid%20overfitting%20(“curse%20of%20dimensionality”)%20and%20also%20reduce%20computational%20costs.%20%20*%20Locate%20the%20&#x27;boundaries&#x27;%20around%20clusters%20of%20classes.%20%20%20%20*%20Projects%20data%20points%20on%20a%20line.%20Trying%20ot%20keep%20these%20as%20aseperate%20as%20possible.%20%20%20*%20A%20centroid%20will%20be%20allocated%20to%20each%20cluster%20or%20have%20a%20centroid%20nearby.%20%20*%20[Sebastian%20Raschka](http:&#x2F;&#x2F;sebastianraschka.com&#x2F;Articles&#x2F;2014_python_lda.html%20):%20Maximising%20the%20component%20axes%20for%20class-separation*%20Supervised%20dimensionality%20reduction*%20Project%20the%20input%20data%20to%20a%20linear%20subspace%20consisting%20of%20the%20directions%20which%20maximize%20the%20separation%20between%20classes.%20Takes%20into%20the%20account%20the%20classes%20using%20the%20training%20data.*%20Most%20useful%20in%20a%20multiclass%20setting.*%20Commonly%20used%20in%20Finance[scikit%20learn](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;lda_qda.html#lda-qda%20)[scikit%20learn](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;lda_qda.html#lda-qda%20)###%20Other%20Dimensionality%20Reduction%20Techniques*%20[Multidimensional%20Scaling%20(MDS)%20](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;manifold.html#multi-dimensional-scaling-mds%20)%20%20*%20Seeks%20a%20low-dimensional%20representation%20of%20the%20data%20in%20which%20the%20distances%20respect%20well%20the%20distances%20in%20the%20original%20high-dimensional%20space.*%20[Isomap%20(Isometric%20Mapping)](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;manifold.html#isomap%20)%20%20*%20Seeks%20a%20lower-dimensional%20embedding%20which%20maintains%20geodesic%20distances%20between%20all%20points.*%20[t-distributed%20Stochastic%20Neighbor%20Embedding%20(t-SNE)](https:&#x2F;&#x2F;en.wikipedia.org&#x2F;wiki&#x2F;T-distributed_stochastic_neighbor_embedding%20)%20%20*%20Nonlinear%20dimensionality%20reduction%20technique%20that%20is%20particularly%20well-suited%20for%20embedding%20high-dimensional%20data%20into%20a%20space%20of%20two%20or%20three%20dimensions,%20which%20can%20then%20be%20visualized%20in%20a%20scatter%20plot.%20%20%20*%20Models%20each%20high-dimensional%20object%20by%20a%20two-%20or%20three-dimensional%20point%20in%20such%20a%20way%20that%20similar%20objects%20are%20modeled%20by%20nearby%20points%20and%20dissimilar%20objects%20are%20modeled%20by%20distant%20points.%20dimensional%20space%20(e.g.,%20to%20visualize%20the%20MNIST%20images%20in%202D).###%20Matrix%20Multiplication%20revision:&lt;p%20align=&quot;center&quot;&gt;&lt;img%20src=&quot;https:&#x2F;&#x2F;latex.codecogs.com&#x2F;gif.latex?A=&amp;#x5C;begin{bmatrix}%201.%20&amp;amp;%202.%20&amp;#x5C;&amp;#x5C;%2010.%20&amp;amp;%2020.%20&amp;#x5C;end{bmatrix}&quot;&#x2F;&gt;&lt;&#x2F;p&gt;%20%20&lt;p%20align=&quot;center&quot;&gt;&lt;img%20src=&quot;https:&#x2F;&#x2F;latex.codecogs.com&#x2F;gif.latex?B=&amp;#x5C;begin{bmatrix}%201.%20&amp;amp;%202.%20&amp;#x5C;&amp;#x5C;%20100.%20&amp;amp;%20200.%20&amp;#x5C;end{bmatrix}&quot;&#x2F;&gt;&lt;&#x2F;p&gt;%20%20&lt;p%20align=&quot;center&quot;&gt;&lt;img%20src=&quot;https:&#x2F;&#x2F;latex.codecogs.com&#x2F;gif.latex?A%20&amp;#x5C;times%20B%20%20=%20&amp;#x5C;begin{bmatrix}%201.%20&amp;amp;%202.%20&amp;#x5C;&amp;#x5C;%2010.%20&amp;amp;%2020.%20&amp;#x5C;end{bmatrix}&amp;#x5C;times%20&amp;#x5C;begin{bmatrix}%201.%20&amp;amp;%202.%20&amp;#x5C;&amp;#x5C;%20100.%20&amp;amp;%20200.%20&amp;#x5C;end{bmatrix}%20=%20&amp;#x5C;begin{bmatrix}%20201.%20&amp;amp;%20402.%20&amp;#x5C;&amp;#x5C;%202010.%20&amp;amp;%204020.%20&amp;#x5C;end{bmatrix}&quot;&#x2F;&gt;&lt;&#x2F;p&gt;%20%20By%20parts:&lt;p%20align=&quot;center&quot;&gt;&lt;img%20src=&quot;https:&#x2F;&#x2F;latex.codecogs.com&#x2F;gif.latex?A%20&amp;#x5C;times%20B%20=%20&amp;#x5C;begin{bmatrix}%201.%20&amp;#x5C;times%201.%20+%202.%20%20&amp;#x5C;times%20100.%20&amp;amp;%20%201.%20&amp;#x5C;times%202.%20+%202.%20&amp;#x5C;times%20200.%20&amp;#x5C;&amp;#x5C;%201.%20%20&amp;#x5C;times%201.%20+%2020.%20&amp;#x5C;times%20100.%20&amp;amp;%2010.%20&amp;#x5C;times%202.%20+%2020.%20&amp;#x5C;times%20200.%20&amp;#x5C;end{bmatrix}&quot;&#x2F;&gt;&lt;&#x2F;p&gt;%20%20&lt;p%20align=&quot;center&quot;&gt;&lt;img%20src=&quot;https:&#x2F;&#x2F;latex.codecogs.com&#x2F;gif.latex?A%20&amp;#x5C;times%20B%20%20=%20&amp;#x5C;begin{bmatrix}%20a%20&amp;amp;%20b%20&amp;#x5C;&amp;#x5C;%20c%20&amp;amp;%20d%20&amp;#x5C;end{bmatrix}&amp;#x5C;times%20&amp;#x5C;begin{bmatrix}%201%20&amp;amp;%202%20&amp;#x5C;&amp;#x5C;%203%20&amp;amp;%204%20&amp;#x5C;end{bmatrix}%20=%20&amp;#x5C;begin{bmatrix}%20(a&amp;#x5C;times1)+%20(1&amp;#x5C;times3)%20&amp;amp;%20(a&amp;#x5C;times1)+%20(2&amp;#x5C;times4)%20&amp;#x5C;&amp;#x5C;%20(c&amp;#x5C;times%20d)+%20(1&amp;#x5C;times3)%20&amp;amp;%20(c&amp;#x5C;times%20d)+%20(2&amp;#x5C;times4)%20&amp;#x5C;end{bmatrix}&quot;&#x2F;&gt;&lt;&#x2F;p&gt;%20%20In%20python%20```pyimport%20numpy%20as%20npA%20=%20[[1.,%202.],%20[10.,%2020.]]B%20=%20[[1.,%202.],%20[100.,%20200.]]np.dot(A,%20B)```Returns%20```pyarray([[%20201.,%20%20402.],%20%20%20%20%20%20%20[2010.,%204020.]])```#%2012%20.%20Unsupervised%20Learning%20Cluster.%20[Wikipedia](https:&#x2F;&#x2F;en.wikipedia.org&#x2F;wiki&#x2F;Cluster_analysis%20)*%20The%20task%20of%20grouping%20a%20set%20of%20objects%20in%20such%20a%20way%20that%20objects%20in%20the%20same%20group%20(called%20a%20cluster)%20are%20more%20similar%20(in%20some%20sense%20or%20another)%20to%20each%20other%20than%20to%20those%20in%20other%20groups%20(clusters).%20*%20With%20Supervised%20learning%20we%20do%20not%20have%20a%20target%20that%20we%20can%20learn%20from.%20With%20Supervised,%20%20There%20are%20features%20that%20go%20along%20with%20the%20class.%20ML%20is%20powerful%20here%20as%20it%20uncovers%20patterns%20in%20the%20data.Examples%20where%20its%20used%20that%20have%20no%20target:*%20Natural%20Language%20Processing%20(NLP)%20-%20Getting%20the%20sentiment.*%20Computer%20Vision*%20Stock%20markets*%20Customer%20&#x2F;%20Market%20Segmentation*%20Customer%20Churn*%20fraud%20detection##%20how%20do%20you%20cluster?%20###%204%20of%20the%20many%20types%20:###%20Connectivity-based%20clustering*%20Distance%20based.%20How%20close%20are%20they%20to%20each%20other*%20E.g.,%20Hierarchical%20clustering%20-%20based%20on%20an%20object%20related%20to%20another%20object%20that%20is%20close%20by.%20%20*%20if%20you%20live%20in%20this%20neighbourhood%20you%20are%20more%20like%20to%20be%20like%20the%20people%20here%20than%20in%20another%20neighborhood.%20*%20Distances%20will%20be%20represented%20by%20Dendrogram###%20Centroid-based%20clustering*%20Represents%20each%20cluster%20by%20a%20single%20mean%20vector.%20Trying%20to%20find%20the%20average%20in%20a%20cluster.*%20E.g.,%20k-means%20Clustering%20algorithm*%201%20catch%20with%20k-means:%20You%20do%20need%20to%20specify%20the%20number%20of%20clusters.%20So%20is%20it%20really%20supervised?%20###%20Distribution-based%20clustering*%20Modeled%20using%20statistical%20distributions*%20E.g.,%20Multivariate%20normal%20distributions%20used%20by%20the%20expectation-maximization%20algorithm.###%20Density-based%20clustering*%20Defines%20clusters%20as%20connected%20dense%20regions%20in%20the%20data%20space.*%20E.g.,%20DBSCAN[MLXTEND](http:&#x2F;&#x2F;rasbt.github.io&#x2F;mlxtend&#x2F;%20)%20is%20an%20extra%20python%20library%20supported%20by%20Sebastien%20Raschka.%20#%20Ward’s%20Agglomerative%20Hierarchical%20ClusteringAgglomeration:%20a%20large%20group%20of%20many%20different%20things%20collected%20or%20brought%20together:%20[Wikipedia](https:&#x2F;&#x2F;en.wikipedia.org&#x2F;wiki&#x2F;Hierarchical_clustering%20)Wards&#x27;%20can%20work%20in%20two%20different%20ways:*%20Agglomerative:%20%20%20*%20Bottom%20up%20%20*%20Each%20observation%20starts%20in%20its%20own%20cluster,%20and%20pairs%20of%20clusters%20are%20merged%20as%20one%20moves%20up%20the%20hierarchy.*%20Divisive:%20%20%20*%20Top%20down%20%20*%20All%20observations%20start%20in%20one%20cluster,%20and%20splits%20(%20like%20a%20Decision%20Tree)%20are%20performed%20recursively%20as%20one%20moves%20down%20the%20hierarchy.So%20how%20does%20it%20actually%20do%20it%20?%20IT%20uses%20a%20distance%20matrix.%20Similar%20(but%20not%20exactly%20the%20same)%20to%20K%20NEarest%20Neighbour.-%20euclidean,%20Manahattan,%20Mahalanobis%20%20[Stackexchange](https:&#x2F;&#x2F;stats.stackexchange.com&#x2F;questions&#x2F;195446&#x2F;choosing-the-right-linkage-method-for-hierarchical-clustering%20)[CMU%20Notes](http:&#x2F;&#x2F;www.stat.cmu.edu&#x2F;~ryantibs&#x2F;datamining&#x2F;lectures&#x2F;05-clus2-marked.pdf%20)[PSE%20Stat505%20Linkage%20Methods](https:&#x2F;&#x2F;onlinecourses.science.psu.edu&#x2F;stat505&#x2F;node&#x2F;143%20):Linkage%20Criteria%201.%20Single%20Linkage:%20shortest%20distance.%20Distance%20between%20two%20clusters%20to%20be%20the%20**minimum%20distance%20between%20any%20single%20data%20point%20in%20the%20first%20cluster%20and%20any%20single%20data%20point%20in%20the%20second%20cluster**.%202.%20Complete%20Linkage:%20Furthest%20distance.%20Distance%20between%20two%20clusters%20to%20be%20the%20**maximum%20distance%20between%20any%20single%20data%20point%20in%20the%20first%20cluster%20and%20any%20single%20data%20point%20in%20the%20second%20cluster**.3.%20Average%20Linkage:%20Average%20of%20all%20pairwise%20links.4.%20Centroid%20Method:%20Distance%20between%20two%20clusters%20is%20the%20**distance%20between%20the%20two%20mean%20vectors%20of%20the%20clusters**.%20These%20may%20not%20exist%20in%20the%20data%20but%20are%20calculated.5.%20Ward’s%20Method:%20ANOVA%20based%20approach.%20%20%20%20%20*%20Iterative%20process%20%20%20%20*%20Minimises%20the%20total%20within%20cluster%20variance.%20So%20it%20will%20calculate%20within%20a%20clusters%20variance.%20%20%20%20%20*%20At%20each%20step,%20the%20pair%20of%20clusters%20with%20minimum%20between%20cluster%20distance%20are%20merged###%20Retrieve%20the%20ClustersBy%20Retrieve%20we%20mean%20labeling%20the%20observations%20with%20the%20class%20somehow.%20The%20scipy%20library%20fcluster%20is%20the%20recomendation.%20*%20Utilise%20the%20[fcluster](https:&#x2F;&#x2F;docs.scipy.org&#x2F;doc&#x2F;scipy&#x2F;reference&#x2F;generated&#x2F;scipy.cluster.hierarchy.fcluster.html%20)%20function.*%20Retrieve%20by%20distance%20or%20number%20of%20clusters***Becasue%20we%20are%20wrking%20with%20data%20that%20has%20no%20target%20or%20label,%20we%20beed%20multiple%20methods%20to%20confirm%20eahc%20other,%20and%20investigate%20those%20that%20say%20otherwise.%20If%20so%20then%20we%20may%20need%20to%20do%20further%20explorations.#%20k-Means%20Clustering*%20Unsupervised%20method%20that%20Analyses%20and%20find%20patterns%20&#x2F;%20clusters%20within%20data*%20Distance%20measures[scikit%20learn](http:&#x2F;&#x2F;scikit-learn.org&#x2F;stable&#x2F;modules&#x2F;clustering.html#k-means%20)*%20Clusters%20data%20by%20trying%20to%20separate%20samples%20in%20n%20groups%20of%20equal%20variance%20eg"/>Sd^2<img src="https://latex.codecogs.com/gif.latex?*%20Minimizing%20a%20criterion%20known%20as%20the%20&quot;inertia&quot;%20or%20&quot;within-cluster%20sum-of-squares&quot;.%20Trying%20to%20minimize%20the%20seperate%20groups%20of%20equal%20variance.*%20Requires%20the%20number%20of%20clusters%20to%20be%20specified.%20*%20Scales%20wellHow%20does%20it%20work?*%20Divides%20a%20set%20of%20samples%20into%20disjoint%20clusters.%20Before%20hand%20you%20neeed%20to%20give%20it%20a%20target%20of%20clusters%20to%20work%20towards.*%20Each%20described%20by%20the%20mean%20of%20the%20samples%20in%20the%20cluster.%20*%20The%20means%20are%20commonly%20called%20the%20cluster%20“centroids”*%20Note%20that%20the%20centroids%20are%20not,%20in%20general,%20points%20from,%20although%20they%20live%20in%20the%20same%20space.%20*%20The%20K-means%20algorithm%20aims%20to%20choose%20centroids%20that%20minimise%20the%20inertia,%20or%20**within-cluster%20sum%20of%20squared%20criterion**.%20Once%20it%20has%20identified%20the%20clusters,%20it%20tries%20to%20find%20a%20point%20which%20minimises%20the%20Sums%20of%20Square(%20deviation%20from%20the%20centroid;%20like%20Linear%20regression)Some%20Challenges%20k-Means%20Clustering%20:*%20The%20globally%20optimal%20result%20may%20not%20be%20achieved*%20The%20number%20of%20clusters%20must%20be%20selected%20beforehand*%20k-means%20is%20limited%20to%20linear%20cluster%20boundaries.%20To%20resolve%20this%20you%20could%20use%20PCA%20to%20get%20down%20to%202%20PCA*%20k-means%20can%20be%20slow%20for%20large%20numbers%20of%20samples.##%20Elbow%20MethodBecasue%20we%20are%20workign%20with%20an%20Unsupervised%20methods%20we%20don&#x27;t%20have%20an%20answer%20,%20a%20Y.%20*%20Use%20intrinsic%20metrics%20-%20another%20name%20for%20&quot;within-cluster%20Sums%20of%20Squared%20Error&quot;*%20An%20example%20fo%20this%20is%20the%20**within-cluster%20Sums%20of%20Squared%20Error**%20*%20scikit%20learn%20has%20already%20provided%20it%20via%20the%20kmeans%20`inertia_`%20attribute.%20So%20it%20will%20do%20the%20heavy%20lifting.When%20we%20plot%20this%20on%20a%20line%20graph,%20there%20will%20be%20an%20elbow%20where%20the%20line%20bends%20sharply.%20This%20indicates%20where%20we%20can%20stop%20clustering%20because%20the%20**within-cluster%20Sums%20of%20Squared%20Error**%20is%20not%20giving%20us%20any%20more%20reduction.#%20Silhouette%20Analysis%20MethodThis%20is%20an%20alternative%20to%20the%20Elbow%20method.%20&lt;p%20align=&quot;center&quot;&gt;&lt;img%20src=&quot;https:&#x2F;&#x2F;latex.codecogs.com&#x2F;gif.latex?&amp;#x5C;text{silhouette%20score}=&amp;#x5C;frac{p-q}{max(p,q)}&quot;&#x2F;&gt;&lt;&#x2F;p&gt;"/>p<img src="https://latex.codecogs.com/gif.latex?is%20the%20mean%20distance%20to%20the%20points%20in%20the%20nearest%20cluster%20that%20the%20data%20point%20is%20not%20a%20part%20of.The%20middle%20of%20one%20centroid%20to%20another."/>q<img src="https://latex.codecogs.com/gif.latex?is%20the%20mean%20intra-cluster%20distance%20to%20all%20the%20points%20in%20its%20own%20cluster.%20Intra-cluster%20means%20within%20the%20cluster,%20so%20the%20mean%20distance%20of%20all%20the%20distances%20within%20one%20cluster.So"/>p-q<img src="https://latex.codecogs.com/gif.latex?will%20contrast%20the%20`p`%20inter%20cluster%20and%20the%20`q`%20intra-cluster.%20You%20want%20to%20minimsie%20p%20and%20maximise%20q%20so%20the%20top%20value%20of%20the%20fraction%20is%20as%20big%20as%20possible.The"/>max(p,q)$ is to standardise it and get it closest to 1. 
  
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
  
  
[wikipedia](https://en.wikipedia.org/wiki/Mean_shift )
  
* The basic intuition is [Non-parametric](https://en.wikipedia.org/wiki/Nonparametric_statistics )
  
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
 * [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis )
 * [Sebastian Raschka](http://sebastianraschka.com/Articles/2014_python_lda.html )
  
  
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
[Orange data mining visulisation software](https://orange.biolab.si/ ) - Looks really good 
  