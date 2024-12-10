---
layout: post
title: ML in Python Part 1 - Classification with Scikit-learn
date:   2023-10-23 12:00:00
description: Getting started with machine learning using Scikit-learn's classification tools
---

Have you ever wondered how to get started with machine learning? This series of posts will guide you through practical implementations using two of Python's most popular frameworks: Scikit-learn and TensorFlow. Whether you're a beginner looking to understand the basics or an experienced developer wanting to refresh your knowledge, we'll progress from basic classification tasks to more advanced regression problems.

The series consists of four parts:

1. **[Getting Started with Classification using Scikit-learn]({{ site.baseurl }}/blog/2023/01_scikit_simple)** (You are here)<br>Introduction to machine learning basics using the MNIST dataset
2. **[Basic Neural Networks with TensorFlow]({{ site.baseurl }}/blog/2023/02_tensorflow_simple)** (Part 2)<br>Building your first neural network for image classification
3. **[Advanced Machine Learning with Scikit-learn]({{ site.baseurl }}/blog/2023/03_scikit_advanced)** (Part 3)<br>Exploring complex regression problems and model optimization
4. **[Advanced Neural Networks with TensorFlow]({{ site.baseurl }}/blog/2023/04_tensorflow_advanced)** (Part 4)<br>Implementing sophisticated neural network architectures

## Why These Tools?

[Scikit-learn](https://scikit-learn.org/stable/) is Python's most popular machine learning library for a reason. It provides:
- A consistent interface across different algorithms
- Extensive preprocessing capabilities
- Built-in model evaluation tools
- Excellent documentation and community support

[TensorFlow](https://www.tensorflow.org/) complements Scikit-learn by offering:
- Deep learning capabilities
- GPU acceleration
- Flexible model architecture design
- Production-ready deployment options

In this first post, we'll start with Scikit-learn and implement a basic classification task using the MNIST dataset. This will establish fundamental concepts that we'll build upon in later posts.

```python
# Standard scientific Python imports
import numpy as np
import matplotlib.pyplot as plt
```

## 1. Load dataset

For our first machine learning task, we'll use the famous MNIST dataset - a collection of handwritten digits that serves as a perfect introduction to image classification.  The MNIST dataset has become the "Hello World" of machine learning for good reason:
- Simple to understand (handwritten digits from 0-9)
- Small enough to train quickly
- Complex enough to demonstrate real ML concepts
- Perfect for learning classification basics

Let's start by loading and exploring this dataset:

```python
# Load dataset
from sklearn import datasets

digits = datasets.load_digits()

# Extract feature matrix X and target vector y
X = digits['data']
y = digits['target']

print(f"Dimension of X: {X.shape}\nDimension of y: {y.shape}")
```

    Dimension of X: (1797, 64)
    Dimension of y: (1797,)

Each of our 1,797 samples contains 64 features, representing an 8 x 8 pixel grid of an image. Let's reshape these features into their original pixel grid format for visualization.

```python
_, axes = plt.subplots(nrows=3, ncols=8, figsize=(8, 4))
for ax, image, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title("Label: %i" % label)
    ax.set_axis_off()
```

<div style="text-align: center">
    <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_5_0.png" data-zoomable width=600px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">
</div><br>

## 2. Split data into train and test set

Next, we need to perform a train/test split so that we can validate the final performance of our trained model.
For the train/test split, we will use a 80:20 ratio. Furthermore, we will use the `stratify` parameter to
ensure that the class distribution in the train and test set is preserved.

```python
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
```

## 3. Train model

For our first classification attempt, we'll use a RandomForestClassifier. While there are many algorithms to choose from, Random Forests are an excellent starting point because they:
- Handle both numerical and categorical data naturally
- Require minimal preprocessing
- Provide insights into feature importance
- Are relatively robust against overfitting
- Perform well even with default parameters

```python
from sklearn.ensemble import RandomForestClassifier

# Define type of classifier
clf = RandomForestClassifier()

# Train classifier on training data
clf.fit(X_tr, y_tr)

# Evaluate model performance on training and test set
score_tr = clf.score(X_tr, y_tr)
score_te = clf.score(X_te, y_te)

print(
    f"Prediction accuracy on train data: {score_tr*100:.2f}%\n\
Prediction accuracy on test data:  {score_te*100:.2f}%"
)
```

    Prediction accuracy on train data: 100.00%
    Prediction accuracy on test data:  96.67%

As you can see, the model performed perfectly on the training set. No wonder, we tested the classifiers
performance on the same data it was trained on. But is there way how we can improve the score on the test data?

Yes there is. But for this we need to fine-tune our random forest classifier. Because as of now we only used
the classifier with it's default parameters.

## 4. Fine-tune model

To fine-tune our classifier model we need to split our dataset into a third part, the so called validation set.
In short, the **training set** is used to train the parameter of a model, the **validation set** is used to
fine-tune the hyperparameter of a model, and the **test set** is used to see how well the fine-tuned model
generalizes on never before seen data.

A common practise for this train/validation split is the usage of a **cross-validation**. In short, the
training data is iteratively split into training and validation set, where each split (or fold) is used once
as the validation set. For an in-depth explanation on this, see
[this helpful resource](https://scikit-learn.org/stable/modules/cross_validation.html).

Now, we also mentioned fine-tuning our model. One way to do this, is to perform a **grid search**, i.e. running
the model with multiple parameter combinations and than deciding which ones work best.

Luckily, `scikit-learn` provides a neat routine that combines the cross-validation with the grid search, called
[`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
So let's go ahead and set everything up.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
parameters = {'max_depth': [5, 25, 50], 'n_estimators': [1, 10, 50, 200]}

# Put parameter grid and classifier model into GridSearchCV
grid = GridSearchCV(clf, parameters, cv=5)

# Train classifier on training data
grid.fit(X_tr, y_tr)

# Evaluate model performance on training and test set
score_tr = grid.score(X_tr, y_tr)
score_te = grid.score(X_te, y_te)

print(
    f"Prediction accuracy on train data: {score_tr*100:.2f}%\n\
Prediction accuracy on test data:  {score_te*100:.2f}%"
)
```

    Prediction accuracy on train data: 100.00%
    Prediction accuracy on test data:  97.22%

Great, our score on the test set has improved. So let's see which parameter combination seems to be the best.

```python
# Show best random forest classifier
best_rf = grid.best_estimator_
best_rf
```

    RandomForestClassifier(max_depth=25, n_estimators=200)

Now, to better understand how the different parameters relate to model performance, let's plot the `max_depth`
and `n_estimators` with respect to the accuracy performance on the validation set.

```python
# Put insights from cross-validation grid search into pandas dataframe
import pandas as pd

df_res = pd.DataFrame(grid.cv_results_)
df_res = df_res.iloc[:, df_res.columns.str.contains('mean_test_score|param_')]
df_res = df_res.astype('float')

# Plot results in table (works only when we investigate two hyper-parameters).
import seaborn as sns

result_table = df_res.pivot(
    index='param_max_depth', columns='param_n_estimators', values='mean_test_score'
)
sns.heatmap(100 * result_table, annot=True, fmt='.2f', square=True, cbar=False)
plt.title("Accuracy on validation set, based on model hyper-parameter")
plt.show()
```

<div style="text-align: center">
    <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_16_0.png" data-zoomable width=500px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">
</div><br>

## 5. Change model

The great thing about `scikit-learn` is that the framework is very dynamic. The only thing we need to change to
do the same classification with a Support Vector Machine (SVM) for example, is changing the model and the
parameter grid we want to explore.

```python
# Create support vector classifier object
from sklearn.svm import SVC

clf = SVC(kernel='rbf')

# Define parameter grid
parameters = {'gamma': np.logspace(-5, -1, 5), 'C': np.logspace(-2, 2, 5)}
```

That's it! The rest can be used as before.

```python
# Put parameter grid and classifier model into GridSearchCV
grid = GridSearchCV(clf, parameters, cv=5)

# Train classifier on training data
grid.fit(X_tr, y_tr)

# Evaluate model performance on training and test set
score_tr = grid.score(X_tr, y_tr)
score_te = grid.score(X_te, y_te)

print(
    f"Prediction accuracy on train data: {score_tr*100:.2f}%\n\
Prediction accuracy on test data:  {score_te*100:.2f}%"
)
```

    Prediction accuracy on train data: 100.00%
    Prediction accuracy on test data:  98.89%

Nice, this is much better. It seems for this particular dataset, with the hyper-parameter's we explored, SVM
is a better model type.

As before, let's take a look at the model with the best parameters.

```python
# Show best SVM classifier
best_svm = grid.best_estimator_
best_svm
```

    SVC(C=10.0, gamma=0.001)

And once more, how do these two hyper-parameters relate to the performance metric `accuracy` in the validation
set?

```python
# Put insights from cross-validation grid search into pandas dataframe
import pandas as pd

df_res = pd.DataFrame(grid.cv_results_)
df_res = df_res.iloc[:, df_res.columns.str.contains('mean_test_score|param_')]
df_res = df_res.astype('float')

# Plot results in table (works only when we investigate two hyper-parameters).
import seaborn as sns

result_table = df_res.pivot(
    index='param_gamma', columns='param_C', values='mean_test_score'
)
sns.heatmap(100 * result_table, annot=True, fmt='.2f', square=True)
plt.title("Accuracy on validation set, based on model hyper-parameter")
plt.show()
```

<div style="text-align: center">
    <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_24_0.png" data-zoomable width=500px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">
</div><br>

## 6. Post-model investigation

Last but certainly not least, let's investigate the prediction quality of our classifier. Two great routines
that you can use for that are `scikit-learn`s
[`classification_report`](
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) and
[`confusion_matrix`](
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

```python
# Predict class predictions on the test set
y_pred = best_svm.predict(X_te)
```

```python
# Print classification report
from sklearn.metrics import classification_report

print(classification_report(y_te, y_pred))
```

                  precision    recall  f1-score   support
               0       1.00      1.00      1.00        36
               1       1.00      1.00      1.00        36
               2       1.00      1.00      1.00        35
               3       1.00      1.00      1.00        37
               4       1.00      1.00      1.00        36
               5       0.97      0.97      0.97        37
               6       1.00      1.00      1.00        36
               7       0.97      1.00      0.99        36
               8       0.97      1.00      0.99        35
               9       0.97      0.92      0.94        36
        accuracy                           0.99       360
       macro avg       0.99      0.99      0.99       360
    weighted avg       0.99      0.99      0.99       360

As you can see, while the scores are comparable between classes, some clearly are harder to detect than others.
To help better understand which target classes are confused more often than others, we can look at the
confusion matrix.

```python
# Compute confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(pd.DataFrame(cm), annot=True, cbar=False, square=True)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
```

<div style="text-align: center">
    <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_29_0.png" data-zoomable width=500px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">
</div><br>

## 7. Additional model and results investigations

Depending on the classifier model you chose, you can investigate many additional things, once your model is
trained.

For example, `RandomForest` model provide a `feature_importances_` attribute that allows you to investigate
which of your features is helping the most with the classification task.

```python
# Collect feature importances from RF model
feat_import = best_rf.feature_importances_

# Putting the 64 feature importance values back into 8x8 pixel grid
feature_importance_image = feat_import.reshape(8, 8)

# Visualize the feature importance grid
plt.figure(figsize=(5, 5))
plt.imshow(feature_importance_image)
plt.show()
```

<div style="text-align: center">
    <img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_31_0.png" data-zoomable width=500px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">
</div><br>

As you can see, feature in the center of the 8x8 grid seem to be more important for the classification task.

Other interesting post-modeling tasks could be the investigation of the prediction probabilities per sample.
For example: What do images look like of digits with 100% prediction probability?

```python
# Compute prediction probabilities
y_prob = best_rf.predict_proba(X_te)

# Extract prediction probabilities of target class
target_prob = [e[i] for e, i in zip(y_prob, y_te)]

# Plot images of easiest to predict samples
_, axes = plt.subplots(nrows=3, ncols=20, figsize=(9, 1.5))
for ax, idx in zip(axes.ravel(), np.argsort(target_prob)[::-1]):
    ax.imshow(X_te[idx].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_axis_off()
```

<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_33_0.png" data-zoomable width=800px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">

And what about the difficult cases? For which digits does the model strugle the most to get above chance level?

```python
# Plot images of easiest to predict samples
_, axes = plt.subplots(nrows=3, ncols=20, figsize=(9, 1.5))
for ax, idx in zip(axes.ravel(), np.argsort(target_prob)):
    ax.imshow(X_te[idx].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_axis_off()
```

<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/ex_plots/ex_01_scikit_simple_output_35_0.png" data-zoomable width=800px style="padding-top: 20px; padding-right: 20px; padding-bottom: 20px; padding-left: 20px">

## Summary and Next Steps

In this first tutorial, we've covered the fundamentals of machine learning with Scikit-learn:
- Loading and visualizing data
- Splitting data into training and test sets
- Training a basic classifier
- Fine-tuning model parameters
- Evaluating model performance

We've seen how Scikit-learn's consistent API makes it easy to experiment with different algorithms and preprocessing techniques. The RandomForest classifier achieved 97.22% accuracy, while the SVM performed even better at 98.89%.

In the next post, we'll tackle the same MNIST classification problem using TensorFlow, introducing neural networks and deep learning concepts. This will help you understand the differences between classical machine learning approaches and deep learning, and when to use each.

Key takeaways:
1. Even simple models can achieve good performance on well-structured problems
2. Start with simple models and gradually increase complexity
3. Cross-validation is crucial for reliable performance estimation
4. Grid search helps find optimal parameters systematically
5. Always keep a separate test set for final evaluation
6. Look beyond accuracy to understand model performance

In Part 2, we'll explore how neural networks approach the same problem using TensorFlow, introducing deep learning concepts and comparing the two approaches.

[Continue to Part 2: Basic Neural Networks with TensorFlow →]({{ site.baseurl }}/blog/2023/02_tensorflow_simple)
