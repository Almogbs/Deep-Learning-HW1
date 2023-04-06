r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. False - We estimate our in-sample error using the samples we train our model with, usually the training set and not the test set.
2. False - We can split the data to train and test set when the train set doesnt represent currectly the data. e.g. we have the dataset S and each sample $$ X \in S $$ is labeled with $$ y \in \{0, 1\}$$ and we split S into S_train and S_test where S_train contain only samples which are labels 0, and the rest are in S_test. This way, our model is not trained with important part of the data and probably result poor performance.
3. True - We want to use the train and validation set in the cross-validation process in order to increase the accuarcy of the predictain on the test-set, so if we use the test-set in the cv, we are training model using the data we will use to evaluate it. 
4. True - When performing cross-validation, in each split, we train our model with all the folds except one, which is the validation set. Our model is not familier with the data in this validation-set and just like the test-set is a proxy for the model's generalization error because it contain samples that the model is not familier with.


"""

part1_q2 = r"""
**Your answer:**

My "friend's" approach is not justified. The model's training process, including hyperparameter tuning shoud be done without "feedback", from our test-set, which represent the model's generalization error. He should have use a validation set instead.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Increasing k improves our generalization for unseen data up to the point when k=3, because k=1 is too prone to overfitting by labeling new samples according to outlayers near the new sample.
After that, our model take into an account further sampels which are irrelevant, and eventually leads to underfitting.

"""

part2_q2 = r"""
**Your answer:**


Explain why (i.e. in what sense) using k-fold CV, as detailed above, is better than:
1. Using k-fold CV is better than training on the entire train-set with various models and selecting the best model with respect to train-set accuracy because on k-fold we are testing our model on unseen data aka validation-set and get a validation-error, which is, in fact, a proxy for the model's generalization error, when on the suggested method, we evaluating our model by the data it was trained with (known data).

2. Using k-fold CV is better than training on the entire train-set with various models and selecting the best model with respect to test-set accuracy because on the suggested method, we using the test-set for tuning, and in a sense, training our model with the test-set, thing that conflict with the fact that the test-set error functions as the model's eneralization error, thing that does not happen in the k-fold method.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The selection of $\Delta $ is arbitrary for the SVM loss $L(\mat{W})$ because of the weights.
For each selection of delta, we could inflate/deflate the values of the weights in order to get the same results.

"""

part3_q2 = r"""
**Your answer:**

1. The linear model is learning by giving bigger wieght to pixels (i.e. certain incdices of the sample vector) that correspondes with the digits in which they usually present. For example, the pixels in the middle of the image are usually black when presenting the digit "0", so the weight that corresponeds with that pixel will be updated in a way that it will increase the score of labeling "0", if those pixelds are black.
We can explain some of the classification error using this explanations above, for example, the
miss predictaions between "7" and "9", in which the model "selecting" the pixels that are usually set to white/black when it saw previus images of "7" or "9". the problem is that that the pixels that are used for "7", are also being used for "9" and thats leads to the errors.

2. Both of the models are giving predictions based on individual pixel-wise similarity.
When the difference between the models are that the KNN will calculate similarity based on all of the pixels with the same weight, but the linear model will select pixels that are "important" by giving them such a weight.
Another difference can be the fact the in KNN we look only at the K nearest samples when on the linear we look at all of the samples.

"""

part3_q3 = r"""
**Your answer:**

1. The learning rate we chose is good choice - lr = 0.02, based on the accuracy we recieved and the fact that for lower lr values, the model is "taking" smaller steps towards the right diractions, which results of the training to finish (under the constrains of epochs = 30) while we still making a progress and the graph would look like it is still getting higher and higher in each epoch while the loss graph we get lower and lower.
From the other hand, the the lr is bigger that the lr we chose, we might take a steps that are too big toward the goal and this will result lower accuracy since we miss the target in each step and the graph would "zig-zag" between good and bad accuracy in each epoch and so does the loass graph.

2. Based on the graphs, we could say that our model is "Slightly overfitted to the training set" because the train accuracy is higher that the validation accuracy in each epoch.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
