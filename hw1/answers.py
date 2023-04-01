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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
