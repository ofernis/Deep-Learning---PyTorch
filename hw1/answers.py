r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. ***False***
The test set allows us to estimate the out-of-sample loss (the theoretical loss).
The train set allows us to calculate the in-sample loss.

2. ***False***
First of all, for technical reasons: if we make the train set too small in the split, it won't be useful.
Also, if one of the classes has a very low frequency (for example - a rare disease we are trying to detect), then some of the splits won't contain samples with the class's label. thus, won't be useful.

3. **True**
Using the test set is forbidden! this can cause a _data-leakage_ - where the model is learning from the test set, making the test performance invalid.
The test set should only be used after the model is fully trained.

4. **False**
We use the test set performance as the proxy, not the validation set performance.

"""

part1_q2 = r"""
**Your answer:**

Our friend's approach is **not justified**.
This approach is manually creating a _data-leakage_.
Hyper-parameter tuning should be performed with a validation set, and not with the test set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Increasing _k_ should improve the model's test accuracy at first, but when we cross a certain threshold the accuracy will start to drop again.
This is because for k=1 we get an over-fit model, which classify a sample by only it's closest neighbor and for $k\rightarrow\inf$ we get an under-fit model which decides an arbitrary decision rule by the majority label in the train set. 

"""

part2_q2 = r"""
**Your answer:**


1. The kNN model with the highest train-set accuracy will always be 1NN because each sample is his own nearest neighbor.
2. Selecting the model based on it's test-set accuracy is also a _data leakage_ as we saw before in part 1.

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
