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

The selection of $\Delta$ is arbitrary because it only affects the margin between the correct score and the incorrect scores.
Thus, increasing or decreasing $\Delta$ will only shift the margin but will not affect the location of the minimum of the loss function.

"""

part3_q2 = r"""
**Your answer:**


1. It seems like the SVM model is learning the general (or average) shape of each class (digit).
We can see in the images that the bright area (corresponding to the higher weights) are formulating a similar shape (in most cases) to the digit shape.
_This is espacially clear for digits '0', '1' and '3'._

2. The similarity is that both models are comparing the new digit shape to the old ones (the train set).
The difference is that kNN is comparing the new digit to its closest neigbors (which can be interpeted as the most similar shapes) and then labels the new digit by the majority.
While the SVM model is learning an "average" digit for each digit 0-9 then comparing the new digit to these averages and selecting the most similar one.

"""

part3_q3 = r"""
**Your answer:**


1. Based on the graph, we would say that the learning rate we chose ($0.05$) is *Good*.
That's because the graph is showing a "nice" convergence to the minimum - the convergence is overall smooth and showing a steady decrease towadrs the minimum.
- For *too high* learning rate, we would see big fluctuations on the graph, or even a divergence.
- For *too low* learning rate, the learning curve would be smoother but it won't be able to reach the minimum at the given number of epochs.
  
2. Based on the graph, we would say that the model is *Slightly overfitted to the training set*.
We can see that there is a small gap between the train curve and the validation curve in the graph,
the train curve is slightly higher than the validation one, indicating that the model is not generalizing well.

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
