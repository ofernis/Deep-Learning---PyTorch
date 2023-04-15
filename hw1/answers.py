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

$\Delta$ is an hyperparameter that determines the minimum margin between the scores of the correct class and the scores of any incorrect class.
thus, the choice of $\Delta$ is arbitrary because $\lambda$ can also control the margin size (generally, higher $\lambda$ = smaller margins) and balances the effect of $\Delta$.

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


The ideal pattern to see in a residual plot is a flat line across the x axis.
This is because that line means that all the residuals are 0.

Comparing the 2 plots:
| Plot                | MSE      | R2       |
|-------------------- |----------|----------|
| top-5               | 27.20    | 0.68     |
| train after CV      | 7.04     | 0.92     |
| test after CV       | 18.11    | 0.74     |

In the plots, we can see that we have much less outliers outside the area of low residual error.
Also, inside the low error zone we can see that the sample's error are relativly closer to 0 than before.
On top of that, we can empirically see in the table above that we improved our MSE and R2 scores for both sets compared to the top-5 train set.

"""

part4_q2 = r"""
**Your answer:**


1. This is still a linear regression. We transformed the features but the regression itself is still the same linear regression.
(The regression line is still an hyperplane, but the number of dimensions has changed).

2. In introduction to ML, we learned about the RBF kernel, which is a feature tranformation we can apply that is very expressive and can fit any non-linear function
(but with the drawback of tendency to overfit).

3. The decision boundry will still be an hyperplane, but it will be an hyperplane in the dimension of the tansformed features, and not the original ones.
Thus, resulting a non-linear boundry in the original feature's plane.

"""

part4_q3 = r"""
**Your answer:**

1. `np.logspace` allows us to creates a range of values that are evenly spaced on a logarithmic scale, while `np.linspace` creates a range of values that are evenly spaced on a linear scale.
Using 'np.logspace' allows us to test different orders of magnitude for our parameters, making the search process much more useful and reducing the number of parameters to test.
The advantages of CV is that it allows us to evaluate the parameter's performance on the test set without actually using the test set.
Also, the CV calculates the mean performance on each fold and by that preventing the case of taking a "bad" validation set to check our parameters.
By that, CV allows us to tune our parameters to be very close to the optimal ones. 

2. The degree range was of length 3 and the lambda range was of length 20, so our grid search tested 60 combinations.
Each of the combinations was trained and tested 3 times (once for each fold). resulting $3\cdot 20 \cdot 3 = 180$ fits for our model.

"""

# ==============
