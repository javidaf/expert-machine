I am currenntly working on a machine learning excerise. 
The description for the excerise is as follows:

Overarching aims of the exercises this week
The aim of the exercises this week is to get started with implementing gradient methods of relevance for project 2. This exercise will also be continued next week with the addition of automatic differentation. Everything you develop here will be used in project 2.

In order to get started, we will now replace in our standard ordinary least squares (OLS) and Ridge regression codes (from project 1) the matrix inversion algorithm with our own gradient descent (GD) and SGD codes. You can use the Franke function or the terrain data from project 1. However, we recommend using a simpler function like $f(x)=a_0+a_1x+a_2x^2$
 or higher-order one-dimensional polynomials. You can obviously test your final codes against for example the Franke function. Automatic differentiation will be discussed next week.

You should include in your analysis of the GD and SGD codes the following elements

A plain gradient descent with a fixed learning rate (you will need to tune it) using the analytical expression of the gradients

Add momentum to the plain GD code and compare convergence with a fixed learning rate (you may need to tune the learning rate), again using the analytical expression of the gradients.

Repeat these steps for stochastic gradient descent with mini batches and a given number of epochs. Use a tunable learning rate as discussed in the lectures from week 39. Discuss the results as functions of the various parameters (size of batches, number of epochs etc)

Implement the Adagrad method in order to tune the learning rate. Do this with and without momentum for plain gradient descent and SGD.

Add RMSprop and Adam to your library of methods for tuning the learning rate.

The lecture notes from weeks 39 and 40 contain more information and code examples. Feel free to use these examples.

In summary, you should perform an analysis of the results for OLS and Ridge regression as function of the chosen learning rates, the number of mini-batches and epochs as well as algorithm for scaling the learning rate. You can also compare your own results with those that can be obtained using for example Scikit-Learn's various SGD options. Discuss your results. For Ridge regression you need now to study the results as functions of the hyper-parameter 
 and the learning rate 
. Discuss your results.

You will need your SGD code for the setup of the Neural Network and Logistic Regression codes. You will find the Python Seaborn package useful when plotting the results as function of the learning rate 
 and the hyper-parameter 
 when you use Ridge regression.

We recommend reading chapter 8 on optimization from the textbook of Goodfellow, Bengio and Courville. This chapter contains many useful insights and discussions on the optimization part of machine learning.