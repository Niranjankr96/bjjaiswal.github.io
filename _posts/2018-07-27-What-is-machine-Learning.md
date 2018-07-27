---
layout: post
title: What is Machine Learning? And What are its types?
date: 2018-05-30
categories: Machine Learning notes
tags: Machine Learning
cover: 'https://d2ufo47lrtsv5s.cloudfront.net/content/sci/358/6370/1530/F1.large.jpg?width=800&height=600&carousel=1'
---

Due to growing volumes and varieties of available data, computational processing that is cheaper and more powerful, and affordable data storage, machine learning is changing domains of technology like never changed before. It has now become vital tools for many business enterprises or in education research for finding new patterns or new predictions. Here are a few widely publicized examples of machine learning applications you may be familiar with:

   * The heavily hyped, self-driving Google car? The essence of machine learning.
   * Online recommendation offers such as those from Amazon and Netflix? Machine learning applications for everyday life.
   * Knowing what customers are saying about you on Twitter? Machine learning combined with linguistic rule creation.
   * Fraud detection? One of the more obvious, important uses in our world today.

# What is Machine Learning?
There are various definitions of machine learning found in web. Some notable definitions are listed below:
1. The oldest and informal definition of machine learning was given by Arthur Samuel which describes as:
  * Def 1: "Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed".
2. The most modern definition was given by Tom Mitchell and he states it as:
  * Def 2: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
* What does Mitchell's definition actually means?
  * Let us say a machine is playing checker game the T, P, and E can be well defined by
    * E = the experience of playing many games of checkers
    * T = the task of playing checkers.
    * P = the probability that the program will win the next game.

## Machine learning methods     
In general, Any Machine Learning problem can be categories in three group.
1. Supervised Learning
2. Unsupervised Learning.
3. Reinforcement Learning but still there are other forms of learning too.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-notes/week1/types-of-ml.jpg" alt="Types of Machine Learning"/>
    <figcaption> DIfferent Types of ML. Credit: JP MORGAN Macro QDS. </figcaption>
  </div>
</figure>


### Supervised Learning

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-notes/week1/sk-cheatsheat.png" alt="Types of Machine Learning"/>
    <figcaption> Credits: Scikit-learn: Machine Learning in Python. </figcaption>
  </div>
</figure>

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output. We can categories supervised learning methods  in a few large categories:      

  1. Classification:
       * In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. Example: Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.  
  2. Regression:
       * In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. Example: Given a picture of a person, we have to predict their age on the basis of the given picture.

### Unsupervised Learning
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback based on the prediction results.  

The goal in such problems may be to discover groups of similar examples within the data, where it is called **clustering**, or to determine the distribution of data within the input space, known as **density estimation**, or to project the data from a high-dimensional space down to two or three dimensions for the purpose of **visualization**

* Examples:
  * Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
  * Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).      

## Learning Model and Cost Functions.
A common practice in machine learning to evaluate an algorithm is to split the data at hand into two sets, one that we call the training set on which we learn data properties and one that we call the testing set on which we test these properties.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function *h : X → Y* so that **h(x)** is a “good” predictor for the corresponding value of **y**. For historical reasons, this function h is called a **hypothesis**. Seen pictorially, the process is therefore like this:

![Model FLow-diagram](/assets/img/ml-notes/week1//model-flowdiagram.png)
*Credits: Machine Learning course on cousera*

Our first supervised learning model will be linear regression with one variable.

## Linear Regression with one variable
Linear regression is a model for modelling the relationship between a numerical variable y and one features as a linear function.

### Cost Function
In machine Learning, the cost function is function that maps parameter of a model to a number (the cost). The cost function measures how well or how badly a model fits the dataset and the objective is often to find a model that describes the data well by picking a model that minimizes some cost function. Some common cost function are squared error loss which is used in linear regression or the logistic loss which is used in logistic regression.

![cost Function](/assets/img/ml-notes/week1//cost-function.png)
![Cost Function:inituitive I](/assets/img/ml-notes/week1//cost-function-1.png)
![Cost Function:inituitive II](/assets/img/ml-notes/week1//cost-function-2.png)

### Gradient Descent
Here, we will talk about an algorithm called gradient descent for minimization of the cost function **J**.

* Gradient descent is an iterative optimization algorithm. I finds a local minimum of a function by repeatedly taking steps proportional to the negative of the gradient of the function at current point. Intuitively we think of this as repeatedly taking steps in the direction of steepest descent.

* Gradient ascent: If one takes steps proportional to the positive of the gradient, one approaches a local maximum of the function, the procedure is then known as gradient ascent.

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields θ<sub>0</sub> and θ<sub>1</sub> (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing x and y itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put θ<sub>0</sub> on the x axis and θ<sub>1</sub> on the y axis, with the cost function on the vertical z axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

![contour of gradient descent](/assets/img/ml-notes/week1//contour.png)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter **α**, which is called the **learning rate**.

- Note: In gradient descent, learning rate refers to parameter that controls how large of a steps we should take in the direction of descent. When the learning rate is high, we take large steps(big changes to parameter values) and when the learning rate is low we take small steps (small changes to parameter values).

For example, the distance between each 'star' in the graph above represents a step determined by our parameter α. A smaller α would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of **J(θ<sub>0</sub>,θ<sub>1</sub>)**. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

![Gradient descent algorithm](/assets/img/ml-notes/week1/gradient-descent-algorithm.png)
![gradient descent intuition](/assets/img/ml-notes/week1/gradient-descent-intuition.png)

### Gradient Descent For Linear Regression
![](/assets/img/ml-notes/week1/gradient-descent-in-linear-regression.png)



The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent. `Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum`. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-notes/week1/quadratic-contuor.png" alt="quadratic-contuor"/>
    <figcaption>quadratic-contuor. </figcaption>
  </div>
</figure>


The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through as it converged to its minimum.
