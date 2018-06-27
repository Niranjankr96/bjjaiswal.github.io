---
layout: post
title: How to Structure Machine Learning Projects
date: 2018-04-20
categories: Deep Learning Notes
cover: 'https://i2.wp.com/philosophyofbrains.com/wp-content/uploads/2016/05/Problem-Mental-Causation-Pic.jpg?resize=1280%2C585'
tags: Deep-Learning
---


Even after developing a model, We find that prediction accuracy is 90% but that isn't good enough for production purpose because there are whole 10% error in prediction. So, we apply some technique to improve it like:
  - Collecting more diverse positive and negative training data
  - train the algorithm with gradient descent or use advance optimization algorithm like Adam.
  - Trying different network from smaller to bigger
  - also trying different regularization and dropout or changing the activation functions and number of hidden units.

## How to Structure Machine Learning Projects
Applying above technique may not give always give desired result despite spending months on collecting data. Therefore, Some strategies is necessary which helps us in improving the accuracy of models.

### Orthogonalization
There are hundreds of parameters and hyperparameters to tune to increase the prediction accuracy but it's cumbersome and tedious task to do. Here comes the role orthogonalization or orthogonality.

The word orthogonal means "Statistically independent". So,`Orthogonalization or orthogonality can be stated as a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effects  to  other components of the system.` It becomes easier to verify the algorithms independently from one another, it reduces testing and development time.

Let's take to metaphorical example of old television and car steering.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/tv-car.jpg" alt="metaphor"/>
    <figcaption> Metaphor for orthogonality </figcaption>
  </div>
</figure>


We can tune the picture on tv by varying the properties of knob of tv. Similarly, changing the steering of car can makes change in different direction to go.

When a supervised learning system is design, these are the 4 assumptions that needs to be true and orthogonal.
1. Fit training set well in cost function
  - If it doesn’t fit well then use of a bigger neural network or use a better optimization algorithm.
2. Fit development set well on cost function
  - If it doesn’t fit well then use regularization or use bigger training set and use bigger training set.
3. Fit test set well on cost function
  - If it doesn't fit well then use bigger Development set  
4. Performs well in real world
  - If it doesn't fit well then change development set  or cost function.

### Different tuning method or method for improving our model.
#### Using a single number evaluations metric. i.e.F1 score

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/decision.png" alt="decision"/>
    <figcaption> Confusion Matrix </figcaption>
  </div>
</figure>

 **Precision** : Of all the images we predicted y=1, what fraction of it have cats?
 <figure>
   <div style="text-align:center">
     <img src="/assets/img/ml-strategies/precision.png" alt="my alt text"/>
   </div>
 </figure>

 **Recall** : Of all the images that actually have cats, what fraction of it did we correctly identifying have cats?
 <figure>
   <div style="text-align:center">
     <img src="/assets/img/ml-strategies/recall.png" alt="my alt text"/>
   </div>
 </figure>

Let’s compare 2 classifiers A and B used to evaluate if there are cat images:
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/evaluation-matrix.png" alt="my alt text"/>
  </div>
</figure>

In this case the evaluation metrics are precision and recall.

For classifier A, there is a 95% chance that there is a cat in the image and a 90% chance that it has correctly detected a cat. Whereas for classifier B there is a 98% chance that there is a cat in the image and a 85% chance that it has correctly detected a cat.

The problem with using precision/recall as the evaluation metric is that you are not sure which one is better since in this case, both of them have a good precision et recall. F1-score, a harmonic mean, combine both precision and recall.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/F1-score.png" alt="my alt text"/>
  </div>
</figure>

Classifier A is a better choice. F1-Score is not the only evaluation metric that can be use, the average, for
example, could also be an indicator of which classifier to use.

#### Satisficing and optimizing metrics.
There are different metrics to evaluate the performance of a classifier, they are called `evaluation matrices`.
They can be categorized as `satisficing` and `optimizing` matrices. It is important to note that these evaluation matrices must be evaluated on a training set, a development set or on the test set.

>   
   - optimizing ---> best accuracy
   - satisficing ---> running <= X ms.


<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/3table.png" alt="my alt text"/>
  </div>
</figure>

In this case, accuracy and running time are the evaluation matrices. Accuracy is the optimizing metric,
because you want the classifier to correctly detect a cat image as accurately as possible. The running time
which is set to be under 100 ms in this example, is the satisficing metric which mean that the metric has
to meet expectation set.

#### Train/dev/test set distributions
Setting up the training, development and test sets have a huge impact on productivity. It is important to
choose the development and test sets from the same distribution and it must be taken randomly from all
the data. `Test set should be big enough to give high confidence in the overall performance of your system.`

In dataset where data instances are few like below 100,000, researcher has used 80-20 rule.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/old-dist.png" alt="my alt text"/>
    <figcaption> Strategy of distribution in machine learning </figcaption>
  </div>
</figure>
but because of generation Big Data, they have changed the rule to 98-1-1.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/new-dist.png" alt="my alt text"/>
    <figcaption> Strategy of distribution in Deep Learning </figcaption>
  </div>
</figure>

#### When to change dev/test sets and metrics
A cat classifier tries to find a great amount of cat images to show to cat loving users. The evaluation metric used is a classification error.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/error.png" alt="my alt text"/>
    <figcaption> Strategy of distribution in machine learning </figcaption>
  </div>
</figure>
It seems that `Algorithm A is better than Algorithm B` since there is only a 3% error, however for some reason, Algorithm A is letting through a lot of the pornographic images. Algorithm B has 5% error thus it classifies fewer images but it doesn't have pornographic images. From a company's point of view, as well as from a user acceptance point of view, `Algorithm B is actually a better
algorithm`. The evaluation metric fails to correctly rank order preferences between algorithms. The evaluation metric or the development set or test set should be changed.

The misclassification error metric can be written as a function as follow:
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/misclass.png" alt="misclassification"/>
  </div>
</figure>

This function counts up the number of misclassified examples.
The problem with this evaluation metric is that it treats pornographic vs non-pornographic images equally. On way to change this evaluation metric is to add the weight term 𝑤<sup>(𝑖)</sup>.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/more-weight.png" alt="misclassification"/>
  </div>
</figure>
The function becomes:
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/changed-equation.png" alt="misclassification"/>
  </div>
</figure>

**Reminder**:

- Define correctly an evaluation metric that helps better rank order classifiers
- Optimize the evaluation metric

#### Summary of orthogonalization
- [x] Define a metric to evaluate a classifier
- [x] Separately improve the metric to improve accuracy
- [x] Change metric and/or dev/test set, if previous choosen  metric + dev/test fails.
