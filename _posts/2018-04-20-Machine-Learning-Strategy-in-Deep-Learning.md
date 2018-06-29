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

## Comparing to human-level performance
Today, machine  learning  algorithms can  compete  with human - level  performance  since  they  are  more productive  and  more  feasible  in  a  lot of  application. Also,  the workflow  of  designing  and building  a machine learning system, is much more efficient than before.

Moreover, some of the tasks that humans do are close  to ‘’perfection’’, which is why machine learning tries to mimic human-level performance.

>Bayes  optimal  error  is  defined  as  the  best  possible  error.  In  other  words, it  means that  any functions mapping from x to y can’t surpass a certain level of accuracy.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/human vs model.png" alt="misclassification"/>
    <figcaption> Performance
of humans and machine learning over time.</figcaption>
  </div>
</figure>

Machine learning  progresses  slowly  when  it  surpasses human-level  performance.  One  of  the  reason  is that human-level  performance  can  be  close to  Bayes  optimal  error,  especially  for  natural  perception problem.

Also, when  the  performance  of machine  learning  is  worse  than the  performance  of humans,  you  can improve it with different tools. They are harder to use once its surpasses human-level performance.

These tools are:
- Get labeled data from humans
- Gain insight from manual error analysis: Why did a person get this right?
- Better analysis of bias/variance

#### Avoidable bias
By knowing what the human-level performance is, it is possible to tell when a training set is performing well or not.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/avoidable-bias.png" alt="misclassification"/>
  </div>
</figure>

In this case, the human level error as a proxy for Bayes error since humans are good to identify images.If you want  to  improve  the performance  of  the  training  set  but  you  can’t do better than the Bayes error otherwise the training set is overfitting. By knowing the Bayes error, it is easier to focus on whether bias or variance avoidance tactics will improve the performance of the model.

`Scenario A:` There is a `7%` gap between the performance of the training set and the human
level error. It means that the algorithm is n’t fitting well with the training set since the target is around 1%. To resolve the issue, we use bias reduction technique such as training a bigger neural network or running the training set longer.

`Scenario B:` The  training  set  is  doing  good  since  there  is  only a `0.5%` difference with the  human  level error.  `The difference between the training set and the human level error is called avoidable bias`. The focus here is to reduce the variance since the difference between the training error and the development error is 2%. To resolve the issue, we use variance reduction technique such as regularization or have a bigger training set

#### Understanding human-level performance
Human-level error gives an estimate of Bayes error.

**Example 1: Medical image classification**
This is an example of a medical image classification in which the input is a radiology image and the output is a diagnosis classification decision.
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/scenario-A.png" alt="misclassification"/>
  </div>
</figure>
The definition of human-level error depends on the purpose of the analysis, in this case, by definition the Bayes error is lower or equal to 0.5%.


**Example 2: Error analysis**
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/error-analysis.png" alt="misclassification"/>
  </div>
</figure>

`Scenario A`
In this case, the choice of human-level performance doesn’t have an impact. The avoidable bias is between 4%-4.5% and the variance is 1%. Therefore, the focus should be on bias reduction technique.

`Scenario B`
In this case, the choice of human-level performance doesn’t have an impact. The avoidable bias is between 0%-0.5% and the variance is 4%. Therefore, the focus should be on variance reduction technique.

`Scenario C`
In this case, the estimate for Bayes error has to be 0.5% since you can’t go lower than the human-level performance otherwise the training set is overfitting. Also, the avoidable bias is 0.2% and the variance is 0.1%. Therefore, the focus should be on bias reduction technique.

Summary of bias/variance with human-level performance

* Human - level error – proxy for Bayes error

* If the difference between human-level error and the training error is bigger than the difference between the training error and the development error. The focus should be on bias reduction technique

* If the difference between training error and the development error is bigger than the difference between the human-level error and the training error. The focus should be on variance reduction technique

#### Surpassing human-level Performance
Example1 : classification task
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/surpassing.png" alt="misclassification"/>
  </div>
</figure>

`Scenario A:`
In this case, the Bayes error is 0.5%, therefore the available bias is 0.1% et the variance is 0.2%.

`Scenario B:`
In this case, there is not enough information to know if bias reduction or variance reduction has to be done on the algorithm. It doesn’t mean that the model cannot be improve, it means that the conventional ways to know if bias reduction or variance reduction are not working in this case.

There are many problems where machine learning significantly surpasses human-level performance, especially with structured data:
* Online advertising
* Product recommendations
* Logistics (predicting transit time)
* Loan approvals

#### Improving model performance

There are `2 fundamental assumptions of supervised learning`. The first one is to `have a low avoidable bias` which means that the training set fits well. The second one is to `have a low or acceptable variance` which means that the training set performance generalizes well to the development set and test set.

If the difference between human-level error and the training error is bigger than the difference between the training error and the development error, the focus should be on bias reduction technique which are training a bigger model, training longer or change the neural networks architecture or try various hyperparameters search.

If the difference between training error and the development error is bigger than the difference between the human-level error and the training error, the focus should be on variance reduction technique which are bigger data set, regularization or change the neural networks architecture or try various hyperparameters search.


#### Summary
<figure>
  <div style="text-align:center">
    <img src="/assets/img/ml-strategies/improving.png" alt="misclassification"/>
  </div>
</figure>

### Error analysis

`Error Analysis`: The process of manually examining mistakes, when learning algorithms do not give the performance of a human level. It helps in finding insight on what to do next.

For Example:

* In the cat classification example, if you have 10% error on your dev set and you want to decrease the error.
* You discovered that some of the mislabeled data are dog pictures that look like cats. Should you try to make your cat classifier do better on dogs (this could take some weeks)?
  * `Error analysis approach`:
    - Get 100 mislabeled dev set examples at random.
    - Count up how many are dogs.
    - if 5 of 100 are dogs then training your classifier to do better on dogs will decrease your error up to 9.5% (called ceiling), which can be too little.
    - if 50 of 100 are dogs then you could decrease your error up to 5%, which is reasonable and you should work on that.

* Based on the last example, error analysis helps you to analyze the error before taking an action that could take lot of time with no need. Sometimes, you can evaluate multiple error analysis ideas in parallel and choose the best idea. Create a spreadsheet to do that and decide, e.g.:

 <figure>
   <div style="text-align:center">
     <img src="/assets/img/ml-strategies/table-strategies.png" alt="misclassification"/>
   </div>
 </figure>
 
This quick counting procedure, which you can often do in, at most, small numbers of hours can really help you make much better prioritization decisions, and understand how promising different approaches are to work on.

#### Carrying out error analysis
#### Cleaning up incorrectly labeled data
#### Build your first system quickly, then iterate

### Mismatched training and dev/test sets
#### Bias and Variance with Mismatched data distributions.
#### Addressing data Mismatched

### Learning from multiple tasks
#### Transfer Learning

* Apply the knowledge you took in a task A and apply it in another task B.
* For example, you have trained a cat classifier with a lot of data, you can use the part of the trained NN it to solve x-ray classification problem.
* To do transfer learning, delete the last layer of NN and it's weights and:
 * Option 1: if you have a small data set - keep all the other weights as a fixed weights. Add a new last layer(-s) and initialize the new layer weights and feed the new data to the NN and learn the new weights.
  * Option 2: if you have enough data you can retrain all the weights.
* Option 1 and 2 are called `fine-tuning` and training on task A called `pretraining`.
* When transfer learning make sense:
  * Task A and B have the same input X (e.g. image, audio).
  * You have a lot of data for the task A you are transferring from and relatively less data for the task B your transferring to.
  * Low level features from task A could be helpful for learning task B.


#### Multi-task Learning

### End-to-end deep Learning
#### What is end-to-end deep learning?
#### Wheteher to use end-to-end deep Learning
