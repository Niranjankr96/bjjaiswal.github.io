## How to Structure Machine Learning Projects


Even after developing a model, We find that prediction accuracy is 90% but that isn't good enough for production purpose because there are whole 10% error in prediction. So, we apply some technique to improve it like:
  - Collecting more diverse positive and negative training data
  - train the algorithm with gradient descent or use advance optimization algorithm like Adam.
  - Trying different network from smaller to bigger
  - also trying different regularization and dropout or changing the activation functions and number of hidden units.

Applying above technique may not give always give desired result despite spending months on collecting data. Therefore, Some strategies is necessary which helps us in improving the accuracy of models.

### Orthogonalization
There are hundreds of parameters and hyperparameters to tune to increase the prediction accuracy but it's cumbersome and tedious task to do. Here comes the role orthogonalization or orthogonality.

The word orthogonal means "Statistically independent". So,`Orthogonalization or orthogonality can be stated as a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effects  to  other components of the system.` It becomes easier to verify the algorithms independently from one another, it reduces testing and development time.

Let's take to metaphorical example of old television and car steering.

<figure>
  <div style="text-align:center">
    <img src="/img/ml-strategies/tv-car.jpg" alt="my alt text"/>
    <figcaption> Tuning the necessary design component of traditional TV and Car steering </figcaption>
  </div>
</figure>


We can tune the picture on tv by varying the properties of knob of tv. Similarly, changing the steering of car can makes change in different direction to go.

When a supervised learning system is design, these are the 4 assumptions that needs to be true and orthogonal.
1. Fit training set well in cost function
  - If it doesn’t fit well:
    - the use of a bigger neural network or
    - use a better optimization algorithm.
2. Fit development set well on cost function
  - If it doesn’t fit well:
    - regularization or using bigger training set
    - use bigger training set.
3. Fit test set well on cost function
  - If it doesn't fit well:
    - Use bigger Development set  
4. Performs well in real world
  - If it doesn't fit well:
    - change development set  or cost function.
