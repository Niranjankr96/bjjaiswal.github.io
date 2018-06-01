Today, we can no more state that we are living in __Information Technology__ era. We have come far beyond the IT. We are living in __Artificial Intelligence__ age. AI is affecting diverse industries ranging from __healthcare to an automobile__, __video games to military__, __finance and economics to audit__, and many more.

 > "Artificial intelligence": When a machine mimics "cognitive" functions that humans associate with other human minds, such as "learning" and "problem solving" - Russell & Norvig, 2009

We all are familiar with word Artificial Intelligence form movies like "The Terminator", "The Matrix", "A Space Odyssey", etc and best classic novels like "Frankenstein", "The Hitchhiker guide to the Galaxy", and many more. But it's an arduous task to explain Artificial Intelligence to novice and distinguish it for its branches like __Machine Learning__ and __Deep Learning__, __Reinforcement learning__. In this post, I will try to explain in a laconic way despite being cogent: How AI, Machine Learning and Deep learning are different from each other nevertheless, shelter under the same umbrella.


During 1900s, when researcher discovered that digital computers can simulate any process of formal reasoning a.k.a Church–Turing thesis- an idea was embroiled among the researchers. The idea to accumulate understanding from __Neurobiology__, __Information Theory__, __Cybernetics__ and many other branches for Natural science and engineering to build a __electronic brain__ or __artificial brain__. Then the perennial movement of Artificial Intelligence begun.


<figure>
  <div style="text-align:center">
    <img src="/img/2018-06-30/Artificial-Brain.png" alt="my alt text"/>
    <figcaption> Artificial brain.  </figcaption>
  </div>
</figure>

>An artificial brain (or artificial mind) is software and hardware with cognitive
abilities similar to those of the animal or human brain. -
> BBC News,2009
For simple games such as tic-tac-toe, it is possible to represent all possible game states (the complete game tree) visually

## Classical AI
Earlier AI system were built on a list of formal, logistic and mathematical rules. Some such rules were Informed/Uninformed search strategies, Propositional Logic, First Order Logic, Planning, Heuristics functions, etc,.

<figure>
  <div style="text-align:center">
    <img src="/img/2018-06-30/knowledge-based-systems.jpg" alt="my alt text"/>
    <figcaption>Contains of Knowledge based Systems. </figcaption>
  </div>
</figure>
&nbsp;

We feted IBM's Deep chess-playing system when it defeated world champion Garry Kasparov. But the system was completely rules and logic based system. A system for defeating games like chess, tic-tac-toe, AlphaGO, etc can be built by programmer ahead of competition from a list of complete formal rules and logics. This is known as __Knowledge based Intelligent System__.

<figure>
  <div style="text-align:center">
    <img src="/img/2018-06-30/tic-toc-toe.png" alt="my alt text"/>
    <figcaption> Game tree of tic-tac-toe. Courtesy: Google DeepMind's AlphaGo: How it works
 </figcaption>
  </div>
</figure>
&nbsp;
<figure>
  <div style="text-align:center">
    <img src="/img/2018-06-30/tree-based.png" alt="my alt text"/>
    <figcaption>Tree based search system for GO. Courtesy Google's Deepmind </figcaption>
  </div>
</figure>
&nbsp;

The __problem__ with the classical AI or simply AI is it's inability to learn from environment or to extract and acquire their own knowledge from patterns of raw data provided to them. As they are bolstered by hand-coded knowledge.

The __solution__ is provided by Machine Learning.
## Machine Learning.
Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. Let me give you formal Book definition:

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - Tom Mitchell's Machine Learning Book

A bit scary definition right? So, let me delineate it simply:
Let us say a machine is playing checker game.
The T, P, and E can be well defined by
   * E = the experience of playing many games of checkers

   * T = the task of playing checkers.

   * P = the probability that the program will win the next game.

With every experience ( E ) collected from environment or dataset or game, what decisions or tasks ( T ) the player has to take, so probability ( P ) of win is maximum.

The performance of every machine learning algorithms depends on the __representation__ aka __features__ of the data they are given. This features helps AI to retain knowledge of patterns from raw data provided to them.
The machine learning algorithm learns from data by correlating each features of data with various output.

<figure>
  <div style="text-align:center">
    <img src="/img/2018-06-30/ml-algorithm.png" alt="my alt text"/>
    <figcaption>Different types of Machine Learning Algorithm. </figcaption>
  </div>
</figure>
&nbsp;     

It is not surprising that the choice of features has an colossal effect on the performance of machine learning algorithms. However, it is difficult to know what feature should be extracted for best performance of machine learning algorithm. For example, if MRI scan of the patient are given to Logistic Regression, it would not be able to make useful prediction as all the features are uniform in the form of pixel.

Hence, the __problem__ with machine learning- it's little bleak to use ML algorithm when we have difficult to extract a representation or best features to solve problem.

For the __solution__ of above problem, practitioner use Deep learning.
## Deep Learning.
