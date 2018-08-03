---
title: How they differ? Artificial Intelligence vs machine Learning vs Deep Learning
date: 2018-05-30
categories: AI
tags: Artificial-Intelligence
feature_image: 'https://analyticsindiamag.com/wp-content/uploads/2017/08/handwriting-human-head-shape-imagination-technology-innovation-artificial-intelligence-at-chalkboard_h7plpruc_thumbnail-full11.png'
image: "https://analyticsindiamag.com/wp-content/uploads/2017/08/handwriting-human-head-shape-imagination-technology-innovation-artificial-intelligence-at-chalkboard_h7plpruc_thumbnail-full11.png"
---

We have come far beyond Information Technology age. We are living in __Intelligent__ era. We are surrounded by ubiquitous intelligent device like smart-phones, smart-watch,etc,. Intelligent has become banal not only in regular life style but also affecting the industries too. Devices which have __Artificial Intelligence__ has started changing the world drastically. As Andrew Ng stated *"Like electricity changed the world a century ago, similarly AI will change the world like never before"*.  Nowadays we look for intelligent software to automate banal labor, understand speech, or images, make diagnosis in medicine and support basic scientific research.


> "Artificial intelligence": When a machine mimics "cognitive" functions that humans associate with other human minds, such as "learning" and "problem solving" - Russell & Norvig, 2009

We all are familiar with word Artificial Intelligence form movies like "The Terminator", "The Matrix", "A Space Odyssey", etc and best classic novels like "Frankenstein", "The Hitchhiker guide to the Galaxy", and many more. But it's an arduous task to explain Artificial Intelligence to novice and distinguish it for its branches like __Machine Learning__ and __Deep Learning__, __Reinforcement learning__. In this post, I will try to explain in a laconic way despite being cogent: How AI, Machine Learning and Deep learning are different from each other nevertheless, shelter under the same umbrella.


During 1900s, when researcher discovered that digital computers can simulate any process of formal reasoning a.k.a Church–Turing thesis- an idea was embroiled among the researchers. The idea to accumulate understanding from __Neurobiology__, __Information Theory__, __Cybernetics__ and many other branches of Natural science and engineering to build a __electronic brain__ or __artificial brain__. Then, the perennial movement of Artificial Intelligence begun.


<figure>
  <div style="text-align:center">
    <img src="/assets/img/2018-05-30/Artificial-Brain.png" alt="my alt text"/>
    <figcaption> Artificial brain. </figcaption>
  </div>
</figure>

>An artificial brain (or artificial mind) is software and hardware with cognitive
abilities similar to those of the animal or human brain. -
> BBC News,2009

## Classical AI
Earlier/classical AI system are built on a list of formal, logistic and mathematical rules. Some such rules were Informed/Uninformed search strategies, Propositional Logic, First Order Logic, Planning, Heuristics functions, etc,.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/2018-05-30/knowledge-based-systems.jpg" alt="kNR"/>
    <figcaption>Contains of Knowledge based Systems. </figcaption>
  </div>
</figure>
&nbsp;

We feted IBM's Deep chess-playing system when it defeated world champion Garry Kasparov. But the system was completely rules and logic based system. If you know the board position, it is possible to know all you need to know to play the game, because given a certain board position, the possible moves are precisely defined and finite in number. Even though you may not know the particular move your opponent will make, you know that he will make a legal move. A system for defeating games like chess, tic-tac-toe, AlphaGO, etc can be built by programmer ahead of competition from a list of complete formal rules and logics. Thus classic AI is known as __Knowledge based Intelligent System__.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/2018-05-30/tic-toc-toe.png" alt="tic-tac-toe"/>
    <figcaption> Game tree of tic-tac-toe. Courtesy: Google DeepMind's AlphaGo: How it works
 </figcaption>
  </div>
</figure>
&nbsp;
<figure>
  <div style="text-align:center">
    <img src="/assets/img/2018-05-30/tree-based.png" alt="tree-based"/>
    <figcaption>Tree based search system for GO. Courtesy Google's Deepmind </figcaption>
  </div>
</figure>
&nbsp;

The __problem__ with the classical AI is: it's inability to learn from environment or to extract and acquire their own knowledge from patterns of raw data provided to them. As they are bolstered by hand-coded knowledge.

The list of Instance where classical AI fails:

- Unable to extract information from real world for ex: camera images, typically in the form of a pixel array, and map them onto internal representations of the world
- Classical AI models do not take the real world sufficiently into account. so, problem of generalization occurs.

## Machine Learning.
Machine Learning solves the problems classical AI by providing systems the ability to acquire their own knowledge by extracting information from real world data.

Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. Let me give you formal Book definition:

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - Tom Mitchell's Machine Learning Book

A bit scary definition right? So, let me delineate it simply:
Let us say a machine is playing checker game.
The T, P, and E can be well defined by
- E = the experience of playing many games of checkers
- T = the task of playing checkers.
- P = the probability that the program will win the next game.

With every experience ( E ) collected from environment or dataset or game, what decisions or tasks ( T ) the player has to take, so probability ( P ) of win is maximized.

The performance of every machine learning algorithms depends on the __representation__ aka __features__ of the data they are given. This features helps AI to retain knowledge of patterns from raw data provided to them. The machine learning algorithm learns from data by correlating each features of data with various output.

<figure>
  <div style="text-align:center">
    <img src="/assets/img/2018-05-30/ml-algorithm.png" alt="ml-algorithm"/>
    <figcaption>Different types of Machine Learning Algorithm. </figcaption>
  </div>
</figure>
&nbsp;     

It is not surprising that the choice of features has an colossal effect on the performance of machine learning algorithms. However, it is difficult to know what feature should be extracted for best performance of machine learning algorithm. For example, if MRI scan of the patient are given to Logistic Regression, it would not be able to make useful prediction as all the features are uniform in the form of pixel.

Hence, the __problem__ with machine learning- it's little bleak to use ML algorithm when we have difficult to extract a representation or best features to solve problem.

The list of instance where machine Learning algorithm fails:
- ML algorithms fails to give better result when datasets or representation of datasets is unstructured like images, audio, video.
- It fails when there is no more data for learning purpose.
- ML fail when, among the training examples, there are often similar inputs that are associated to different outputs.
- Ml algorithms are designed to deal with noise in datasets at certain level but as more noise increase more difficult learning process becomes, which leads to bad accuracy.

## Deep Learning.
