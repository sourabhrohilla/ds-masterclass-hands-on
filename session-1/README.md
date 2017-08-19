**Problem statement**  
Software to detect network intrusions protects a computer network from unauthorized users.
Learning the “signatures” of such attacks from a TCP dump is an important data mining
Problem.
A connection is a sequence of TCP packets starting and ending at some well defined times,
between which data flows to and from a source IP address to a target IP address under some
well defined protocol. Assuming the existence of a labelled sample of connection data in which
each connection is labelled as either normal, or as an attack, with exactly one specific attack
type, the learning task is to build a classifier capable of distinguishing between “bad”
connections (intrusions or attacks), and “good” (normal) connections. The task is then extended to predict specific type
of attack.

**Approach**  
In the hands-on session, we start with looking at the dataset and understand features and label. 
We model the problem as a binary classification problem first, and distinguish a normal connection from a intrusion connection.
We’ll use different algorithms like Naive Bayes, Logistic Regression, Support Vector Machines, Decision trees, Random Forests,
Gradient boosted decision trees. We’ll compare different algorithms by doing K-fold cross validation and metrics like accuracy,
precision and recall. The hyper-parameters of algorithms will be tweaked using grid-search cross validation. 
We’ll look at feature engineering, i.e. combining features in interesting ways to improve the accuracy. 
Then, we move to the task of predicting specific type of attack, i.e. a multi-class classification problem. 
We look at various ways to balance class labels and improve the performance. The session is concluded by bringing all the
parts together and building a machine learning pipeline which can be explored further and re-used in other classification
problems.
