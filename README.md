# ds-masterclass-hands-on
The folder 
1. session-1 : contains code for session on Intrusion Detection, divided into R and python code folders.
2. session-2 : contains code for session on Text classification, divided into R and python code folders.


### For participants who will be using python
1. Anaconda distribution for python : Go to https://www.continuum.io/downloads and download the latest Anaconda distribution.
2. Run `conda install jupyter`
3. Run `conda install -c glemaitre imbalanced-learn`
4. Install the libraries listed below using pip. 

Steps to install a library in python.
1. Go to terminal/command-prompt.
2. Run ```pip install <library name>```
3. For instance, to install numpy, you’d run ```pip install numpy```

**List of libraries used in the hands-on session**  
Session 1 :  Intrusion detection
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. sklearn
6. imblearn
7. xgboost  

Session 2 : News articles recommender
1. numpy
2. pandas
3. sklearn
4. nltk 3.2.4
5. Install nltk corpus and model:  
   ```$ python
   > import nltk
   > nltk.download('stopwords')
   > nltk.download('punkt')
   > nltk.download('maxent_ne_chunker')
   > nltk.download('averaged_perceptron_tagger')
   > nltk.download('words')   
   ```
6. gensim 0.12.4  
   ```
   conda install -c anaconda gensim
   ```

### For participants who will be using R
1. Set up R : Go to https://cran.rstudio.com/ and download R for your OS. Please download R 3.4.1
2. Set up R Studio : Go to  https://www.rstudio.com/products/rstudio/ and download open source version of RStudio Desktop.
3. Install the libraries listed below. 

Steps to install a library in RStudio
1. Open RStudio.
2. In the console, run ```install.packages(“<library name>”)```
3. For instance, to install ggplot2, you’d run ```install.packages(“ggplot2”)```  


**List of libraries used in the hands-on session**  
Session 1 :  Intrusion detection
1. ggplot2
2. randomForest
3. caret
4. rpart
5. plyr
6. gbm
7. rpart.plot
8. reshape2
9. naivebayes
10. corrplot
11. e1071  

Session 2 : News articles recommender
1. tm
2. topicmodels
3. lda
4. MASS
5. NLP
6. R.utils
7. stringdist
8. dplyr
9. openNLP
10. rjava
11. NLP
12. openNLP
13. RWeka
14. qdap
15. magrittr
16. openNLPmodels.en
17. data.table



If you are not able to setup your machine, please send an email to sourabh@tatrasdata.com
