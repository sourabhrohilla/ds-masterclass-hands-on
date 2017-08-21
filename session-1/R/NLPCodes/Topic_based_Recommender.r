## Topic Based Recommender 
# 1. Represent articles in terms of Topic Vector
# 2. Represent read articles in terms of Topic Vector
# 3. Calculate cosine similarity between read and unread articles
# 4. Output the recommended articles

#Describing parameters:
#  1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#  2. No_of_topics, (k) : Number of topics specified when training your topic model. This would refer to the dimension of each vector representing an article 
#  3. User_list: List of Article_Ids read by the user 
#  4. Recommended_Articles, N: Refers to the number of recommended articles as a result 
#  

PATH_NEWS_ARTICLES = "~/Desktop/Banglaore learning/Recommender system/newsaritcles.csv"
User_list = c(1,13,15)
k = 4 
N = 5


#setwd("~/Desktop/Banglaore learning/Recommender system")
library(tm)
library(topicmodels)
library(lda)
library(MASS)
#library(openNLP)
library(NLP)
library(R.utils)
library(stringdist)
library(dplyr) #data wrangling
rm(list = setdiff(ls(), "x"))

#Variety of sources supported by tm
getSources()

#REading our file
library(data.table)

Articles<-read.csv("~/Desktop/Banglaore learning/Recommender system/newsaritcles.csv")

# creating a corpus from the pre-processed data frame and further post-processing
#Setting up source and Corpus(list of content of articles)
myCorpus = Corpus(VectorSource(Articles$Content))
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x)) 
myCorpus<- tm_map(myCorpus, toSpace, "/|@|\\|")
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
myCorpus <- tm_map(myCorpus, removeURL)
myCorpus <- tm_map(myCorpus, tolower)
myCorpus <- tm_map(myCorpus, removePunctuation)
myCorpus <- tm_map(myCorpus, removeNumbers)
myCorpus <- tm_map(myCorpus, PlainTextDocument)
myCorpus <- tm_map(myCorpus, removeWords, stopwords("english"))
myCorpus <- tm_map(myCorpus,stemDocument)

writeLines(as.character(myCorpus))


#Tokenise
removeRegex<- function(x) strsplit(gsub("[^[:alnum:] ]", "", x), " +")
myCorpus<- tm_map(myCorpus, removeRegex)

# creating document term matrix
Dtm <- DocumentTermMatrix(myCorpus, 
                            control = list(wordLengths=c(2,Inf), 
                                           weighting=weightTf))
#Dtm <- Dtm[rowTotals>0,]

# running lda with k topics
k = 4
Dtm_LDA <- LDA(Dtm,4,method = "Gibbs")

Topic_terms = as.data.frame(t(posterior(Dtm_LDA)$terms))
#head(Topic_terms)

Article_topics =  as.data.frame(Dtm_LDA@gamma)
Article_topics$Article_ID = x$Article_ID

#EDA
get_terms(Dtm_LDA,15)


# predicting the topic distribution for user input text 
#@@@@R
User_list = c(1,4,15,36,72,89)

library(dplyr)
art_idx<- which(Article_topics$Article_ID %in% User_list)

user_article_topics<- matrix(colMeans(Article_topics[art_idx, 1:4]))

User_article_row = dim(Article_topics)[1]

Articles_ranking = data.frame('Article_ID'= integer(0), 'Score'= numeric(0))
for (i in 1:(User_article_row)){
  print(i)
  Articles_ranking[i,2] = sum(Article_topics[i,1:k]*user_article_topics)/
    (sqrt(sum((Article_topics[i,1:k])^2))*sqrt(sum((user_article_topics)^2)))
}
Articles_ranking$Article_ID = Articles$Article_ID

Articles_ranking<-Articles_ranking[order(-Articles_ranking$Score),]

write.csv(Topic_terms, 'Topic_terms.csv')
write.csv(Article_topics, 'Article_topics.csv')

## Recommendations based on topic distribution similarity 
N<-4
articles_recommended_id = Articles_ranking[1:N,1]
Top_n_articles = Articles[which(x$Article_ID %in% articles_recommended_id),c(1,2)]

## Print user articles 
User_read_articles = Articles[which(x$Article_ID %in% User_list),c(1,2)]
print('User read articles')
print(User_read_articles[,2])

## Print recoommended articles 

print('Recommended articles')
print(Top_n_articles$Title)
