########## Topic Modeling Using LDA ##############
#1.Text Processing of articles and user read articles 
#2.Represent articles in terms of Topic Vector using LDA model
#3.Represent user in terms of Topic Vector of read articles using trained LDA model
#4.Calculate cosine similarity between read and unread articles
#5.Get the recommended articles

############### Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. User_list: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result
Path_News_Articles="~/Desktop/Banglaore learning/Recommender system/news_articles.csv"
User_list=c(7,6,76,61,761)
N=5

#########Loading the required libraries and installing the missing ones ###################
load.libraries <- c('tm', 'topicmodels','lda','MASS','NLP','R.utils', 'stringdist','dplyr')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

###### 1.Represent articles in terms of bag of words ##############
#1.Reading the csv file to get the Article id, Title and News Content
#2.Remove punctuation marks and other symbols from each article
#3.Tokenize each article
#4. Stem token of every article

News_Articles<-read.csv(Path_News_Articles)
head(News_Articles)

#Select relevant columns and remove rows with missing values

News_Articles = News_Articles[,c('Article_Id','Title','Content')]
Articles = News_Articles[complete.cases(News_Articles),]
Articles$Content[0] # an uncleaned article

# Text cleaning and Tokenizing 
myCorpus = Corpus(VectorSource(Articles$Content))
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x)) 
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

#Tokenise
removeRegex<- function(x) strsplit(gsub("[^[:alnum:] ]", "", x), " +")
myCorpus<- tm_map(myCorpus, removeRegex)

################ 2. Text processing of user read articles ########################
idx<-which(Articles$Article_ID %in% User_list)
User_articles = Articles[idx,]
#Combining user preferred articles together 
text<- paste(User_articles$Content, collapse=" ")

# Text processing of User articles 
userCorpus = Corpus(VectorSource(text))
toSpace    = content_transformer(function(x, pattern) gsub(pattern, " ", x)) 
userCorpus = tm_map(userCorpus, toSpace, "/|@|\\|")
removeURL  = function(x) gsub("http[[:alnum:]]*", "", x)
userCorpus = tm_map(userCorpus, removeURL)
userCorpus = tm_map(userCorpus, tolower)
userCorpus = tm_map(userCorpus, removePunctuation)
userCorpus = tm_map(userCorpus, removeNumbers)
userCorpus = tm_map(userCorpus, PlainTextDocument)
userCorpus = tm_map(userCorpus, removeWords, stopwords("english"))
userCorpus = tm_map(userCorpus,stemDocument)

#Tokenise
removeRegex = function(x) strsplit(gsub("[^[:alnum:] ]", "", x), " +")
userCorpus  = tm_map(userCorpus, removeRegex)

################ 3. Training of LDA model using Article corpus ###################################
# creating document term matrix
Dtm <- DocumentTermMatrix(myCorpus, 
                          control = list(wordLengths=c(2,Inf), 
                                         weighting=weightTf))
# running lda with k topics
k=4
Dtm_LDA <- LDA(Dtm,k,method = "Gibbs")
Topic_terms = as.data.frame(t(posterior(Dtm_LDA)$terms))
Article_topics =  as.data.frame(Dtm_LDA@gamma)
View_topic_list = get_terms(Dtm_LDA,15)

setwd("~/Desktop/Banglaore learning/Recommender system")
write.csv(Article_topics, 'Article_topics.csv')
Article_topics = read.csv('Article_topics.csv')
save.image(file = "~/Desktop/Banglaore learning/Recommender system/Lda_model_final.Rdata")
# Load the trained LDA model

Article_topics$Article_ID = Articles$Article_Id
library(dplyr)
art_idx<- which(Article_topics$Article_Id %in% User_list)
user_article_topics<- matrix(colMeans(Article_topics[art_idx, 1:4]))

######################### 4. Calculate cosine similarity between user read articles and unread articles ######################


Articles_ranking                         = Articles[,c("Article_Id","Title")]
Articles_ranking$Cosine_Similarity_Score = NA
head(Articles_ranking)

for (i in 1:(nrow(Articles_ranking))){
  print(i)
  Articles_ranking[i,2] = sum(Article_topics[i,1:k]*user_article_topics)/
    (sqrt(sum((Article_topics[i,1:k])^2))*sqrt(sum((user_article_topics)^2)))
}
# Remove the articles which are already read by the user 
Articles_ranking = subset(Articles_ranking, !(Article_Id %in% User_list))

# Sorting based on the cosine similarity score
Articles_ranking = Articles_ranking[order(-Articles_ranking$Cosine_Similarity_Score),]

## Recommendations based on topic modelling 
articles_recommended_id = Articles_ranking[1:N,1]
Top_n_articles = Articles[which(Articles$Article_Id %in% articles_recommended_id),c(1,2)]

## Print user articles 
User_read_articles = Articles[which(Articles$Article_Id %in% User_list),c(1,2)]
print('User read articles')
print(User_read_articles)

## Print recoommended articles 
print(Top_n_articles)




















