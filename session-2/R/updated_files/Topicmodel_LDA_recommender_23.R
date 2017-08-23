########## Topic Modeling Using LDA ##############
#1.Represent articles in terms of Topic Vector using trained LDA model (Load the trained model)
#2.Represent user in terms of Topic Vector of read articles using trained LDA model
#3.Calculate cosine similarity between read and unread articles
#4.Get the recommended articles

#########Loading the required libraries and installing the missing ones ###################
load.libraries <- c('tm', 'topicmodels','lda','MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

############### Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. User_list: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result
Path_News_Articles="~/Desktop/Banglaore learning/Recommender system/news_articles.csv"
User_list=c(7,6,76,61,761)
N=5

###### 1.Represent user read articles in terms of Topic Vector using trained LDA model  ##############
#1.Reading the csv file to get the Article id, Title and News Content

News_Articles<-read.csv(Path_News_Articles)
head(News_Articles)

#Select relevant columns and remove rows with missing values

News_Articles = News_Articles[,c('Article_Id','Title','Content')]
Articles = News_Articles[complete.cases(News_Articles),]
Articles$Content[0] # an uncleaned article

idx<-which(Articles$Article_ID %in% User_list)
User_articles = Articles[idx,]
#Combining user preferred articles together 
text<- paste(User_articles$Content, collapse=" ")

#################2.Represent articles in terms of Topic Vector using trained LDA model#####################
# Load the trained model
Article_topics = read.csv('Article_topics_150_topics.csv')
# Load the trained LDA model
load("Lda_model_150_topics.Rdata")

################# 3. Represent user in terms of Topic Vector of read articles using trained LDA model ###########
Article_topics$Article_ID = Articles$Article_Id
library(dplyr)
art_idx<- which(Article_topics$Article_ID %in% User_list)
user_article_topics<- matrix(colMeans(Article_topics[art_idx, 1:k]))

################# 4. Calculate cosine similarity between user read articles and unread articles ######################
Articles_ranking                         = Articles[,c("Article_Id","Title")]
Articles_ranking$Cosine_Similarity_Score = NA

for (i in 1:(nrow(Articles_ranking))){
  Articles_ranking[i,3] = sum(Article_topics[i,1:k]*user_article_topics)/
    (sqrt(sum((Article_topics[i,1:k])^2))*sqrt(sum((user_article_topics)^2)))
}

head(Articles_ranking)

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




















