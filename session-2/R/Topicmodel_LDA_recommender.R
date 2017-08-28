########## Topic Modeling Using LDA ##############
#1.Represent articles in terms of Topic Vector using trained LDA model (Load the trained model)
#2.Represent user in terms of Topic Vector of read articles using trained LDA model
#3.Calculate cosine similarity between read and unread articles
#4.Get the recommended articles

#########Loading the required libraries and installing the missing ones ###################
load.libraries = c('tm', 'topicmodels','lda','MASS','dplyr','NLP','R.utils', 'stringdist','dplyr','SnowballC')
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

############### Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. USER_LIST: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result
PATH_NEWS_ARTICLES="/home/karan/Downloads/news_articles.csv"
USER_LIST=c(7,6,76,61,761)
NUM_RECOMMENDED_ARTICLES = 5

###### 1.Represent user read articles in terms of Topic Vector using trained LDA model  ##############
#1.Reading the csv file to get the Article id, Title and News Content

news_articles=read.csv(PATH_NEWS_ARTICLES)
head(news_articles)

#Select relevant columns and remove rows with missing values

news_articles = news_articles[,c('Article_Id','Title','Content')]
articles = news_articles[complete.cases(news_articles),]
articles$Content[0] # an uncleaned article

idx=which(articles$Article_Id %in% USER_LIST)
user_articles = articles[idx,]
#Combining user preferred articles together 
text= paste(user_articles$Content, collapse=" ")

#################2.Represent articles in terms of Topic Vector using trained LDA model#####################
# Load the trained LDA model
load("/home/karan/Downloads/LDA_Model_150_topics.Rdata")

################# 3. Represent user in terms of Topic Vector of read articles using trained LDA model ###########
article_topics$Article_Id = articles$Article_Id
art_idx= which(article_topics$Article_Id %in% USER_LIST)
user_article_topics= matrix(colMeans(article_topics[art_idx, 1:NUM_TOPICS]))

################# 4. Calculate cosine similarity between user read articles and unread articles ######################
articles_ranking                         = articles[,c("Article_Id","Title")]
articles_ranking$Cosine_Similarity_Score = NA

for (i in 1:(nrow(articles_ranking))){
  articles_ranking[i,3] = sum(article_topics[i,1:NUM_TOPICS]*user_article_topics)/
    (sqrt(sum((article_topics[i,1:NUM_TOPICS])^2))*sqrt(sum((user_article_topics)^2)))
}

head(articles_ranking)

# Remove the articles which are already read by the user 
articles_ranking = subset(articles_ranking, !(Article_Id %in% USER_LIST))

# Sorting based on the cosine similarity score
articles_ranking = articles_ranking[order(-articles_ranking$Cosine_Similarity_Score),]

## Recommendations based on topic modelling 
articles_recommended_id = articles_ranking[1:NUM_RECOMMENDED_ARTICLES,1]
top_n_articles = articles[which(articles$Article_Id %in% articles_recommended_id),c(1,2)]

## Print user articles 
user_read_articles = articles[which(articles$Article_Id %in% USER_LIST),c(1,2)]
print('User read articles')
print(user_read_articles)

## Print recoommended articles 
print(top_n_articles)




















