setwd("~/Desktop/Banglaore learning/Recommender system")
library(tm)
library(topicmodels)
library(MASS)
library(NLP)
library(R.utils)
library(stringdist)
library(dplyr) 

rm(list = setdiff(ls(), "x"))

#x<-fread(file.choose())
Articles<-read.csv('newsaritcles.csv')

## User input 
User_list = c(1,3,15)
idx<-which(Articles$Article_ID %in% User_list)
User_articles = Articles[idx,]
N = 5

#Combining all articles together 
text<- paste(User_articles$Content, collapse=" ")
user_articles_id = max(Articles$Article_ID)+1
User_content = data.frame(user_articles_id,text)
colnames(User_content)<-c('Article_ID','Content')

# combining the user articles to the corpus
Content = rbind(Articles[,c('Article_ID','Content')], User_content)
head(Articles[,c('Content','Article_ID')])


# creating a corpus from the pre-processed data frame and further post-processing
#Setting up source and Corpus(list of content of articles)
myCorpus = Corpus(VectorSource(Content$Content))
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

# creating document term matrix
Dtm_TfIdf <- DocumentTermMatrix(myCorpus, 
                       control = list(wordLengths=c(2,Inf), 
                                         weighting=function(x)
                                           weightTfIdf(x, normalize =FALSE)))
TfIDf = as.matrix(Dtm_TfIdf)
rownames(TfIDf) = Content$Article_ID

User_article_row = length(TfIDf[,1])

Articles_ranking = data.frame('Article_ID'= integer(0), 'Score'= numeric(0))
for (i in 1:(User_article_row)){
  print(i)
  Articles_ranking[i,2] = (TfIDf[i,]%*%TfIDf[User_article_row,])/((sqrt(sum(TfIDf[i,]*TfIDf[i,])))*(sqrt(sum(TfIDf[User_article_row,]*TfIDf[User_article_row,]))))
}
user_list_ID = max(Content$Article_ID)
User_list = c(User_list,user_list_ID)

Articles_ranking = subset(Articles_ranking, !(Article_ID %in% User_list))

# Top 3 articles 
Top_Articles_ranking[order(-Articles_ranking$Score),][1:3,'Article_ID']

## Recommendations based on topic modelling 
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


