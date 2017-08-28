########## Topic Modeling Using LDA ##############
#1.Text Processing of articles 
#2.Represent articles in terms of Topic Vector using LDA model (Save the trained model)

#########Loading the required libraries and installing the missing ones ###################
load.libraries = c('tm', 'topicmodels','lda','MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC')
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

############### Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
PATH_NEWS_ARTICLES="/home/karan/Downloads/news_articles.csv"
NUM_TOPICS = 150  # No of topics

###### 1.Represent articles in terms of bag of words ##############
#1.Reading the csv file to get the Article id, Title and News Content
#2.Remove punctuation marks and other symbols from each article
#3.Tokenize each article
#4. Stem token of every article

news_articles=read.csv(PATH_NEWS_ARTICLES)
head(news_articles)

#Select relevant columns and remove rows with missing values

news_articles = news_articles[,c('Article_Id','Title','Content')]
articles = news_articles[complete.cases(news_articles),]
articles$Content[0] # an uncleaned article

#Creating corpus
article_corpus = VCorpus(VectorSource(articles$Content))

# Text cleaning using transformation function tm_map
toSpace = content_transformer(function(x, pattern) gsub(pattern, "", x)) 
article_corpus= tm_map(article_corpus, toSpace, "/|@|\\|")
article_corpus = tm_map(article_corpus, removeNumbers)
removeHyphen = function(x) gsub("-", " ", x)  #Replace hyphen with space
article_corpus = tm_map(article_corpus, removeHyphen)
article_corpus = tm_map(article_corpus, removePunctuation) 
article_corpus = tm_map(article_corpus, stripWhitespace)
article_corpus = tm_map(article_corpus, removeWords, stopwords("english"))
article_corpus = tm_map(article_corpus,stemDocument)
#writeLines(as.character(article_corpus[2]))

#Tokenise
removeRegex= function(x) strsplit(gsub("[^[:alnum:] ]", "", x), " +") 
article_corpus= tm_map(article_corpus, removeRegex)
writeLines(as.character(article_corpus[2]))
article_corpus = tm_map(article_corpus, PlainTextDocument)


################ 2. Training of LDA model using Article corpus ###################################
# creating document term matrix
dtm = DocumentTermMatrix(article_corpus, 
                         control = list(wordLengths=c(2,Inf), 
                                        weighting=weightTf))
corpus_terms = Terms(dtm)
# running lda with NUM_TOPICS topics
dtm_LDA = LDA(dtm,NUM_TOPICS,method = "Gibbs")
topic_terms = as.data.frame(t(posterior(dtm_LDA)$terms))
article_topics =  as.data.frame(dtm_LDA@gamma)
view_topic_list = get_terms(dtm_LDA,15)

#setwd("~/Desktop/Banglaore learning/Recommender system")
write.csv(article_topics, 'article_topics_150_topics.csv')
write.csv(topic_terms, 'topic_terms_150_topics.csv')
save.image(file = "~/Desktop/Banglaore learning/Recommender system/Lda_model_150_topics.Rdata")

