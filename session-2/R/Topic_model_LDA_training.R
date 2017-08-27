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
Path_News_Articles="~/Desktop/R/news_articles.csv"
k = 150  # No of topics

###### 1.Represent articles in terms of bag of words ##############
#1.Reading the csv file to get the Article id, Title and News Content
#2.Remove punctuation marks and other symbols from each article
#3.Tokenize each article
#4. Stem token of every article

News_Articles=read.csv(Path_News_Articles)
head(News_Articles)

#Select relevant columns and remove rows with missing values

News_Articles = News_Articles[,c('Article_Id','Title','Content')]
Articles = News_Articles[complete.cases(News_Articles),]
Articles$Content[0] # an uncleaned article

#Creating corpus
ArticleCorpus = VCorpus(VectorSource(Articles$Content))

# Text cleaning using transformation function tm_map
toSpace = content_transformer(function(x, pattern) gsub(pattern, "", x)) 
ArticleCorpus= tm_map(ArticleCorpus, toSpace, "/|@|\\|")
ArticleCorpus = tm_map(ArticleCorpus, removeNumbers)
removeHyphen = function(x) gsub("-", " ", x)  #Replace hyphen with space
ArticleCorpus = tm_map(ArticleCorpus, removeHyphen)
ArticleCorpus = tm_map(ArticleCorpus, removePunctuation) 
ArticleCorpus = tm_map(ArticleCorpus, stripWhitespace)
ArticleCorpus = tm_map(ArticleCorpus, removeWords, stopwords("english"))
ArticleCorpus = tm_map(ArticleCorpus,stemDocument)
#writeLines(as.character(ArticleCorpus[2]))

#Tokenise
removeRegex= function(x) strsplit(gsub("[^[:alnum:] ]", "", x), " +") 
ArticleCorpus= tm_map(ArticleCorpus, removeRegex)
writeLines(as.character(ArticleCorpus[2]))
ArticleCorpus = tm_map(ArticleCorpus, PlainTextDocument)


################ 2. Training of LDA model using Article corpus ###################################
# creating document term matrix
Dtm = DocumentTermMatrix(ArticleCorpus, 
                          control = list(wordLengths=c(2,Inf), 
                                         weighting=weightTf))
Corpus_Terms = Terms(Dtm)
# running lda with k topics
Dtm_LDA = LDA(Dtm,k,method = "Gibbs")
Topic_terms = as.data.frame(t(posterior(Dtm_LDA)$terms))
Article_topics =  as.data.frame(Dtm_LDA@gamma)
View_topic_list = get_terms(Dtm_LDA,15)

#setwd("~/Desktop/Banglaore learning/Recommender system")
write.csv(Article_topics, 'Article_topics_150_topics.csv')
write.csv(Topic_terms, 'Topic_terms_150_topics.csv')
save.image(file = "~/Desktop/Banglaore learning/Recommender system/Lda_model_150_topics.Rdata")

