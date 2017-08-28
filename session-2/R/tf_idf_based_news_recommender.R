############### TF-IDF Based Recommender ###################
#1.Represent articles in terms of word vectors. 
#2.Generate TF-IDF matrix for user read articles 
#3.Represent user in terms of read/preferred articles associated words and TFIDF values of the vector
#4.Calculate cosine similarity between user read articles and unread articles
#5.Get the recommended articles

################ Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. USER_READ_LIST: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result

PATH_NEWS_ARTICLES="/home/karan/Downloads/news_articles.csv"
USER_READ_LIST=c(1,959,2500,82,174)
NUM_RECOMMENDED_ARTICLES=5

#########Loading the required libraries and installing the missing ones ###################
load.libraries = c('tm', 'topicmodels', 'MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC', 'text2vec')
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

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

# define preprocessing functions and tokenization function
remove_punctuation = function(x) gsub("[[:punct:]]", "",x)

remove_stopwords = function(x) x[!x %in% stopwords("en")]
stem_tokenizer = function(x) {
  x = tolower(x)
  tokens = word_tokenizer(x)
  lapply(tokens, SnowballC::wordStem, language="en")
  lapply(tokens, function(x)(x[!(x %in% stopwords("en"))]))
}

# Tokenizing the content of the articles 
articles$Content = as.character(articles$Content)
articles_cleaned = itoken(articles$Content, 
                  preprocessor = remove_punctuation, 
                  tokenizer = stem_tokenizer, 
                  ids = articles$Article_Id,
                  progressbar = FALSE)
## 
articles_vocab = create_vocabulary(articles_cleaned)
vectorizer = vocab_vectorizer(articles_vocab)
dtm_articles = create_dtm(articles_cleaned, vectorizer)

############### 2. Generate TF-IDF matrix for the articles ################
# define tfidf model
tfidf = TfIdf$new()
article_tfidf_matrix = fit_transform(dtm_articles, tfidf)

############### 3.Represent user in terms of read/preferred articles and as vector with TfIdf values #######

# Extracting user read articles 
idx=which(articles$Article_Id %in% USER_READ_LIST)
user_articles = articles[idx,]
text= paste(user_articles$Content, collapse=" ")

# Cleaning and tokenizing user read text
user_text = itoken(text, 
                 preprocessor = remove_punctuation, 
                 tokenizer = stem_tokenizer, 
                 progressbar = FALSE)

# Generating TfIDf vector for user read articles 
user_article_tfidf = create_dtm(user_text, vectorizer) %>% transform(tfidf)


############## 4. Calculate cosine similarity between user read articles and unread articles ###############
articles_ranking                         = articles[,c("Article_Id","Title")]
articles_ranking$Cosine_Similarity_Score = NA

for (i in 1:(nrow(articles_ranking))){
  articles_ranking[i,3] = sum(article_tfidf_matrix[i,]*user_article_tfidf)/((sqrt(sum(article_tfidf_matrix[i,]*article_tfidf_matrix[i,])))*(sqrt(sum(user_article_tfidf *user_article_tfidf))))
}

head(articles_ranking)

# Remove the articles which are already read by the user 
articles_ranking = subset(articles_ranking, !(Article_Id %in% USER_READ_LIST))

# Sorting based on the cosine similarity score
articles_ranking = articles_ranking[order(-articles_ranking$Cosine_Similarity_Score),]

## Recommendations based on tfidf
articles_recommended_id = articles_ranking[1:NUM_RECOMMENDED_ARTICLES,1]
top_n_articles = articles[which(articles$Article_Id %in% articles_recommended_id),c(1,2)]

## Print user articles 
user_read_articles = articles[which(articles$Article_Id %in% USER_READ_LIST),c(1,2)]
print('User read articles')
print(user_read_articles)

## Print recoommended articles 
print(top_n_articles)



