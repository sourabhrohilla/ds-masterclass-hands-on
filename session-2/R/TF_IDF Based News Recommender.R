############### TF-IDF Based Recommender ###################
#1.Represent articles in terms of word vectors. 
#2.Generate TF-IDF matrix for user read articles 
#3.Represent user in terms of read/preferred articles associated words and TFIDF values of the vector
#4.Calculate cosine similarity between user read articles and unread articles
#5.Get the recommended articles

################ Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. User_list: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result

Path_News_Articles="news_articles.csv"
User_list=c(1,959,2500,82,174)
N=5

#########Loading the required libraries and installing the missing ones ###################
load.libraries = c('tm', 'topicmodels', 'MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC', 'text2vec','')
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

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
Articles$Content = as.character(Articles$Content)
Articles_cleaned = itoken(Articles$Content, 
                  preprocessor = remove_punctuation, 
                  tokenizer = stem_tokenizer, 
                  ids = Articles$Article_Id,
                  progressbar = FALSE)
## 
Articles_vocab = create_vocabulary(Articles_cleaned)
vectorizer = vocab_vectorizer(Articles_vocab)
Dtm_articles = create_dtm(Articles_cleaned, vectorizer)

############### 2. Generate TF-IDF matrix for the articles ################
# define tfidf model
tfidf = TfIdf$new()
article_tfidf_matrix = fit_transform(Dtm_articles, tfidf)

############### 3.Represent user in terms of read/preferred articles and as vector with TfIdf values #######

# Extracting user read articles 
idx=which(Articles$Article_Id %in% User_list)
User_articles = Articles[idx,]
text= paste(User_articles$Content, collapse=" ")

# Cleaning and tokenizing user read text
User_text = itoken(text, 
                 preprocessor = remove_punctuation, 
                 tokenizer = stem_tokenizer, 
                 progressbar = FALSE)

# Generating TfIDf vector for user read articles 
user_article_tfidf = create_dtm(User_text, vectorizer) %>% transform(tfidf)


############## 4. Calculate cosine similarity between user read articles and unread articles ###############
Articles_ranking                         = Articles[,c("Article_Id","Title")]
Articles_ranking$Cosine_Similarity_Score = NA

for (i in 1:(nrow(Articles_ranking))){
  Articles_ranking[i,3] = sum(article_tfidf_matrix[i,]*user_article_tfidf)/((sqrt(sum(article_tfidf_matrix[i,]*article_tfidf_matrix[i,])))*(sqrt(sum(user_article_tfidf *user_article_tfidf))))
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



