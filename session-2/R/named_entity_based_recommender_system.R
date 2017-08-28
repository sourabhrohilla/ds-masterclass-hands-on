########## Named Entity based Recommender System ###########
# Recommender System based on Named Entities as representation of documents
# Named Entity Based Recommender
#1.Derived named entities in user read/preferred articles 
#2.Append the user read articles to existing corpus and represent articles in terms TF-IDF Matrix
#2.Represent user in terms of -
#  (Alpha) <TF-IDF Vector> + (1-Alpha) <NER Vector>
# where Alpha => [0,1] 
#3.[TF-IDF Vector] => TF-IDF vector representation of concatenated read articles
#4.[NER Vector] => TF-IDF vector representation of NERs associated with concatenated read articles 
#5.Calculate cosine similarity between user vector and articles TF-IDF matrix 
#6.Get the recommended articles 
#7.What if Alpha is 0 when defining user vector?

################ Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. USER_READ_LIST: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result
#4. alpha: Weightage parameters for Named entity based vector

PATH_NEWS_ARTICLES="/home/karan/Downloads/news_articles.csv"
USER_READ_LIST=c(7,6,76,61,761)
NUM_RECOMMENDED_ARTICLES=5
ALPHA = 0.5

#########Loading the required libraries and installing the missing ones ###################

# OpenNLPModels.en
loadOpenNLP.libraries = c('openNLPmodels.en')
install.lib = loadOpenNLP.libraries[!loadOpenNLP.libraries %in% installed.packages()]
for(libs in install.lib){install.packages("openNLPmodels.en",
                                          repos = "http://datacube.wu.ac.at/",
                                          type = "source")} 
sapply(loadOpenNLP.libraries, require, character = TRUE)


load.libraries = c('tm', 'topicmodels', 'MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC',
                   'rJava','NLP','openNLP','RWeka','magrittr')
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)


# Load the articles 

news_articles = read.csv(PATH_NEWS_ARTICLES)
head(news_articles)

#Select relevant columns and remove rows with missing values

articles = news_articles[,c('Article_Id','Title','Content')]
articles = articles[complete.cases(news_articles),]
articles$Content[0] # an uncleaned article

################### 1. Extracting NEs from user read article text #############################
idx = which(articles$Article_Id %in% USER_READ_LIST)
user_articles = articles[idx,]
#Combining user preferred articles together 
text = paste(user_articles$Content, collapse=" ")

# In user text, we will retain only Organization/Location/Person entities from the user text
text = as.String(text)

word_ann = Maxent_Word_Token_Annotator()
sent_ann = Maxent_Sent_Token_Annotator()
person_ann = Maxent_Entity_Annotator(kind = "person")
location_ann = Maxent_Entity_Annotator(kind = "location")
organization_ann = Maxent_Entity_Annotator(kind = "organization")
pipeline = list(sent_ann, word_ann,person_ann, location_ann,organization_ann)
text_annotations = NLP::annotate(text, pipeline)
text_doc = AnnotatedPlainTextDocument(text, text_annotations)

# Extract entities from an AnnotatedPlainTextDocument
entities = function(doc, kind) {
  doc_content = doc$content
  doc_annotations = annotations(doc)[[1]]
  if(hasArg(kind)) {
    entity_kind = sapply(doc_annotations$features, `[[`, "kind")
    doc_content[doc_annotations[entity_kind == kind]]
  } else {
    doc_content[doc_annotations[doc_annotations$type == "entity"]]
  }
}


text_entities = list()
persons = (entities(text_doc, kind = "person"))
locations = (entities(text_doc, kind = "location"))
organizations = (entities(text_doc, kind = "organization"))
user_text_ner= paste(c(persons, locations, organizations), collapse = " ")
print(user_text_ner)

#### 2. Text processing of user read articles and generating TFIDF value vector  #####

# TfIdf value vector for user text 
user_article_tfidf

## TfIdf value vector for user NER text 
# Cleaning and tokenizing user read text
user_ner_text = itoken(user_text_ner, 
                       preprocessor = remove_punctuation, 
                       tokenizer = stem_tokenizer, 
                       progressbar = FALSE)

# Generating TfIDf vector for user read articles 
user_article_ner_tfidf = create_dtm(user_ner_text, vectorizer) %>% transform(tfidf)

#User_Vector =>  (Alpha) [TF-IDF Vector] + (1-Alpha) [NER Vector] 

user_vector = alpha*(user_article_tfidf ) + (1-alpha)*user_article_ner_tfidf[1,]

######### 5. Calculate cosine similarity between user read articles and unread articles ######################

articles_ranking                         = articles[,c("Article_Id","Title")]
articles_ranking$cosine_similarity_score = NA
i=1
for (i in 1:(nrow(articles_ranking))){
  articles_ranking[i,3] = sum(article_tfidf_matrix[i,]*user_vector)/((sqrt(sum(article_tfidf_matrix[i,]*article_tfidf_matrix[i,])))*(sqrt(sum(user_vector *user_vector))))
}
head(articles_ranking)


# Remove the articles which are already read by the user 
articles_ranking = subset(articles_ranking, !(article_id %in% USER_READ_LIST))

# Sorting based on the cosine similarity score
articles_ranking = articles_ranking[order(-articles_ranking$cosine_similarity_score),]

## Recommendations based on tfidf ner
articles_recommended_id = articles_ranking[1:NUM_RECOMMENDED_ARTICLES,1]
top_n_articles = articles[which(articles$article_id %in% articles_recommended_id),c(1,2)]

## Print user articles 
user_read_articles = articles[which(articles$article_id %in% USER_READ_LIST),c(1,2)]
print('User read articles')
print(user_read_articles)

## Print recoommended articles 
print(top_n_articles)

######################### when alpha is equal to zero ######################

alpha = 0
user_vector = alpha*user_article_tfidf + (1-alpha)*user_article_ner_tfidf
dim(user_vector)

articles_ranking                         = articles[,c("Article_Id","Title")]
articles_ranking$cosine_similarity_score = NA
i=1
for (i in 1:(nrow(articles_ranking))){
  articles_ranking[i,3] = sum(article_tfidf_matrix[i,]*user_vector)/((sqrt(sum(article_tfidf_matrix[i,]*article_tfidf_matrix[i,])))*(sqrt(sum(user_vector *user_vector))))
}
head(articles_ranking)


# Remove the articles which are already read by the user 
articles_ranking = subset(articles_ranking, !(article_id %in% USER_READ_LIST))
# Sorting based on the cosine similarity score
articles_ranking = articles_ranking[order(-articles_ranking$cosine_similarity_score),]

## Recommendations based on topic modelling 
articles_recommended_id = articles_ranking[1:NUM_RECOMMENDED_ARTICLES,1]
top_n_articles = articles[which(articles$article_id %in% articles_recommended_id),c(1,2)]

## Print user articles 
user_read_articles = articles[which(articles$article_id %in% USER_READ_LIST),c(1,2)]
print('User read articles')
print(user_read_articles)

## Print recoommended articles 
print(top_n_articles)