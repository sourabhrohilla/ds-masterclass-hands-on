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
#2. User_list: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result
#4. alpha: Weightage parameters for Named entity based vector

Path_News_Articles="~/Desktop/Banglaore learning/Recommender system/news_articles.csv"
User_list=c(7,6,76,61,761)
N=5
alpha = 0.5

#########Loading the required libraries and installing the missing ones ###################

load.libraries <- c('tm', 'topicmodels', 'MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC',
                    'rJava','NLP','openNLP','RWeka','magrittr')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

# OpenNLPModels.en
loadOpenNLP.libraries <- c('openNLPmodels.en')
install.lib <- loadOpenNLP.libraries[!loadOpenNLP.libraries %in% installed.packages()]
for(libs in install.lib){install.packages("openNLPmodels.en",
                                          repos = "http://datacube.wu.ac.at/",
                                          type = "source")} 
sapply(loadOpenNLP.libraries, require, character = TRUE)

# Load the articles 

News_Articles<-read.csv(Path_News_Articles)
head(News_Articles)

#Select relevant columns and remove rows with missing values

News_Articles = News_Articles[,c('Article_Id','Title','Content')]
Articles = News_Articles[complete.cases(News_Articles),]
Articles$Content[0] # an uncleaned article

################### 1. Extracting NEs from user read article text #############################
idx<-which(Articles$Article_Id %in% User_list)
User_articles = Articles[idx,]
#Combining user preferred articles together 
text<- paste(User_articles$Content, collapse=" ")

# In user text, we will retain only Organization/Location/Person entities from the user text
text <- as.String(text)

word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()
person_ann <- Maxent_Entity_Annotator(kind = "person")
location_ann <- Maxent_Entity_Annotator(kind = "location")
organization_ann <- Maxent_Entity_Annotator(kind = "organization")
pipeline <- list(sent_ann, word_ann,person_ann, location_ann,organization_ann)
text_annotations <- NLP::annotate(text, pipeline)
text_doc <- AnnotatedPlainTextDocument(text, text_annotations)

# Extract entities from an AnnotatedPlainTextDocument
entities <- function(doc, kind) {
  s <- doc$content
  a <- annotations(doc)[[1]]
  if(hasArg(kind)) {
    k <- sapply(a$features, `[[`, "kind")
    s[a[k == kind]]
  } else {
    s[a[a$type == "entity"]]
  }
}

text_entities = list()
persons = (entities(text_doc, kind = "person"))
locations = (entities(text_doc, kind = "location"))
organizations = (entities(text_doc, kind = "organization"))
user_Text_Ner<- paste(c(persons, locations, organizations), collapse = " ")
print(user_Text_Ner)

#### 2. Text processing of user read articles and generating TFIDF value vector  #####

# TfIdf value vector for user text 
user_article_tfidf
  
## TfIdf value vector for user NER text 
# Cleaning and tokenizing user read text
User_NER_text = itoken(user_Text_Ner, 
                   preprocessor = removepunctuation, 
                   tokenizer = stem_tokenizer, 
                   progressbar = FALSE)

# Generating TfIDf vector for user read articles 
user_article_NER_tfidf = create_dtm(User_text, vectorizer) %>% transform(tfidf)

#User_Vector =>  (Alpha) [TF-IDF Vector] + (1-Alpha) [NER Vector] 

user_vector = alpha*(user_article_tfidf ) + (1-alpha)*user_article_NER_tfidf

######### 5. Calculate cosine similarity between user read articles and unread articles ######################

Articles_ranking                         = Articles[,c("Article_Id","Title")]
Articles_ranking$Cosine_Similarity_Score = NA
i=1
for (i in 1:(nrow(Articles_ranking))){
  Articles_ranking[i,3] = sum(article_tfidf_matrix[i,]*user_vector)/((sqrt(sum(article_tfidf_matrix[i,]*article_tfidf_matrix[i,])))*(sqrt(sum(user_vector *user_vector))))
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

######################### when alpha is equal to zero ######################

alpha = 0
user_vector = alpha*user_article_tfidf + (1-alpha)*user_article_NER_tfidf
dim(user_vector)

Articles_ranking                         = Articles[,c("Article_Id","Title")]
Articles_ranking$Cosine_Similarity_Score = NA
i=1
for (i in 1:(nrow(Articles_ranking))){
  Articles_ranking[i,3] = sum(article_tfidf_matrix[i,]*user_vector)/((sqrt(sum(article_tfidf_matrix[i,]*article_tfidf_matrix[i,])))*(sqrt(sum(user_vector *user_vector))))
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