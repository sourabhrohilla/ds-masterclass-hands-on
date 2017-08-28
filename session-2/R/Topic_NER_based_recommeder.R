####### Topics + NER Recommender
#1.Represent user in terms of - 
#  (Alpha) <Topic Vector> + (1-Alpha) <NER Vector> <br/>
#  where 
#2.Alpha => [0,1] 
#3.[Topic Vector] => Topic vector representation of concatenated read articles 
#4.[NER Vector] => Topic vector representation of NERs associated with concatenated read articles 
#5.Calculate cosine similarity between user vector and articles Topic matrix
#6.Get the recommended articles


############### Describing parameters ######################
#1. PATH_NEWS_ARTICLES: specify the path where news_article.csv is present 
#2. User_list: List of Article_Ids read by the user 
#3. No_Recommended_Articles, N: Refers to the number of recommended articles as a result
#4. alpha: Weightage parameters for Named entity based vector

Path_News_Articles="/home/karan/Downloads/news_articles.csv"
User_list=c(85,526,896,1596,2431)
N=5
alpha =0.5

#########Loading the required libraries and installing the missing ones ###################

load.libraries = c('tm', 'topicmodels','lda','MASS','NLP','R.utils', 'stringdist','dplyr','SnowballC',
                    'rJava','NLP','openNLP','RWeka','magrittr')
install.lib = load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

# OpenNLPModels.en
loadOpenNLP.libraries = c('openNLPmodels.en')
install.lib = loadOpenNLP.libraries[!loadOpenNLP.libraries %in% installed.packages()]
for(libs in install.lib){install.packages("openNLPmodels.en",
                                          repos = "http://datacube.wu.ac.at/",
                                          type = "source")} 
sapply(loadOpenNLP.libraries, require, character = TRUE)


################ 1. Text processing of user read articles ########################
#1. Represent User in terms of Topic Distribution and NER
#Represent user in terms of read article topic distribution
#Represent user in terms of NERs associated with read articles
#2.1 Get NERs of read articles
#2.2 Load LDA model
#2.3 Get topic distribution for the concated NERs
#3. Generate user vector

# Extracting NEs from user read article text
idx=which(articles$Article_Id %in% User_list)
User_articles = articles[idx,]
#Combining user preferred articles together 
text= paste(User_articles$Content, collapse=" ")

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
user_Text_Ner= paste(c(persons, locations, organizations), collapse = " ")
print(user_Text_Ner)

#Creating corpus
UserNerCorpus = VCorpus(VectorSource(user_Text_Ner))

# Text cleaning using transformation function tm_map
toSpace = content_transformer(function(x, pattern) gsub(pattern, "", x)) 
UserNerCorpus= tm_map(UserNerCorpus, toSpace, "/|@|\\|")
UserNerCorpus = tm_map(UserNerCorpus, removeNumbers)
removeHyphen = function(x) gsub("-", " ", x)  #Replace hyphen with space
UserNerCorpus = tm_map(UserNerCorpus, removeHyphen)
UserNerCorpus = tm_map(UserNerCorpus, removePunctuation) 
UserNerCorpus = tm_map(UserNerCorpus, stripWhitespace)
UserNerCorpus = tm_map(UserNerCorpus, removeWords, stopwords("english"))
UserNerCorpus = tm_map(UserNerCorpus,stemDocument)
#writeLines(as.character(UserNerCorpus[2]))

#Tokenise
removeRegex= function(x) strsplit(gsub("[^[:alnum:] ]", "", x), " +") 
UserNerCorpus= tm_map(UserNerCorpus, removeRegex)
#writeLines(as.character(UserNerCorpus[2]))
UserNerCorpus = tm_map(UserNerCorpus, PlainTextDocument)

################ 2. Load the trained LDA model on Article corpus and represent NER of user text as a vector ##############
# Load the trained model
#Article_topics = read.csv('Article_topics_150_topics.csv')
# Load the trained LDA model
#load("Lda_model_150_topics.Rdata")

user_article_NER_tf  = DocumentTermMatrix(UserNerCorpus, 
                           control = list(wordLengths=c(2,Inf),dictionary = corpus_terms,
                                          weighting=weightTf))

###### 3. User topic vector based on NEs of user read article text #############

# Topic vector of user_text_Ner using trained LDA model 
user_article_NER_tf= (as.matrix(user_article_NER_tf))

# predicting topics for user_NER_txt using trained LDA model 

user_NER_article_topics = posterior(dtm_LDA,user_article_NER_tf)
user_NER_article_topics = (user_NER_article_topics[[2]])

# user article topic vector 
article_topics$Article_ID = articles$Article_Id
art_idx= which(article_topics$Article_ID %in% User_list)
user_article_topics= matrix(colMeans(article_topics[art_idx, 1:NUM_TOPICS]))

# User_Vector =>  (Alpha) [TF-IDF Vector] + (1-Alpha) [NER Vector] 
user_vector = alpha*(t(user_article_topics)) + (1-alpha)*user_NER_article_topics

######### 4. Calculate cosine similarity between user read articles and unread articles ######################

Articles_ranking                         = articles[,c("Article_Id","Title")]
Articles_ranking$Cosine_Similarity_Score = NA

for (i in 1:(nrow(Articles_ranking))){
  Articles_ranking[i,3] = sum(article_topics[i,1:NUM_TOPICS]*user_vector)/
    (sqrt(sum((article_topics[i,1:NUM_TOPICS])^2))*sqrt(sum((user_vector)^2)))
}
head(Articles_ranking)

# Remove the articles which are already read by the user 
Articles_ranking = subset(Articles_ranking, !(Article_Id %in% User_list))

# Sorting based on the cosine similarity score
Articles_ranking = Articles_ranking[order(-Articles_ranking$Cosine_Similarity_Score),]

## Recommendations based on topic modelling 
articles_recommended_id = Articles_ranking[1:N,1]
Top_n_articles = articles[which(articles$Article_Id %in% articles_recommended_id),c(1,2)]

## Print user articles 
User_read_articles = articles[which(articles$Article_Id %in% User_list),c(1,2)]
print('User read articles')
print(User_read_articles)

## Print recoommended articles 
print(Top_n_articles)

