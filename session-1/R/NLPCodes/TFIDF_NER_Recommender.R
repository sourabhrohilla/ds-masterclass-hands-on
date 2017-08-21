setwd("~/Desktop/Banglaore learning/Recommender system")
library(tm)
library(topicmodels)
library(MASS)
library(NLP)
library(R.utils)
library(stringdist)
library(dplyr) 
library(openNLP)

rm(list = setdiff(ls(), "x"))

#x<-fread(file.choose())
Articles<-read.csv('newsaritcles.csv')

## User input 
User_list = c(1,3,15)
idx<-which(Articles$Article_ID %in% User_list)
User_articles = Articles[idx,]

#Combining all articles together 
text<- paste(User_articles$Content, collapse=" ")

#Only keeping Named Entities in the text in the user read articles 

######################################################################################
library(rjava)
library(NLP)
library(openNLP)
library(RWeka)
library(qdap)
library(magrittr)
library(openNLPmodels.en)
#install.packages("openNLPmodels.en",repos = "http://datacube.wu.ac.at/",type = "source")

text <- as.String(text)

word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()
person_ann <- Maxent_Entity_Annotator(kind = "person")
location_ann <- Maxent_Entity_Annotator(kind = "location")
organization_ann <- Maxent_Entity_Annotator(kind = "organization")
pipeline <- list(sent_ann, word_ann,person_ann,location_ann,organization_ann)
text_annotations <- annotate(text, pipeline)
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

#############################################################################


user_articles_id = max(Articles$Article_ID)+1
User_content = data.frame(user_articles_id,user_Text_Ner)
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
Articles_ranking$Article_ID = Content$Article_ID

user_list_ID = max(Content$Article_ID)
User_list = c(User_list,user_list_ID)
        
Articles_ranking = subset(Articles_ranking, !(Article_ID %in% User_list))

# Top 3 articles 
Articles_ranking[order(-Articles_ranking$Score),][1:3,]

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


