#########Loading the required libraries and installing the missing ones ###################
load.libraries <- c('ggplot2', 'randomForest', 'caret', 'rpart','plyr', 'gbm','rpart.plot','reshape2', 'naivebayes','corrplot','e1071')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

####### Reading the data from csv #########
file_path  <- ('/home/karan/bangalore_conf/kddcup.data_10_percent1.csv')
input_data <- read.csv(file_path,stringsAsFactors = FALSE,header = TRUE)

######Removing duplicates from data####
input_data <- unique(input_data)

#####Structure of data ######
dim(input_data)
str(input_data)

###### Check for Missing data ########
cols_wid_miss_vals <- sapply(df, function(x) sum(is.na(x)/nrow(df))) 

#####Grouping attack labels into sub categories#########
input_data$label[input_data$label %in% c('back.','land.','neptune.','pod.','smurf.',
                                         'teardrop.')] <- 'dos'
input_data$label[input_data$label %in% c('buffer_overflow.','loadmodule.','perl.','rootkit.')] <- 'utr'
input_data$label[input_data$label %in% c('ftp_write.','guess_passwd.','imap.','multihop.'
                                         ,'phf.','spy.','warezclient.','warezmaster.')] <- 'rtl'
input_data$label[input_data$label %in% c('satan.','ipsweep.','nmap.','portsweep.')] <- 'probes'

#####Grouping all attacks togeother##############
input_data$label[input_data$label %in% c('dos','utr','probes','rtl')] <- 'attacks'

######Cleaning label data ###
input_data$label[input_data$label %in% c('normal.')] <- 'normal'

### Converting columns to factors #### 
cols_to_factors             <- c('protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login','label')
input_data[cols_to_factors] <- lapply(input_data[cols_to_factors],factor)

###### Distribution of the class variable ###########
label_info <- table(input_data$label)
barplot(label_info,ylim=c(0,85000), xlab="Label", ylab="Frequency", col="black",
        main="Label Distribution")

####Distribution of features with class variable######
categorical_cols <- names(input_data)[ sapply(input_data, is.factor)]
categorical_cols <- categorical_cols[-which(categorical_cols %in%  c("label"))]
numeric_cols     <- names(input_data)[ sapply(input_data, is.numeric)]

for(col_name in categorical_cols)
{
  counts <- table(input_data$label,input_data[[col_name]])
  barplot(counts,xlab=col_name,col=c("red", "lightblue"),
          beside=TRUE,
          ylab ="frequency", main = col_name)
  legend("topleft",c("Attack","Normal"),fill = c("red","lightblue"))
}

for (col_name in numeric_cols)
{
  minm<-min(input_data[[col_name]])  
  maxm<-max(input_data[[col_name]])
  boxplot(input_data[[col_name]]~input_data$label, col = "lightblue", xlab=col_name, ylab="frequency", main = col_name)
}

#### Columns to be Removed######
input_data <- input_data [ ,-which(names(input_data) %in% c('is_host_login','num_outbound_cmds','service'))]

#### Correlation matrix #####
numeric_cols      <- names(input_data)[ sapply(input_data, is.numeric)]
df_num            <- input_data[,numeric_cols]
correlationMatrix <- cor(df_num)
row_indic         <- apply(correlationMatrix, 1, function(x) sum(x > 0.8 | x < -0.8) > 1)
correlations      <- correlationMatrix[row_indic ,row_indic ]
corrplot(correlations, method="square")

######Creating train-test split using stratified sampling########### 
intot      <- createDataPartition(y = input_data$label, p= 0.7, list = FALSE)
training   <- as.data.frame(input_data[intot,])
testing    <- as.data.frame(input_data[-intot,])
test_label <- testing$label
testing    <- testing[-which(names(testing) %in% c('label'))]

#####Building Naive-bayes Model#########
nb_model    <- naive_bayes(training$label ~ ., data=training)
nb_pred     <- predict(nb_model,testing)
nb_cnmatrix <- confusionMatrix(table(nb_pred,test_label))

####Building logistic-regression Model #############
lr_model    <- glm(training$label ~. ,data=training,family = binomial(logit))
lr_pred     <- predict.glm(lr_model,testing,type = 'response')
lr_cnmatrix <- table(test_label,lr_pred > 0.5)
 

####Building Random Forest Model#######
rf_model    <- randomForest(training$label ~ ., data=training,importance=TRUE,ntree=100)
rf_pred     <- predict(rf_model,testing)
rf_cnmatrix <- confusionMatrix(table(rf_pred,test_label))

###Building SVM #####
svm_model <- svm(training$label ~ ., data=training)
svm_pred     <- predict(svm_model,testing)
svm_cnmatrix <- confusionMatrix(table(svm_pred,test_label))

####Building Decision Tree #####
dtree_model<-rpart(training$label ~ ., data=training)
rpart.plot(dtree_model)
dtree_pred <- predict(dtree_model,testing, type = "class")
dtree_cnmatrix <- confusionMatrix(table(dtree_pred,test_label))


###Gradient Boosted and XG BOOST ###


library(dummies)
training_v2 <- dummy.data.frame(training,sep='.',names=c('protocol_type','flag','logged_in','land','is_guest_login'))
x <- names(training_v2)[ sapply(training_v2, is.factor)]
x<- x[x != "label"]

training_v2[x] <- lapply(training_v2[x],numeric)

gb_model = gbm((unclass(training_v2$label)-1) ~. , data=training_v2, shrinkage=0.01, cv.folds=5, n.trees=70, verbose=F)


gb_pred <- predict.gbm(gb_model,testing)




