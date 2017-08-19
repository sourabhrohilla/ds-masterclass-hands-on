#########Loading the required libraries and installing the missing ones ###################
load.libraries <- c('ggplot2', 'randomForest', 'caret','dplyr','DMwR', 'rpart','plyr', 'gbm','rpart.plot','reshape2', 'naivebayes','corrplot','e1071')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

####### Reading the data from csv #########
# Enter your own path containing the dataset.
file_path  <- ('/home/karan/bangalore_conf/kddcup.data_10_percent1.csv')
input_data <- read.csv(file_path,stringsAsFactors = FALSE,header = TRUE)

#####Structure of data ######
dim(input_data)
str(input_data)

######Removing duplicates from data####
dim(unique(input_data))
input_data <- unique(input_data)

###### Check for Missing data ########
cols_with_missing_vals <- sapply(input_data, function(x) sum(is.na(x)/nrow(input_data))) 

#####Grouping attack labels into sub categories#########
input_data$label[input_data$label %in% c('back.','land.','neptune.','pod.','smurf.',
                                         'teardrop.')] <- 'dos'
input_data$label[input_data$label %in% c('buffer_overflow.','loadmodule.','perl.','rootkit.')] <- 'utr'
input_data$label[input_data$label %in% c('ftp_write.','guess_passwd.','imap.','multihop.'
                                         ,'phf.','spy.','warezclient.','warezmaster.')] <- 'rtl'
input_data$label[input_data$label %in% c('satan.','ipsweep.','nmap.','portsweep.')] <- 'probes'

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
  legend("topright",c("dos","normal","probes","rtl","utr"),fill = c("red","lightblue"))
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
numeric_cols      <- names(input_data)[sapply(input_data, is.numeric)]
correlationMatrix <- cor(input_data[,numeric_cols])
relevant_correlations <- as.matrix(subset(melt(correlationMatrix, na.rm = TRUE), abs(value) > 0.8 & abs(value) <1))

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
svm_model    <- svm(training$label ~ ., data=training)
svm_pred     <- predict(svm_model,testing)
svm_cnmatrix <- confusionMatrix(table(svm_pred,test_label))

####Building Decision Tree #####
dtree_model<-rpart(training$label ~ ., data=training,cp=0.0009)
rpart.plot(dtree_model,tweak=1.5)
dtree_pred <- predict(dtree_model,testing, type = "class")
dtree_cnmatrix <- confusionMatrix(table(dtree_pred,test_label))


####Oversampling ########
get_smote_variables <- function(data,col_name){
  #returns perc_over and perc_under for binary class SMOTE
  class_dist = table(data[col_name])
  majority_class_count = max(class_dist)
  minority_class_count = min(class_dist)
  perc_over = majority_class_count*100/minority_class_count
  perc_under = (1+(minority_class_count/majority_class_count))*100
  return (c(perc_over,perc_under))
}
modified_data <- input_data 
modified_data$new_label[modified_data$label %in% c('dos','rtl','probes','normal')] <- 'others' 
modified_data$new_label[modified_data$label %in% c('utr')] <- 'utr'
modified_data <- modified_data[,-which(names(modified_data) %in% c('label'))]
modified_data$new_label <- as.factor(modified_data$new_label)
var_list <- get_smote_variables(modified_data,'new_label')


######Creating train-test split using stratified sampling########### 
intot          <- createDataPartition(y = modified_data$new_label, p= 0.7, list = FALSE)
mod_training   <- as.data.frame(modified_data[intot,])
mod_testing    <- as.data.frame(modified_data[-intot,])
mod_test_label <- mod_testing$new_label
mod_testing    <- mod_testing[-which(names(mod_testing) %in% c('new_label'))]

#####Building Naive-bayes Model#########
mod_nb_model    <- naive_bayes(mod_training$new_label ~ ., data=mod_training)
mod_nb_pred     <- predict(mod_nb_model,mod_testing)
mod_nb_cnmatrix <- confusionMatrix(table(mod_nb_pred,mod_test_label))


smote_modified_data <- SMOTE(modified_data$new_label ~ ., modified_data, perc.over = var_list[1],perc.under=var_list[2])


######Creating train-test split using stratified sampling########### 
intot                <- createDataPartition(y = smote_modified_data$new_label, p= 0.7, list = FALSE)
smote_mod_training   <- as.data.frame(smote_modified_data[intot,])
smote_mod_testing    <- as.data.frame(smote_modified_data[-intot,])
smote_mod_test_label <- smote_mod_testing$new_label
smote_mod_testing    <- smote_mod_testing[-which(names(smote_mod_testing) %in% c('new_label'))]

#####Building Naive-bayes smote_model#########
smote_mod_nb_model    <- naive_bayes(smote_mod_training$new_label ~ ., data=smote_mod_training)
smote_mod_nb_pred     <- predict(smote_mod_nb_model,smote_mod_testing)
smote_mod_nb_cnmatrix <- confusionMatrix(table(smote_mod_nb_pred,smote_mod_test_label))
