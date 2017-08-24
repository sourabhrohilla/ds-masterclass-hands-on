#########Loading the required libraries and installing the missing ones ###################
load.libraries <- c('ggplot2', 'randomForest', 'caret','dplyr','DMwR', 'rpart','plyr', 'gbm','rpart.plot','reshape2', 'naivebayes','corrplot','e1071')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dep = T)
sapply(load.libraries, require, character = TRUE)

####### Reading the data from csv #########
# Enter your own path containing the dataset.
train_file_path  <- ('/home/karan/Downloads/session_1_data_train.csv')
test_file_path  <- ('/home/karan/Downloads/session_1_data_test.csv')
train_data <- read.csv(train_file_path,stringsAsFactors = FALSE,header = TRUE)
test_data <- read.csv(test_file_path,stringsAsFactors = FALSE,header = TRUE)

#####Structure of data ######
dim(train_data)
str(train_data)

######Removing duplicates from data####
dim(unique(train_data))
train_data <- unique(train_data)

###################################
get_attack_type_label <- function(attack_name){
  if(attack_name %in% c('back.','land.','neptune.','pod.','smurf.','teardrop.')){
    return('dos')
  }
  else if(attack_name %in% c('buffer_overflow.','loadmodule.','perl.','rootkit.')){
    return('utr')
  }
  else if(attack_name %in% c('ftp_write.','guess_passwd.','imap.','multihop.',
                             'phf.','spy.','warezclient.','warezmaster.')){
    return('rtl')
  }
  else if(attack_name %in% c('satan.','ipsweep.','nmap.','portsweep.')){
    return('probes')
  }
  else if(attack_name %in% c('normal.')){
    return('normal')
  }
  return('others')
}


###### Check for Missing data ########
cols_with_missing_vals <- sapply(train_data, function(x) sum(is.na(x)/nrow(train_data))) 

train_data$label = mapply(get_attack_type_label,train_data$label)

### Converting columns to factors #### 
cols_to_factors             <- c('protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login','label')
train_data[cols_to_factors] <- lapply(train_data[cols_to_factors],factor)

###### Distribution of the class variable ###########
label_info <- table(train_data$label)
barplot(label_info,ylim=c(0,85000), xlab="Label", ylab="Frequency", col="black",
        main="Label Distribution")

####Distribution of features with class variable######
categorical_cols <- names(train_data)[ sapply(train_data, is.factor)]
categorical_cols <- categorical_cols[-which(categorical_cols %in%  c("label"))]
numeric_cols     <- names(train_data)[ sapply(train_data, is.numeric)]

for(col_name in categorical_cols)
{
  counts <- table(train_data$label,train_data[[col_name]])
  barplot(counts,xlab=col_name,col=c("red", "lightblue"),
          beside=TRUE,
          ylab ="frequency", main = col_name)
  legend("topright",c("dos","normal","probes","rtl","utr"),fill = c("red","lightblue"))
}

for (col_name in numeric_cols)
{
  boxplot(train_data[[col_name]]~train_data$label, col = "lightblue", xlab=col_name, 
          ylab="frequency", main = col_name)
}

#### Columns to be Removed######
train_data <- train_data [ ,-which(names(train_data) %in% c('is_host_login','num_outbound_cmds','service'))]

#### Correlation matrix #####
numeric_cols      <- names(train_data)[sapply(train_data, is.numeric)]
correlationMatrix <- cor(train_data[,numeric_cols])
relevant_correlations <- as.matrix(subset(melt(correlationMatrix, na.rm = TRUE), abs(value) > 0.8 & abs(value) <1))



######Some preprocessing on test dataset.###########
test_data$label = mapply(get_attack_type_label,test_data$label)
cols_to_factors             <- c('protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login','label')
test_data[cols_to_factors] <- lapply(test_data[cols_to_factors],factor)
test_data <- test_data [ ,-which(names(test_data) %in% c('is_host_login','num_outbound_cmds','service'))]
X_test                       <- test_data[-which(names(test_data) %in% c('label'))]
y_test                       <- test_data$label


#TODO : Handle confusion matrix when others is missing in training data.
#####Building Naive-bayes Model#########
nb_model    <- naive_bayes(train_data$label ~ ., data=train_data)
nb_pred     <- predict(nb_model,X_test)
nb_cnmatrix <- confusionMatrix(table(nb_pred,y_test))

####Building logistic-regression Model #############
lr_model    <- glm(train_data$label ~. ,data=train_data,family = binomial(logit))
lr_pred     <- predict.glm(lr_model,X_test,type = 'response')
lr_cnmatrix <- table(y_test,lr_pred > 0.5)

####Building Random Forest Model#######
rf_model    <- randomForest(train_data$label ~ ., data=train_data,importance=TRUE,ntree=100)
rf_pred     <- predict(rf_model,X_test)
rf_cnmatrix <- confusionMatrix(table(rf_pred,y_test))

###Building SVM #####
svm_model    <- svm(train_data$label ~ ., data=train_data)
svm_pred     <- predict(svm_model,test_data)
svm_cnmatrix <- confusionMatrix(table(svm_pred,y_test))

####Building Decision Tree #####
dtree_model<-rpart(train_data$label ~ ., data=train_data,cp=0.0009)
rpart.plot(dtree_model,tweak=1.5)
dtree_pred <- predict(dtree_model,X_test, type = "class")
dtree_cnmatrix <- confusionMatrix(table(dtree_pred,y_test))


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


#One-class classification for one of the attack types, e.g. utr
train_data_utr <- train_data 
train_data_utr$new_label[train_data_utr$label %in% c('dos','rtl','probes','normal')] <- 'not_utr' 
train_data_utr$new_label[train_data_utr$label %in% c('utr')] <- 'utr'
train_data_utr <- train_data_utr[,-which(names(train_data_utr) %in% c('label'))]
train_data_utr$new_label <- as.factor(train_data_utr$new_label)
smote_var_list <- get_smote_variables(train_data_utr,'new_label')


######Modify test data########### 
test_data_utr <- test_data 
test_data_utr$new_label[test_data_utr$label %in% c('dos',
                                       'rtl','probes','normal','others')] <- 'not_utr'
test_data_utr$new_label[test_data_utr$label %in% c('utr')] <- 'utr'
test_data_utr <- test_data_utr[,-which(names(test_data_utr) %in% c('label'))]
test_data_utr$new_label <- as.factor(test_data_utr$new_label)
X_test_utr                       <- test_data_utr[-which(names(test_data_utr) %in% c('new_label'))]
y_test_utr                       <- test_data_utr$new_label



#####Building Naive-bayes Model#########
mod_nb_model    <- naive_bayes(train_data_utr$new_label ~ ., data=train_data_utr)
mod_nb_pred     <- predict(mod_nb_model,X_test_utr)
mod_nb_cnmatrix <- confusionMatrix(table(mod_nb_pred,y_test_utr))


train_data_utr_smote <- SMOTE(train_data_utr$new_label ~ ., train_data_utr, 
                              perc.over = smote_var_list[1],perc.under=smote_var_list[2])

#####Building Naive-bayes smote_model#########
smote_mod_nb_model    <- naive_bayes(train_data_utr_smote$new_label ~ ., 
                                     data=train_data_utr_smote)
smote_mod_nb_pred     <- predict(smote_mod_nb_model,X_test_utr)
smote_mod_nb_cnmatrix <- confusionMatrix(table(smote_mod_nb_pred,y_test_utr))
