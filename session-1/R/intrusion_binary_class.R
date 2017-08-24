#########Loading the required libraries and installing the missing ones ###################
load.libraries <- c('ggplot2', 'randomForest', 'caret', 'rpart','plyr', 'gbm','rpart.plot',
                    'reshape2', 'naivebayes','corrplot','e1071','glmnet')
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

###### Check for Missing data ########
cols_with_missing_vals <- sapply(train_data, function(x) sum(is.na(x)/nrow(train_data))) 

#####Grouping all attacks together##############
train_data$label[!train_data$label %in% c('normal.')] <- 'attacks'

######Cleaning label data ###
train_data$label[train_data$label %in% c('normal.')] <- 'normal'

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
numeric_cols     <- names(train_data)[sapply(train_data, is.numeric)]

for(col_name in categorical_cols)
{
  counts <- table(train_data$label,train_data[[col_name]])
  barplot(counts,xlab=col_name,col=c("red", "lightblue"),
          beside=TRUE,
          ylab ="frequency", main = col_name)
  legend("topleft",c("Attack","Normal"),fill = c("red","lightblue"))
}

for (col_name in numeric_cols)
{
  # minm<-min(train_data[[col_name]])  
  # maxm<-max(train_data[[col_name]])
  boxplot(train_data[[col_name]]~train_data$label, col = "lightblue", xlab=col_name, 
          ylab="frequency", main = col_name)
}

#### Columns to be Removed######
train_data <- train_data [ ,-which(names(train_data) %in% c(
                  'is_host_login','num_outbound_cmds','service'))]

#### Correlation matrix #####
numeric_cols      <- names(train_data)[sapply(train_data, is.numeric)]
correlationMatrix <- cor(train_data[,numeric_cols])
relevant_correlations <- as.matrix(subset(melt(correlationMatrix, na.rm = TRUE), 
                                          abs(value) > 0.8 & abs(value) <1))

######Some preprocessing on test dataset.########### 
test_data$label[!test_data$label %in% c('normal.')] <- 'attacks'
test_data$label[test_data$label %in% c('normal.')] <- 'normal'
### Converting columns to factors #### 
cols_to_factors             <- c('protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login','label')
test_data[cols_to_factors] <- lapply(test_data[cols_to_factors],factor)
#### Columns to be Removed######
test_data <- test_data [ ,-which(names(test_data) %in% c(
  'is_host_login','num_outbound_cmds','service'))]
X_test                       <- test_data[-which(names(test_data) %in% c('label'))]
y_test                       <- test_data$label


#####Building Naive-bayes Model#########
nb_model    <- naive_bayes(train_data$label ~ ., data=train_data)
nb_predictions     <- predict(nb_model,X_test)
nb_cnmatrix <- confusionMatrix(table(nb_predictions,y_test))

####Building logistic-regression Model #############
lr_model    <- glm(train_data$label ~. ,data=train_data,family = binomial(logit))
lr_predictions     <- predict.glm(lr_model,X_test,type = 'response')
lr_cnmatrix <- table(y_test,lr_predictions > 0.5)
 

####Building Random Forest Model#######
rf_model    <- randomForest(train_data$label ~ ., data=train_data,importance=TRUE,ntree=100)
rf_predictions     <- predict(rf_model,X_test)
rf_cnmatrix <- confusionMatrix(table(rf_predictions,y_test))

###Building SVM #####
svm_model <- svm(train_data$label ~ ., data=train_data)
svm_pred     <- predict(svm_model,X_test)
svm_cnmatrix <- confusionMatrix(table(svm_pred,y_test))

####Building Decision Tree #####
dtree_model<-rpart(train_data$label ~ ., data=train_data)
rpart.plot(dtree_model)
dtree_pred <- predict(dtree_model,X_test, type = "class")
dtree_cnmatrix <- confusionMatrix(table(dtree_pred,y_test))



##Hyperparameter tuning with GridSearchCV to tune random forest
control <- trainControl(method="repeatedcv", number=3, repeats=3,
                        classProbs = TRUE,
                        summaryFunction  = twoClassSummary,search="grid")
grid <- expand.grid(mtry=c(6,8))
rf_model <- train(train_data$label ~. ,data=train_data,family = "binomial",
                  method="rf", metric="ROC",
                  trControl=control, tuneGrid=grid)
rf_predictions     <- predict(rf_model,X_test)
rf_cnmatrix <- confusionMatrix(table(rf_predictions,y_test))

##Balance class data
#1. Adjust class weights
control <- trainControl(method="repeatedcv", number=10, repeats=1)
mod1 <- train(training$label ~. ,data=training,
              method = "C5.0Cost",
              tuneGrid = expand.grid(model = "tree", winnow = FALSE,
                                     trials = c(1:10, (1:5)*10),
                                     cost = 1:10),
              trControl = control)