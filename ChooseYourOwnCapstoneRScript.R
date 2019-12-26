#Install packages we may need
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

#Download the Adult Census Income file
census <- read.csv("https://raw.githubusercontent.com/meganlambert/HarvardX_Capstone2/master/adult.csv")

#Examine the first rows of the census dataset with headers
head(census)

#Examine the structure of the dataset
str(census)

#Remove rows with missing values
clean_census <- filter(census, 
                       !workclass == "?", 
                       !occupation == "?", 
                       !native.country == "?")
clean_census <- droplevels(clean_census)

# Examine the structure of the dataset to confirm missing values are removed.
str(clean_census)

#View the summary statistics of the dataset.
summary(clean_census)

#Remove education and fnlwgt variables.
clean_census <- clean_census %>% select(-c(education,fnlwgt))

#Splitting the dataset for validation and training.
set.seed(1,sample.kind = "Rounding")  #if using R3.5 or earlier set.seed(1)
test_index <- createDataPartition(clean_census$income, times = 1, p = 0.2, list = FALSE)
census_validation <- clean_census[test_index, ]
census_training <- clean_census[-test_index, ]

#splitting the census_training dataset again for testing
set.seed(10,sample.kind = "Rounding")  #if using R3.5 or earlier set.seed(10)
test_indexsplit <- createDataPartition(census_training$income, times = 1, p = 0.2, list = FALSE)
testing <- census_training[test_indexsplit, ]
training <- census_training[-test_indexsplit, ]

#Make a histogram of the distribution of age for each income value
training %>% ggplot(aes(age)) + geom_histogram(aes(fill=income),color='black',binwidth=1) + labs(title= "Age Distribution for each Income")

#Make a boxplot of the distribution of age by income.
boxplot(age ~ income, data=training, main="Age Distribution by Income")

#View the summary statistics for the education.num variable
summary(training$education.num)

#Make a histogram of the distribution of education.num for each income value.
training %>% ggplot(aes(education.num)) + geom_histogram(aes(fill=income),color='black',binwidth=1) + labs(title= "Education Distribution for each Income")

#Make a bar graph to show the proportion of income levels by marital status
training %>% ggplot(aes(marital.status, fill = income)) + geom_bar(position = "fill") + labs(y = "proportion", title=" Proportion of Income Levels by Marital Status")  + theme(axis.text.x = element_text(angle=90))

#Make a bar graph showing the proportion of income levels by occupation
training %>% ggplot(aes(occupation, fill = income)) + geom_bar(position = "fill") + labs(y = "proportion", title=" Proportion of Income Levels by Occupation")  + theme(axis.text.x = element_text(angle=90))

#Plot the distribution of income levels by gender
qplot(income, data = training, fill = sex) + facet_grid (. ~ sex)

#Use the nearZeroVar function to identify variables with zero or low variance
nearZeroVar(training, saveMetrics=TRUE)

#View summary statistics for individuals with less than or equal to $50k income
summary (training[ training$income == "<=50K", 
                   c("capital.gain", "capital.loss")])

#View summary statistics for individuals with greater than $50k income
summary (training[ training$income == ">50K", 
                   c("capital.gain", "capital.loss")])

#Make a bar graph to show the proportion of income levels by native.country
training %>% ggplot(aes(native.country, fill = income)) + geom_bar(position = "fill") + labs(y = "proportion", title=" Proportion of Income Levels by Native Country")  + theme(axis.text.x = element_text(angle=90))

#Plot the distribution of income levels by race
qplot (income, data = training, fill = race) + facet_grid (. ~ race) + theme(axis.text.x = element_text(angle=90))

#Train a knn model on our training dataset optimizing k as the tuning parameter.
set.seed(9,sample.kind = "Rounding")  #if using R3.5 or earlier set.seed(9)
#Using a 10 fold cross-validation method to make our code run faster.
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(income ~ ., 
                   method = "knn", 
                   data = training, 
                   tuneGrid = data.frame(k = seq(5,33,2)), 
                   trControl = control)

#Highlighting the optimized k value on this plot.
ggplot(train_knn,highlight = TRUE)
#Use this code to see that the best k value is 15
train_knn$bestTune
#Compute the accuracy of the knn model on the testing dataset
knn_accuracy <- confusionMatrix(predict(train_knn, testing, type = "raw"), 
                                testing$income)$overall["Accuracy"]

#Create a table to save our results for each model
accuracy_results <- tibble(method = "knn", Accuracy = knn_accuracy)
#View the knn accuracy results in our table
accuracy_results %>% knitr::kable()

#Train a gbm model on our training dataset using 10-fold cross-validation
set.seed(2000,sample.kind = "Rounding") #if using R3.5 or earlier set.seed(2000)
#Using a 10 fold cross-validation method to make our code run faster
trCtrl <- trainControl (method = "cv", number = 10)
#Train a gbm model
train_gbm <- train (income~ ., trControl = trCtrl, 
                    method = "gbm", preProc="zv", data = training, verbose = FALSE)

#Compute the accuracy of our gbm model on the testing dataset
gbm_accuracy <- confusionMatrix(predict(train_gbm, testing, type = "raw"), 
                                testing$income)$overall["Accuracy"]

#Save the gbm accuracy results to our table
accuracy_results <- bind_rows(accuracy_results, tibble(method="gbm", Accuracy = gbm_accuracy))
#View the gbm accuracy results in our table
accuracy_results %>% knitr::kable()

#Train a Classification Tree model using rpart and optimizing for the complexity parameter
set.seed(300,sample.kind = "Rounding")  #if using R3.5 or earlier set.seed(300)
train_rpart <- train(income ~ ., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.01, len=100)), data = training)

#Highlight the optimized complexity parameter
ggplot(train_rpart, highlight=TRUE)
#Use this code to see the best cp value
train_rpart$bestTune

#Compute the accuracy of our Classification Tree model on the testing dataset
rpart_accuracy <- confusionMatrix(predict(train_rpart, testing), testing$income)$overall["Accuracy"]

#Save the Classification Tree model accuracy results to our table.
accuracy_results <- bind_rows(accuracy_results, tibble(method="rpart", Accuracy = rpart_accuracy))
#View the rpart accuracy results in our table
accuracy_results %>% knitr::kable()

#View the text version of our classification tree
plot(train_rpart$finalModel, margin = 0.1)  
text(train_rpart$finalModel, cex = 0.7)

#Train a random forest model on our training dataset
set.seed(3,sample.kind = "Rounding")  #if using R3.5 or earlier set.seed(3)
train_rf <- randomForest(income ~ ., data = training)
#Compute the accuracy of our random forest model on the testing dataset
rf_accuracy <- confusionMatrix(predict(train_rf, testing), testing$income)$overall["Accuracy"]

#Save the random forest accuracy results to our table
accuracy_results <- bind_rows(accuracy_results, tibble(method="random forest", Accuracy = rf_accuracy))
#View the random forest accuracy results in our table
accuracy_results %>% knitr::kable()

#Train our final random forest model on our census_training dataset
set.seed(3, sample.kind = "Rounding") #if using R3.5 or earlier set.seed(3)
final_train_rf <- randomForest(income ~ ., data = census_training)

#Compute the accuracy of our final random forest model on the validation set
final_accuracy <- confusionMatrix(predict(final_train_rf, census_validation), census_validation$income)$overall["Accuracy"]

#Save the random forest accuracy results to our table.
accuracy_results <- bind_rows(accuracy_results, tibble(method="Final Random Forest Model", Accuracy = final_accuracy))
#View the final random forest model accuracy results in our table
accuracy_results %>% knitr::kable()

#Results
accuracy_results %>% knitr::kable()

