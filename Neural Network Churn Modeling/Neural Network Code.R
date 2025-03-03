# Modeling Churn with a Neural Network
# Author: Riley Fencl
# Date: 2/20/2025

# Loading Packages
# I use a general package script that cover a wide variety of functions.
library(ggplot2)
library(dplyr)
library(ggpubr)
library(broom)
library(car)
library(Hmisc)
library(tidyr)
library(scales)
library(class)
library(stringr)
library(forecast)
library(Metrics)
library(tidymodels)
library(readxl)
library(readr)
library(treemapify)
library(lubridate)
library(caret)
library(smotefamily)
library(nnet)

# Loading in the data
df <- read.csv('churndata.csv') 

# Glimpse() the data to get a general idea of what we're working with.
glimpse(df)

# First step is to check data integrity. Confirm NAs, outliers, etc.
# na_sums is a handbuilt function that grabs all of the nas from the columns. 
na_sums(df)

# 0 missing values in all columns

# Checking distributions for outliers
ggplot(df, aes(x = CreditScore)) + geom_histogram(bins = 100, color = 'black', fill = 'cyan') +
  labs(title = 'Dristribution of Credit Score')

# Very large frequency count at right end of the distribution. May indicate some data input issues OR it might be that the
# distribution itself is capped. (Credit score systems have max values. I.E. FICO caps at 850.)

# Check Max
max(df$CreditScore)

# Call a frequency table to confirm
table(df$CreditScore)

# Calling the frequency table shows a large grouping at 850, this makes sense, due to the fact that 850 is most likely the cap
# of the score they collected.

# Continuing checking distributions
ggplot(df, aes(x = Balance)) + geom_histogram(bins = 100, color = 'black', fill = 'purple') +
  labs(title = 'Dristribution of Balance')

# Balance has a large frequency of 0s, this makes sense given the nature of bank balances.

# Continuing checking distributions
ggplot(df, aes(x = EstimatedSalary)) + geom_histogram(bins = 50, color = 'black', fill = 'salmon') +
  labs(title = 'Dristribution of Estimated Salary')

# Fairly uniform distribution no action necessary.

# Encoding categorical variables

# One-hot encoding Gender
cats1 <- model.matrix(~ Gender - 1, data = df)
# One-hot encoding Geography
cats2 <- model.matrix(~ Geography - 1, data = df)
# Combining both one-hot encoded variables
catsall <- cbind(cats1, cats2)
# Creating a new df, columning binding the cat variables, and removing the original cat variables.
df1 <- cbind(df,catsall)
df1 <- df1 %>%
  select(-Gender, -Geography)

# We don't need to look at HasCrCard or IsActiveMember, since these variables are already binary. We don't have to worry about
# them being overbalanced. 

# Scale Numeric Features
# Need to scale numeric columns to prevent large values from being higher weighted. Standardizes values around z-score so that
# every value is an a zscore of it's column. Allows all numerical columns to be equally weighted. 

df1$CreditScore <- scale(df$CreditScore)
df1$Age <- scale(df1$Age)
df1$Balance <- scale(df1$Balance)
df1$EstimatedSalary <- scale(df$EstimatedSalary)
df1$Tenure <- scale(df$Tenure)
df1$NumOfProducts <- scale(df$NumOfProducts)

# Exporting df1 for sharing
csv(df1, 'churndata1')

# With distributions checked, NAs handled, categorical variables one-hot encoded, and numeric variables scaled, we may now move
# on to splitting the data for train/test sets, firstly however we need to drop the non-necessary variables. 

# New Dataframe for this step
df2 <- df1 %>%
  select(-RowNumber, -CustomerId, -Surname)

# Lastly, before we split into test/training datasets we need to confirm the 'balance' of the target variable. I.E. do we 
# have 50/50 representation in the target variable 'exited' or is our data imbalanced 90/10, 80/20, 70/30, etc. 

table(df2$Exited)

# Because of the imbalance in our dataset, we will have to use a datapartition, normally we could sample, but doing so may 
# result in an even more imbalanced train/test set so we will use a data partition to preserve the current imbalance and we 
# will handle that imbalance later. 

# Set seed for reproduction 
set.seed(123)

# Splitting with createDataPartition()
train_i <- createDataPartition(df2$Exited, p = .8, list = FALSE)
train_df <- df2[train_i, ]
test_df <- df2[-train_i, ]

# SMOTE Balancing the train data. 
# Since we have an imbalance in our Exited variable, roughly 80%/20%, we need to balance the training set in someway to prevent
# the model from predicting the majority class. If we ran it as is, it is likely that the model itself would simply just learn
# to predict the majority class. SMOTE allows us to synthetically balance the training set by creating observations that will
# bring the Exited variable closer to a 50/50 distribution. This will allow the model to process the data more effectively, and
# it will have a much better performance when we apply the test data to the model which will be kept in-balanced since that is
# supposed to be representative the real-world imbalance. This method gives us the best of both worlds. We can build the model
# in a balanced environment so it learns the data more effectively and we can then test that model against the real-world 
# imbalance to see how it would perform in a deployment environment.

# SMOTE Balancing the Training Set
smote_df <- SMOTE(train_df[, !names(train_df) %in% 'Exited'], target = train_df$Exited, K = 5, dup_size = 2)

# We use K = 5 here since it is a fairly common value for synthetic sampling, and we don't fit any of the criteria for raising
# or lowering. 

# Dup_size = 2 means we are doubling the minority class.

# Convert Exited to factor for Classification
smote_df$data$class <- as.factor(smote_df$data$class)
test_df$Exited <- as.factor(test_df$Exited)

# Since this is fundamentally a classification problem, our target variable, exited, has to be converted to a factor for R.

# Grab test features/target
test_features <- test_df[, !names(test_df) %in% "Exited"]
test_target <- test_df$Exited

# To run our predictions on the model after the model is built we will need to have the features and the target variable 
# separated. We'll run the features in the predict() function after we build the model, and then we will use test_target
# to create our target matrix. 

# First model iteration
nnmodel <- nnet( 
  class ~ ., 
  data = smote_df$data,
  size = 10, 
  maxit = 200, 
  decay = .01, 
  linout = FALSE
)

# For this initial model, we ran size = 10, because this is a pretty reasonable starting point for model complexity. More
# neurons may lead to overfitting and less might not be complex enough to capture the patterns in the data. 
# Max iterations at 200 is fine, its just the initial starting point and we will change iterations as necessary as the model
# builds. 
# Decay at .01 is a fairly small decay value, it adds a small penalty to larger values and simplifies the model. We can raise
# this value as needed if we see signs of over fitting. 

# Create Predictions
preds <- predict(nnmodel, test_features, type = "class")

# Convert to factors
preds <- as.factor(preds)

# Create confusion matrix for evaluation
cmatrix <- table(Predicted = preds, Actual = test_target)

# Grab precision, recall and F1
confusionMatrix(preds, test_target, positive = '1')

# For our first iteration, our metrics weren't great but they weren't bad. 
# An accuracy of 82.4% is not bad but because we have an imbalanced dataset this can be misleading. 
# Our balanced accuracy was 76.77% which is still solid, but it could be better. 
# Sensitivity is not great as well, 67.25% means we are failing to identify around a third of the actual exits. 
# Specificity is 86.29%, pretty solid but to be expected somewhat due to the imbalance. 
# Positive Prediction Val is 55.31% this means when we predict an exit or 1, we are right 55.31% of the time and this is also
# not great. 
# F1 is moderate at .606, the formula is 2 * (precision * recall) / (precision + recall). 

# Tuning the parameters for better values. 

# Grid-tuning the model
# Grid tuning allows us to test multiple values and combinations and iterate automatically to better tune the model.
tgrid <- expand.grid(
  size = c(5, 10, 15, 20), 
  decay = c(.001, .005, .01, .02)
)

# Set control 
control <- trainControl(method = 'cv', number = 5)

# The control tests the model on the number of 'folds' that we specify. Folds are subsets of the data, that the model will 
# train itself on and then it will average the results across the folds to produce a more robust model. 

# Use tune-grid
set.seed(123)
t_model <- train(
  class ~ .,
  data = smote_df$data,
  method = 'nnet', 
  tuneGrid = tgrid,
  trControl = control,
  maxit = 500, 
  linout = FALSE,
  trace = FALSE
)

# Pulling Optimal Metrics
optmodel <- nnet (
  class ~ .,
  data = smote_df$data,
  size = t_model$bestTune$size,
  decay = t_model$bestTune$decay,
  maxit = 500,
  linout = FALSE
)

# Create Predictions
preds <- predict(optmodel, test_features, type = "class")

# Convert to factors
preds <- as.factor(preds)

# Create confusion matrix for evaluation
cmatrix <- table(Predicted = preds, Actual = test_target)

# Grab precision, recall and F1
confusionMatrix(preds, test_target, positive = '1')

# Unfortunately, the tuning-grid seemed to actually produce worse results across the board, so rather than automating the 
# process I elected to hand-tune the model, because the grid producing worse results indicates that somewhere near the default
# values we were already optimal. I suspect that this is due ot the fact that the tuning grid in it's values may have favored
# too heavily on the side of complexity due to the iterations on hidden neurons (size) in particular. 


# Final Model iteration
set.seed(123)
nnmodel <- nnet( 
  class ~ ., 
  data = smote_df$data,
  size = 5, 
  maxit = 1000, 
  decay = .07, 
  linout = FALSE
)

# Create Predictions
preds <- predict(nnmodel, test_features, type = "class")

# Convert to factors
preds <- as.factor(preds)

# Create confusion matrix for evaluation
cmatrix <- table(Predicted = preds, Actual = test_target)

# Grab precision, recall and F1
confusionMatrix(preds, test_target, positive = '1')

# After hand tuning the model quite a bit these were the parameters that produced the best output by far. While I do think this
# model is performing well it most certainly needs so upgrades and some work before deployment. Currently it is still missing 
# about 30% of churners which is by far the biggest issue. The model cannot be deployed in a state where 30% of churners are 
# not identified. That's 30% of exits going without any sort of churn-prevention actions taken to keep the customer. We also 
# have a fairly high false-positive rate which still needs to be reduced. Ultimately, before the model is deployed we will 
# need to further iterate on it with more data and perhaps even a form of ensemble modeling that will help take care of the 
# current issues. Overall, for a first neural network model I would say that it shows quite a bit of potential. 
