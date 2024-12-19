# Modeling Project for IT ticket outcomes
# Author: Riley Fencl
# Date: 12/10/24

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
library(randomForest)

# Reading the Data Set
tdata <- read.csv('techsupport.csv')

# Observing the data structure
glimpse(tdata)

# Creating bar charts of the frequency per issue of each category

# First step is to create a function that we can use to produce bar chart frequency
# For this function we create a vector of the columns names that we want to create bar charts for and pass that as 
# the first argument, followed by the dataset that we are using.
# I chose to create a function here since this is a frequently occuring task.
freqbar <- function(cols, dataset) {
  for (i in cols) {
    plot <- ggplot(dataset, aes_string(x = i)) +
      geom_bar(color = 'black', fill = 'cyan') +
      labs(title = paste('Frequency of', i),
           x = i, 
           y = 'Frequency') +
      # The theme adjustsments make the text more readable, they bold the text, turn the labels at an angle
      # and lower them so they appear beneath the chart.
      theme(axis.text.x = element_text(face = 'bold',angle = 45, hjust = 1))
    print(plot)
  }
}

# Creating Vector of categorical columns we want bar charts for
cat_cols <- c("Customer_Issue", "Tech_Response", "Issue_Category", "Issue_Status")

# Running the function, and grabbing the plots for the cat variables.
freqbar(cat_cols, tdata)

# Converting Resolution_Time to a numeric variable.
# Creating a new dataset iteration, and removing all of the non-numeric characters from the category. 
tdata2 <- tdata %>%
  mutate(resolution_num = as.numeric(gsub('[^0-9]', '', Resolution_Time))) %>%
  # Renaming the column
  mutate(Resolution_Time = resolution_num) %>%
  # Lastly, we remove resolution_num since it is no longer needed
  select(-resolution_num)
# Confirming output
glimpse(tdata2)

# Preparing the Classification Dataset

# Converting Issue_Status to Resolved or Unresolved
c_tdata <- tdata2 %>%
  mutate(Issue_Status = case_when(
    Issue_Status %in% c('Resolved', 'Resolved after follow-up') ~ 'Resolved', 
    Issue_Status %in% c('Escalated', 'Pending') ~ 'Not Resolved',
    TRUE ~ Issue_Status
  ))
# Confirming Output
table(c_tdata$Issue_Status)

# Convert Issue_Status to Binary Coding
c_tdata <- c_tdata %>%
  mutate(Issue_Status = ifelse(Issue_Status == 'Resolved', 1, 0))

# Confirming Output
table(c_tdata$Issue_Status)

# Converting Issue_Category to One-hot encoding
# Model.Matrix creates dummy variables for each unique category, the -1 removes the intercept column that is 
# auto-generated. 
cats1 <- model.matrix(~ Issue_Category -1, data = c_tdata)

# Creating a 2nd dataset iteration and binding the output from Model.Matrix
c_tdata2 <- cbind(c_tdata, cats1)

# Checking output
glimpse(c_tdata2)

# Converting Customer_Issue to One-hot encoding
cats2 <- model.matrix(~ Customer_Issue - 1, data = c_tdata)

# Binding columns
c_tdata2 <- cbind(c_tdata2, cats2)

# Checking Output
glimpse(c_tdata2)

# Converting Tech_Response to One-hot encoding
cats3 <- model.matrix(~ Tech_Response - 1, data = c_tdata)

# Binding columns
c_tdata2 <- cbind(c_tdata2,cats3)

# Checking Output
glimpse(c_tdata2)

# Removing Original columns and creating a third data iteration
c_tdata3 <- c_tdata2 %>%
  select(-Issue_Category, -Customer_Issue, -Tech_Response)

# Checking Output
glimpse(c_tdata3)


# Data is ready for classification model and the object should be saved as a file if sharing is necessary. 
# Outputing to csv, this is my own csv function. 
# csv() intakes a dataset and a name to create a csv file automatically in the R_Dataset_Exports file
# csv <- function(dataset, filename) {
# write.csv(dataset, paste("C:/Users/User_Name/Documents/R_Dataset_Exports/", filename, ".csv", sep =""))
# }
csv(c_tdata3, 'c_tdata3')

# Prepping the regression model data

# One-hot encoding Issue_Status
cats4 <- model.matrix(~ Issue_Status - 1, data = tdata2)

# # Cbinding cats1, 2, and 3 since they will be needed for the regression model
catsall <- cbind(cats1, cats2, cats3, cats4)

# Cbinding all one-hot encoded variables, removing their original variable columns, and creating a dataset iteration
r_tdata <- tdata2 %>%
  cbind(catsall) %>%
  select(-Customer_Issue, -Tech_Response, -Issue_Category, -Issue_Status)

# Check output
glimpse(r_tdata)

# Checking it against orignal data
glimpse(tdata2)

# Data is ready for regression model. Outputing the dataset to a csv file for sharing if necessary. 
csv(r_tdata, 'r_tdata')

# With data preparation complete, it is time to begin model building starting with the random_forest regression model.

# Removing the id column
r_tdata1 <- r_tdata[,-1]

# Set seed for reproducibility
set.seed(111)
# Randomly sample row nums for 70% of the data
trainrows <- sample(1:nrow(r_tdata1), .7 * nrow(r_tdata1))
# Filtering those row nums for train data
train_data <- r_tdata1[trainrows, ]
# Filtering out those rows for test data
test_data <- r_tdata1[-trainrows, ]

# With the Way our model names work, randomForest models will get hung up on names with spaces.
# With this code we change the spaces to a period so the model will run them properly
colnames(train_data) <- make.names(colnames(train_data))
colnames(test_data) <- make.names(colnames(test_data))

# Creating the model
rfmodel <- randomForest(
  Resolution_Time ~ ., # Setting the formula so that Resolution_Time is our response predictor against all other columns
  data = train_data, # 70% of the data is used to create the model itself
  ntree = 500, # 500 trees is often a standard starting point, provides stability/efficient processing
  mtry = floor(sqrt(ncol(train_data) - 1)), # Selecting the number of predictors at each split, -1 for the response
  importance = TRUE # We want to see the hierarchy of variable importance
)

# Creating Predictions
predictions <- predict(rfmodel, newdata = test_data)

# Calculating MAPE
mape <- mean(abs((test_data$Resolution_Time - predictions) / test_data$Resolution_Time)) * 100
print(paste("MAPE:", mape, "%"))
# Mape is 96.1% Which really bad. That means on average, our predictions are off by 96%. 

# After increase the trees in increments of 1000, up to 5000 it was clear the number of trees was not going to work.

# Checking Variable Importance Plot
varImpPlot(rfmodel)

# Checking outliers
# With the data being perfectly normally distributed, dimensionality reduction is the next step. 
boxplot(r_tdata1$Resolution_Time, main = "Resolution_Time")

# Dimensionality reduction followed here because it was clear that the predictors were all redundant versions of 
# each other. For example, if it was a passowrd issue, it would guarantee be an account issue and tech response of
# password reset. 

# Simplifying the data
r_tdatas1 <- r_tdata1 %>% 
  select(Resolution_Time, starts_with('Customer'))

# Rebuilding the test/train sets
set.seed(111)
trainrows <- sample(1:nrow(r_tdatas1), 0.7 * nrow(r_tdatas1))
train_data <- r_tdatas1[trainrows, ]
test_data <- r_tdatas1[-trainrows, ]

# Changing the names
colnames(train_data) <- make.names(colnames(train_data))
colnames(test_data) <- make.names(colnames(test_data))

# Rebuilding the model
rfmodels1 <- randomForest(
  Resolution_Time ~ ., 
  data = train_data, 
  ntree = 1000, 
  mtry = floor(sqrt(ncol(train_data) - 1)), 
  importance = TRUE
)

# Creating Predictions
predictions <- predict(rfmodels1, newdata = test_data)

# Calculating MAPE
mape <- mean(abs((test_data$Resolution_Time - predictions) / test_data$Resolution_Time)) * 100
print(paste("MAPE:", mape, "%"))

# Checking the Variable Importance Plot
varImpPlot(rfmodels1)

# Assessing the R^2 for all groups
summary(lm(Resolution_Time ~ ., data = r_tdatas1))


# Checking a Linear Model of the data
lmmodel <- lm(Resolution_Time ~ ., data = train_data)
lmpredictions <- predict(lmmodel, newdata = test_data)
mape <- mean(abs((test_data$Resolution_Time - lmpredictions) / test_data$Resolution_Time)) * 100
print(paste("Linear Regression MAPE:", mape, "%"))


# With essentially all avenues failing to produce any sort of significant or even slight increase in the prediction
# power of the model, or at the very least indicate some of the failure points of where the model is having issues,
# the last ditch effort was to create a basic linear model with the non-one-hot-encoded variables.

# Creating the test data set
tdata3 <- tdata2[-1]

# Set seed
set.seed(111)

# Test/Train
trainrows <- sample(1:nrow(tdata3), 0.7 * nrow(tdata3))  
train_data <- tdata3[trainrows, ]  
test_data <- tdata3[-trainrows, ]  

# Building the model
lmmodel <- lm(Resolution_Time ~ ., data = train_data)

# Summary of the Linear Regression Model
summary(lmmodel)

# Predict on the Test Set
lmpredictions <- predict(lm_model, newdata = test_data)

# Calculate MAPE
mape <- mean(abs((test_data$Resolution_Time - lmpredictions) / test_data$Resolution_Time)) * 100
print(paste("MAPE:", round(mape, 2), "%"))
# MAPE is 125.21%


# After the final round of attempting to correct the error rate of the model my conclusion is the following:
# Due to the nature of the data, it is not possible to create any sort of accurately predicting model. While it stands
# That perhaps a principle component analysis or CatBoost model may have a higher level of performance, with a MAPE
# Of the given magnitude of ~ 125%, it is not reasonable to assume that either of these models or any form of parameter
# tuning could bring the error down to a considerable or useful value. The reality is that the errors of the model are due
# to the predictors themselves. In essence, each one of the predictors within its own category, is a one to one of the
# predictors in another category. I.E. in Tech_Response Reset Password, it will guarantee the category of Customer_Issue
# Forgot Password, and Issue_Category Account. Not only does this indicate that the multicolinearity of these predictors
# Is perfectly 1 in many cases, it also demonstrates that the predictors themselves are non-unique and completely redundant
# Effectively meaning that we are trying to create a prediction model with a single categorical variable of something
# like Issue_Category. Not only are we in a simple regression, we are also in it without any sort of substantive
# continuous data to provide granularity and base to work from. Generally, categorical variables are meant to be
# complimentary to their continuous counterparts, but they are very very rarely, a substitute and in this case that 
# rule is clear. Given the failure of the model in regards to its random forest and linear regressions and given the nature
# of why those models failed, it is unreasonable to continue forward with attempting a classification model. If the 
# regressions model failed to find any sort of substantive link to the response variable, it is unreasonable to assume
# that a classification model would perform any more effectively. While it might seem as if though there is potential
# in a classification model using Response_Time as a predictor, the previous analysis has shown that it is completely
# unrelated to any of the categorical variables. And therefore adding it to the model is unlikely to have any sort of
# significant impact especially when given the magnitude of error from previous models. 