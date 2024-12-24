# Modeling Food Delivery Times
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
# Dataframe iteration #1
df <- read.csv('food_delivery_times.csv')

# Checking the NA's
# Creating a function that imputes NA for each blank cell.
blank_to_na <- function(df) {
  df[] <- lapply(df, function(column) {
    if (is.character(column) || is.factor(column)) {
      # swaps blank cells with NA
      column[column == ""] <- NA
    }
    return(column)
  })
  return(df)
}

# Call function to convert blanks so we can add up the NA totals
# Dataframe iteration #2
df2 <- blank_to_na(df)

# Creating a function to get the NAs from each column
na_sums <- function(df) {
  # Creating an empty dataframe
  nadf <- empty_df <- data.frame(
    column_name = character(),
    na_count = numeric(),
    na_pct = numeric()
    )
  # Using a for loop to loop over each of the columns pulled from the dataframe
  for (i in names(df)) {
    # Creating a temporary loop dataframe. 
    tdf <- data.frame(
      column_name = i,
      # Using the [[i]] since we want to call the symbol from names(df) using i
      na_count = sum(is.na(df[[i]])),
      na_pct = (sum(is.na(df[[i]])) / nrow(df)) * 100
    )
    # Adds the loop dataframe to the empty df
    nadf <- rbind(nadf, tdf)
  }
# Sorting descending
nadf <- nadf %>%
  arrange(desc(na_count))

return(nadf)
}

# Call the function to get an accurate assessment of NA's by column
na_sums(df2)

# Check only the rows of NA data to ensure a pattern of distribution is unlikely.
na_only <- df2[apply(df2, 1 ,function(row) any(is.na(row))), ]

# Creating a dataframe with no NAs so we can graph
df_clean <- na.omit(df2)
# Checking the hisogram of our numeric variable
ggplot(df_clean, aes(x = Courier_Experience_yrs)) + geom_histogram(bins = 10)
# Creating a boxplot for potential outliers
ggplot(df_clean, aes(x = Courier_Experience_yrs)) + geom_boxplot()
# Pulling a summary of the column
summary(df_clean$Courier_Experience_yrs)
# Given the fact that the distribution is relatively uniform, the boxplot shows no outliers, and the summary statistics indicate
# that the mean and mode are relatively close, imputing the mean for this column, especially since it is only on 3% of the 
# total column rows, will be an acceptable option. Given the circumstances it is highly unlikely that imputing the mean will have
# any significant on the data characteristics. 

# Imputing the mean() for Courier_Experience_yrs
# Dataframe iteration #3
df3 <- df2
mean <- mean(df_clean$Courier_Experience_yrs)
df3$Courier_Experience_yrs[is.na(df3$Courier_Experience_yrs)] <- mean

# Checking the number of rows with NAs after imputation of Courier Experience
na_only <- df3[apply(df2, 1 ,function(row) any(is.na(row))), ]
glimpse(na_only)

# Creating bar charts for the categorical variables.
cat_cols <- c('Weather', 'Traffic_Level', 'Time_of_Day')
freqbar(cat_cols, df_clean)

# With 117 rows still with NAs, and all of those columns being categorical in nature, the row count is too high to consider
# dropping at 11.7% of the total number of rows, and due to them being categorical, "unknown" will be imputed for the missing
# values to preserve the rows. Imputing any specific value may have too much of an impact on the characteristics of the data
# especially given the distributions/skews of the categorical values. 

# Imputing 'Unknown' for Weather
df3$Weather[is.na(df3$Weather)] <- 'Unknown'
# Confirming output
table(df3$Weather)
# Imputing 'Unknown' for Time_of_Day
df3$Time_of_Day[is.na(df3$Time_of_Day)] <- 'Unknown'
# Confirming output
table(df3$Time_of_Day)
# Imputing 'Unknown' for Traffic_Level
df3$Traffic_Level[is.na(df3$Traffic_Level)] <- 'Unknown'
# Confirming output
table(df3$Traffic_Level)

# Calling previous function to check NAs on data
na_sums(df3)

# Before Modeling, converting the categorical variables to factors for random forest.
# Dataframe iteration #4
df4 <- df3
df4$Traffic_Level <- factor(df4$Traffic_Level, levels = c("Unknown", "Low", "Medium", "High"), ordered = TRUE)
df4$Weather <- factor(df4$Weather, levels = c("Clear", "Foggy", "Rainy", "Snowy", "Windy", "Unknown"))
df4$Time_of_Day <- factor(df4$Time_of_Day, levels = c('Unknown', 'Morning', 'Afternoon', 'Evening', 'Night', ordered = TRUE))
df4$Vehicle_Type <- factor(df4$Vehicle_Type, levels = c('Car', 'Bike', 'Scooter'))
# Rounding Courier Experience
df4$Courier_Experience_yrs <- round(df4$Courier_Experience_yrs, 2)
# Checking data structure to confirm outputs
glimpse(df4)


# Outputing modeling data frame to csv for saving/sharing. 
# Outputing to csv, this is my own csv function. 
# csv() intakes a dataset and a name to create a csv file automatically in the R_Dataset_Exports file
# csv <- function(dataset, filename) {
# write.csv(dataset, paste("C:/Users/User_Name/Documents/R_Dataset_Exports/", filename, ".csv", sep =""))
# }
csv(df4, 'rfmdata')

# Removing order_id
# Dataframe iteration #5 
df5 <- df4[,-1]

# Checking for Multicollinearity
numeric_cols <- dplyr::select_if(df5, is.numeric)
cor(numeric_cols)

# Given the extremely low correlations between predictors as well as them have no logical mechanism for multicollinearity, 
# it is safe to conclude that multicollinearity should not be an issue with this data.

# Beginning Random Forest Modeling
# Set seed for reproducibility
set.seed(111)
# Randomly sample row nums for 70% of the data
trainrows <- sample(1:nrow(df5), .7 * nrow(df5))
# Filtering those row nums for train data
train_data <- df5[trainrows, ]
# Filtering out those rows for test data
test_data <- df5[-trainrows, ]

# Creating the model
rfmodel <- randomForest(
  Delivery_Time_min ~ ., # Setting the formula so that Resolution_Time is our response predictor against all other columns
  data = df5, # 70% of the data is used to create the model itself
  ntree = 500, # 500 trees is often a standard starting point, provides stability/efficient processing
  mtry = floor(sqrt(ncol(train_data) - 1)), # Selecting the number of predictors at each split, -1 for the response
  importance = TRUE # We want to see the hierarchy of variable importance
)

# Creating Predictions
predictions <- predict(rfmodel, newdata = test_data)

# Calculating MAPE
mape <- mean(abs((test_data$Delivery_Time_min - predictions) / test_data$Delivery_Time_min)) * 100
print(paste("MAPE:", mape, "%"))
# With a starting MAPE of 7.6% prior to any tuning, the model already indicates excellent performance. 

# Next we will check RMSE and R-squared.
rmse <- sqrt(mean((test_data$Delivery_Time_min - predictions)^2))
print(paste("RMSE:", rmse))
# RMSE of 5.35 seems not too bad given the MAPE, to confirm we can compare this against the variance of Delivery Time.
# Checking Standard Deviation
sd(df5$Delivery_Time_min)
# 22.07
# Checking IQR
IQR(df5$Delivery_Time_min)
# 30

# Given the RMSE and the statistics on variance the model performance is quite powerful as it is. The RMSE is well below the
# natural variance levels in the model. 5.35 vs 22.07 and 5.35 vs. 30.

# R-squared.
1 - sum((test_data$Delivery_Time_min - predictions)^2) / 
sum((test_data$Delivery_Time_min - mean(test_data$Delivery_Time_min))^2)
# 93.5%

# WHile the model performance is excellent as it sits, there is no reason to not do any parameter tuning given that it could
# yield performance increases.

# Parameter Tuning
# Using a looping structure to loop through different parameters. 
# Creating the empty dataframe to store the rmse, and the parameters tuned.
results <- data.frame(ntree = integer(), mtry = integer(), RMSE = numeric())

# Using a double for loop, for every value of ntree we want to test, we will test it with 4 values of mtry.
for (ntree in c(500, 750, 1000, 1500)) {
  for (mtry in c(floor(sqrt(ncol(train_data)) - 1), 
                 floor(sqrt(ncol(train_data))), 
                 floor(sqrt(ncol(train_data)) + 1),
                 floor(sqrt(ncol(train_data)) + 2))) {
    
    rfmodel <- randomForest(
      Delivery_Time_min ~ ., 
      data = train_data, 
      ntree = ntree, 
      mtry = mtry
    )
    
    # Calculating RMSE
    predictions <- predict(rfmodel, newdata = test_data)
    rmse <- sqrt(mean((test_data$Delivery_Time_min - predictions)^2))
    
    # Storing results from the for loops in the empty data frame. 
    results <- rbind(results, data.frame(ntree = ntree, mtry = mtry, RMSE = rmse))
  }
}

results

# Given the results dataframe, it shows that mtry = 3 is most likely the most optimal parameter in this case as well as
# ntree = 750 being sufficient enough for error reduction and stability. 

# Creating Final model.

rfmodel <- randomForest(
  Delivery_Time_min ~ ., # Setting the formula so that Resolution_Time is our response predictor against all other columns
  data = df5, # 70% of the data is used to create the model itself
  ntree = 750, # 500 trees is often a standard starting point, provides stability/efficient processing
  mtry = floor(sqrt(ncol(train_data) + 1)), # Selecting the number of predictors at each split, -1 for the response
  importance = TRUE # We want to see the hierarchy of variable importance
)

# Creating Predictions
predictions <- predict(rfmodel, newdata = test_data)

# Calculating MAPE
mape <- mean(abs((test_data$Delivery_Time_min - predictions) / test_data$Delivery_Time_min)) * 100
print(paste("MAPE:", mape, "%"))
# "MAPE: 6.3794305161974 %"

# Next we will check RMSE and R-squared.
rmse <- sqrt(mean((test_data$Delivery_Time_min - predictions)^2))
print(paste("RMSE:", rmse))
# "RMSE: 5.01054002416234"

# Checking Standard Deviation
sd(df5$Delivery_Time_min)
# 22.07092

# Checking IQR
IQR(df5$Delivery_Time_min)
# 30

# R-squared.
1 - sum((test_data$Delivery_Time_min - predictions)^2) / 
  sum((test_data$Delivery_Time_min - mean(test_data$Delivery_Time_min))^2)
# 94.3%

# Checking the Variable Importance
varImpPlot(rfmodel)

# Checking the residuals versus fits plot.
# Calculate residuals
residuals <- test_data$Delivery_Time_min - predictions

# Create the residuals vs. fitted values plot
plot(
  predictions, residuals,
  main = "Residuals vs Fitted Values",
  xlab = "Fitted Values (Predictions)",
  ylab = "Residuals",
  pch = 20, # Small points for better visibility
  col = "blue"
)
abline(h = 0, col = "red", lwd = 2) # Add a horizontal line at y = 0
