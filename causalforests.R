library(grf)
library(shapr)
library("caret") # used for various predictive models
library(rpart.plot) # used to plot decision tree

#setwd("G:/My Drive/Stern Teaching/Data mining in R/Labs - in class use-cases/Lab13 - HTE-CausalForest")
path <- rstudioapi::getActiveDocumentContext()$path
Encoding(path) <- "UTF-8"
setwd(dirname(path))
df <- read.csv('focal_user_df_females.csv')


# Make sure the data has no missing values. Here I limit the data first to just
# variables used in analysis and only drop observations with missing values in those variables
vars <- c("manipulation", "msg_sent_cnt_2", "black", "asian", "mideast", "white", "latin", "indian",
          "age_raw", "height_in_raw")
df <- df[vars]

df$height_in_raw[is.na(df$height_in_raw)] <- median(df$height_in_raw, na.rm = T)

df <- df[complete.cases(df),]

# replace height with median value

# Let's use training and test data
set.seed(123)
smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

df.train <- df[train_ind,]
df.test <- df[-train_ind,]

# Isolate the "treatment" as a matrix
manipulation <- as.matrix(df.train$manipulation)

# Isolate the outcome as a matrix
msg_sent_cnt_2 <- as.matrix(df.train$msg_sent_cnt_2)

# Use model.matrix to get our predictor matrix
# We might also consider adding interaction terms
X <- model.matrix(lm(msg_sent_cnt_2 ~ -1 + black + asian + mideast + white + latin + indian +
                     age_raw + height_in_raw, data = df.train))

# Estimate causal forest
cf <- causal_forest(X,msg_sent_cnt_2,manipulation, num.trees = 5000)

# Get predicted causal effects for each observation
effects <- predict(cf, estimate.variance = TRUE)

# And use test X's for prediction
X.test <- model.matrix(lm(msg_sent_cnt_2 ~ -1 + + black + asian + mideast + white + latin + indian +
                            age_raw + height_in_raw, data = df.test))
# And get effects
effects.test <- predict(cf, X.test, estimate.variance = TRUE)

mean(effects$predictions)


## Decision Tree Classification 


# Cross validation
cross_validation <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated three times
  repeats = 3)
# Hyperparamter tuning
# maxdepth =  the maximum depth of the tree that will be created or
# the length of the longest path from the tree root to a leaf.

Param_Grid <-  expand.grid(maxdepth = 2:5)

dtree_fit <- train(df.train[,3:10],
                   effects$predictions, 
                   method = "rpart2",
                   # split - criteria to split nodes
                   parms = list(split = "gini"),
                   tuneGrid = Param_Grid)

# check the accuracy for different models
dtree_fit


# print the final model
dtree_fit$finalModel

prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)

