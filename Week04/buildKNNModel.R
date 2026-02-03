# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(e1071)

# Read the data
df <- read.csv('segments_fs_width_50pct_overlap_DTAB_GT.csv')

# Convert seizure type to factor
df$seizure <- as.factor(df$seizure)
# Convert the levels of the factor to valid R variable names
levels(df$seizure) <- make.names(levels(df$seizure))

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(df$seizure, p = .8, 
                                  list = FALSE, 
                                  times = 1)
X_train <- df[trainIndex, -ncol(df)]
X_test <- df[-trainIndex, -ncol(df)]
y_train <- df[trainIndex, ncol(df)]
y_test <- df[-trainIndex, ncol(df)]

# Initialize KNN classifier
knn <- train(seizure ~ ., data = df, method = "knn", trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary), metric = "ROC")

# Perform 5-fold cross-validation
set.seed(42)
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Cross-validation predictions
knn_cv <- train(seizure ~ ., data = df, method = "knn", trControl = train_control, metric = "ROC")

# Initialize lists to store metrics for each fold
sensitivity_list <- c()
specificity_list <- c()
precision_list <- c()
auc_list <- c()

# Compute metrics for each fold
folds <- createFolds(df$seizure, k = 5, list = TRUE, returnTrain = TRUE)
for (i in 1:5) {
  train_indices <- folds[[i]]
  test_indices <- setdiff(1:nrow(df), train_indices)
  
  X_train <- df[train_indices, -ncol(df)]
  X_test <- df[test_indices, -ncol(df)]
  y_train <- df[train_indices, ncol(df)]
  y_test <- df[test_indices, ncol(df)]
  
  knn_model <- train(X_train, y_train, method = "knn")
  y_pred_fold <- predict(knn_model, X_test)
  y_prob_fold <- predict(knn_model, X_test, type = "prob")[,2]
  
  cm <- confusionMatrix(y_pred_fold, y_test)
  sensitivity <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  precision <- cm$byClass["Precision"]
  roc_auc <- roc(y_test, y_prob_fold)$auc
  
  sensitivity_list <- c(sensitivity_list, sensitivity)
  specificity_list <- c(specificity_list, specificity)
  precision_list <- c(precision_list, precision)
  auc_list <- c(auc_list, roc_auc)
}

# Create a DataFrame with the results
results_df <- data.frame(
  Sensitivity = sensitivity_list,
  Specificity = specificity_list,
  Precision = precision_list,
  AUC = auc_list
)

print("Cross-validation results:")
print(results_df)
print(paste("Mean Sensitivity:", mean(results_df$Sensitivity)))
print(paste("Mean Specificity:", mean(results_df$Specificity)))
print(paste("Mean Precision:", mean(results_df$Precision)))
print(paste("Mean AUC:", mean(results_df$AUC)))

# Fit the model on the entire dataset
knn_final <- train(seizure ~ ., data = df, method = "knn")

# Predict probabilities for the positive class
y_prob <- predict(knn_final, df, type = "prob")[,2]

# Compute ROC curve and AUC
roc_obj <- roc(df$seizure, y_prob)
roc_auc <- auc(roc_obj)

# Create a DataFrame with TPR, FPR, and thresholds
roc_results_df <- data.frame(
  TPR = roc_obj$sensitivities,
  FPR = 1-roc_obj$specificities,
  Threshold = roc_obj$thresholds
)

# Plot the ROC curve
ggplot(roc_results_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "darkorange", size = 1) +
  geom_abline(linetype = "dashed", color = "navy") +
  xlim(0, 1) +
  ylim(0, 1) +
  labs(title = paste("Receiver Operating Characteristic (AUC =", round(roc_auc, 2), ")"),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal()
