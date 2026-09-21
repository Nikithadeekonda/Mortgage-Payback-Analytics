#---------------------------------------
#Case-3: "Mortgage Payback Analytics"
#---------------------------------------

# Load the dataset 
mortgage <- read.csv("Mortgage.csv")

# Quick overview
str(mortgage)
summary(mortgage)

# ---------------------------
# Install Necessary Packages 
# ---------------------------

# List of required packages
packages <- c(
  "ggplot2", "dplyr", "corrplot", "GGally", "patchwork",
  "fastDummies", "caret", "rpart", "pROC", "tidyr",
  "xgboost", "cluster"
)

to_install <- packages[!(packages %in% installed.packages()[,"Package"])]

if(length(to_install) > 0) install.packages(to_install)

# ----------------------------------------------------
# Libraries Required
# ----------------------------------------------------

library(ggplot2)      # For data visualization
library(dplyr)        # For data manipulation
library(corrplot)     # For correlation matrix visualization
library(GGally)       # For pair plots
library(patchwork)    # For combining multiple ggplot objects
library(fastDummies)  # For dummy variable creation
library(caret)        # For model training control and cross-validation
library(rpart)        # For Decision Tree (used by caret)
library(pROC)         # For ROC curve and AUC calculation
library(tidyr)        # For data tidying (e.g., drop_na)
library(xgboost)      # For XGBoost model
library(cluster)      # For K-means and Hierarchical clustering

# Data Visualizations
# Summary statistics
summary(mortgage)

# Basic numeric summary (short & clean)
num_summary <- summary(mortgage[, sapply(mortgage, is.numeric)])
print(num_summary)

# Distribution Analysis of Key Mortgage Variables
key_vars <- c("FICO_orig_time", "LTV_time", "balance_time", "interest_rate_time", "hpi_time")

# Create a list of plots
plots <- lapply(key_vars, function(v) {
  ggplot(mortgage, aes(x = !!sym(v))) +
    geom_histogram(fill = "steelblue", color = "white", bins = 30) +
    labs(title = paste("Distribution of", v),
         x = v, y = "Frequency") +
    theme_minimal(base_size = 13)
})

# Combine all plots into one grid
combined_plot <- wrap_plots(plots, ncol = 2)
combined_plot

# Outlier Detection Using Boxplots
# Variables to visualize
box_vars <- c("balance_time", "LTV_time", "interest_rate_time")

# Create a list of boxplots
plots_box <- lapply(box_vars, function(v) {
  ggplot(mortgage, aes(y = !!sym(v))) +
    geom_boxplot(fill = "lightgreen", color = "black") +
    labs(title = paste("Boxplot of", v), y = v) +
    theme_minimal(base_size = 13)
})

# Combine all boxplots in a single grid
combined_boxplots <- wrap_plots(plots_box, ncol = 2)
combined_boxplots

# Correlation heatmap 
numeric_vars <- mortgage %>% select(where(is.numeric))
corr_matrix <- cor(numeric_vars, use = "pairwise.complete.obs")
corrplot(corr_matrix, method = "color", type = "lower", tl.cex = 0.6, tl.col = "black",
         title = "Correlation Heatmap of Numeric Variables")

# Relationship plots 
# LTV vs FICO (credit vs leverage)
ggplot(mortgage, aes(x = FICO_orig_time, y = LTV_time)) +
  geom_point(alpha = 0.4, color = "darkblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Relationship Between FICO Score and LTV Ratio",
       x = "FICO Score at Origination", y = "Loan-to-Value Ratio") +
  theme_minimal()

# GDP vs Unemployment (macro trends)
ggplot(mortgage, aes(x = gdp_time, y = uer_time)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "GDP Growth vs Unemployment Rate",
       x = "GDP Growth (%)", y = "Unemployment Rate (%)") +
  theme_minimal()

#------------------------
#Check missing values
#------------------------
colSums(is.na(mortgage))

#-----------------------------------
# Count zero values in each column
#-----------------------------------
colSums(mortgage == 0, na.rm = TRUE)

#-------------------------------------------------------
# Keep only the last observation for each borrower (id)
#------------------------------------------------------
mortgage_last <- mortgage %>%
  group_by(id) %>%
  arrange(id, time) %>%        
  slice_tail(n = 1) %>%        
  ungroup()

# Check results again
nrow(mortgage_last)                 
length(unique(mortgage_last$id))    

# Quick preview
head(mortgage_last)

# ------------------------
# 5.Dimension Reduction
# ------------------------

# 1) Select key predictors from borrower-level data
preds <- c(
  "FICO_orig_time", "LTV_orig_time", "Interest_Rate_orig_time",
  "balance_time", "LTV_time",
  "gdp_time", "uer_time", "hpi_time", "interest_rate_time"
)

# Subset and keep only complete cases
tmp <- mortgage_last[, preds]
df  <- tmp[complete.cases(tmp), ]

# Ensure df is a base data.frame (not tibble)
df <- as.data.frame(df)

# -------------------------------------
# Multicollinearity Check using VIF
# -------------------------------------

vif_vals <- sapply(names(df), function(col) {
  others <- setdiff(names(df), col)
  form <- as.formula(paste(col, "~", paste(others, collapse = " + ")))
  r2 <- summary(lm(form, data = df))$r.squared
  1 / (1 - r2)
})

vif_vals_rounded <- round(vif_vals, 2)
vif_vals_rounded   # <- report min/max range in your writeup

#--------------------------------------
# 6. Data Engineering and Transformation
#--------------------------------------

df <- mortgage_last

# -----------------------------
# Missing Value Imputation
# -----------------------------

# Median of LTV_time excluding missing
med_LTV <- median(mortgage_last$LTV_time, na.rm = TRUE)

# Replace missing values
mortgage_last$LTV_time[is.na(mortgage_last$LTV_time)] <- med_LTV

# Check missing values in each column
colSums(is.na(mortgage_last))


#---------------------------------------------------------------------
# Zero-Value Imputation (Only for variables where zero is INVALID)
# --------------------------------------------------------------------

vars_zero <- c("balance_time",
               "LTV_time",
               "interest_rate_time",
               "balance_orig_time")   

for (v in vars_zero) {
  x <- df[[v]]
  med_nonzero <- median(x[x != 0], na.rm = TRUE)
  x[x == 0] <- med_nonzero
  df[[v]] <- x
}

# update dataset
mortgage_last <- df

vars_zero <- c("balance_time", "LTV_time",
               "interest_rate_time", "balance_orig_time")

# checking for zero values after imputation
sapply(mortgage_last[vars_zero], function(x) sum(x == 0))

#---------------------------------------
# Macroeconomic variables for PCA
#---------------------------------------
econ_vars <- c("gdp_time", "uer_time", "hpi_time", "interest_rate_time")

# Standardize macro variables from cleaned df
econ_data <- scale(df[, econ_vars])

# PCA
pca_result <- prcomp(econ_data, center = TRUE, scale. = TRUE)

# Cumulative variance explained
var_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)

# Choose number of PCs to retain (≥ 90% variance)
k <- which(var_explained >= 0.90)[1]

# Create PCA components
econ_pcs <- as.data.frame(pca_result$x[, 1:k])
names(econ_pcs) <- paste0("econ_PC", 1:k)

# Merge PCA components into df
df <- cbind(df, econ_pcs)

mortgage_last <- cbind(mortgage_last, econ_pcs)





#------------------------------------------------------
# 1. Keep last observation per borrower
#------------------------------------------------------
mortgage_last <- mortgage %>%
  group_by(id) %>%
  arrange(id, time) %>%
  slice_tail(n = 1) %>%
  ungroup()

#------------------------------------------------------
# 2. Missing value imputation (LTV_time)
#------------------------------------------------------
med_LTV <- median(mortgage_last$LTV_time, na.rm = TRUE)

mortgage_last$LTV_time[is.na(mortgage_last$LTV_time)] <- med_LTV

# Sanity check: no NAs now
colSums(is.na(mortgage_last))

#------------------------------------------------------
# 3. Zero-value imputation (where 0 is invalid)
#------------------------------------------------------
vars_zero <- c("balance_time", "LTV_time",
               "interest_rate_time", "balance_orig_time")

for (v in vars_zero) {
  x <- mortgage_last[[v]]
  med_nonzero <- median(x[x != 0], na.rm = TRUE)
  x[x == 0] <- med_nonzero
  mortgage_last[[v]] <- x
}

# Check zero values after imputation
sapply(mortgage_last[vars_zero], function(x) sum(x == 0, na.rm = TRUE))

#------------------------------------------------------
# 4. PCA on macro variables (using cleaned mortgage_last)
#------------------------------------------------------
econ_vars <- c("gdp_time", "uer_time", "hpi_time", "interest_rate_time")

econ_data <- scale(mortgage_last[, econ_vars])

pca_result   <- prcomp(econ_data, center = TRUE, scale. = TRUE)
var_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
k <- which(var_explained >= 0.90)[1]

econ_pcs <- as.data.frame(pca_result$x[, 1:k])
names(econ_pcs) <- paste0("econ_PC", 1:k)

# Attach PCA components
mortgage_last <- cbind(mortgage_last, econ_pcs)

# -----------------------------
# Feature engineering
# -----------------------------
df <- df %>%
  mutate(
    balance_change = balance_time - balance_orig_time,
    LTV_change     = LTV_time - LTV_orig_time,
    loan_age       = time - orig_time,
    default_flag   = as.integer(status_time == 1),
    payoff_flag    = as.integer(status_time == 2),
    active_flag    = as.integer(status_time == 0)
  )

# -----------------------------
# Normalization (scaling)
# -----------------------------
norm_vars <- c(
  "FICO_orig_time",
  "LTV_orig_time",
  "Interest_Rate_orig_time",
  "balance_time",
  "LTV_time",
  "interest_rate_time",
  "balance_orig_time",
  "balance_change",
  "LTV_change",
  "loan_age"
)

# Apply standardization (z-score normalization)
df[, norm_vars] <- scale(df[, norm_vars])

# final engineered + cleaned dataset
mortgage_final <- df

#----------------------------
# Data Partitioning
#----------------------------
set.seed(42)

# Use unique borrower IDs for partitioning (70% train, 30% test)
ids    <- unique(mortgage_final$id)
n_ids  <- length(ids)

tr_ids <- sample(ids, size = floor(0.7 * n_ids))

train <- mortgage_final[mortgage_final$id %in% tr_ids, ]
test  <- mortgage_final[!(mortgage_final$id %in% tr_ids), ]

# Create classification target for default vs no_default
train$y_default <- factor(
  ifelse(train$default_flag == 1, "default", "no_default")
)

test$y_default <- factor(
  ifelse(test$default_flag == 1, "default", "no_default")
)

# checking number of rows
nrow(train)
nrow(test)

#----------------------------------------------------------
# Model fitting, validation accuracy and test accuracy
#----------------------------------------------------------

# ----------------------------
# 1. Decision Tree
# ----------------------------

set.seed(42)

# Select predictor candidates (borrower, loan, engineered, PCA)
candidate_feats <- c(
  "FICO_orig_time", "LTV_orig_time", "Interest_Rate_orig_time",
  "balance_time", "balance_orig_time",
  "LTV_time", "interest_rate_time",
  "balance_change", "LTV_change", "loan_age",
  "econ_PC1", "econ_PC2", "econ_PC3",
  "REtype_CO_orig_time", "REtype_PU_orig_time",
  "REtype_SF_orig_time", "investor_orig_time"
)





candidate_feats <- c(
  "FICO_orig_time", "LTV_orig_time", "Interest_Rate_orig_time",
  "balance_time", "balance_orig_time",
  "LTV_time", "interest_rate_time",
  "balance_change", "LTV_change", "loan_age",
  "econ_PC1", "econ_PC2", "econ_PC3",
  "REtype_CO_orig_time", "REtype_PU_orig_time",
  "REtype_SF_orig_time", "investor_orig_time"
)

feats <- intersect(names(train), candidate_feats)

# Check missing values in the modeling columns
colSums(is.na(train[, c(feats, "y_default")]))

# Keep only those columns that actually exist in train
feats <- intersect(names(train), candidate_feats)

# Build modeling frames (train/test with same predictors)
model_df <- train %>%
  dplyr::select(all_of(feats), y_default) %>%
  mutate(
    balance_change = ifelse(is.na(balance_change), 0, balance_change),
    LTV_change     = ifelse(is.na(LTV_change),     0, LTV_change)
  ) %>%
  tidyr::drop_na()

test_df <- test %>%
  dplyr::select(all_of(feats), y_default) %>%
  mutate(
    balance_change = ifelse(is.na(balance_change), 0, balance_change),
    LTV_change     = ifelse(is.na(LTV_change),     0, LTV_change)
  ) %>%
  tidyr::drop_na()

cat("Rows kept in training after NA fix:", nrow(model_df), "\n")

# Model formula
form <- as.formula(
  paste("y_default ~", paste(setdiff(names(model_df), "y_default"), collapse = " + "))
)

# Cross-validation setup (5-fold, upsampling, fixed seeds)
set.seed(42)
folds <- createFolds(model_df$y_default, k = 5, returnTrain = TRUE)

dt_grid <- data.frame(cp = seq(1e-4, 1e-2, length.out = 10))

seed_list <- vector("list", length(folds) + 1)
for (i in seq_along(folds)) seed_list[[i]] <- rep(42L, nrow(dt_grid))
seed_list[[length(seed_list)]] <- 42L

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  index           = folds,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "up",          # handle class imbalance
  savePredictions = "final",
  seeds           = seed_list
)

# Train Decision Tree with caret::train + rpart
dt_fit <- train(
  form,
  data      = model_df,
  method    = "rpart",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = dt_grid
)

# Validation (CV) metrics
best_cp <- dt_fit$bestTune$cp
i <- which.min(abs(dt_fit$results$cp - best_cp))
val_row <- dt_fit$results[i, ]

cat(sprintf("Validation AUC: %.3f | Validation Accuracy: %.3f\n",
            val_row$ROC, val_row$Accuracy))

# Test set evaluation
pred_prob <- predict(dt_fit, newdata = test_df, type = "prob")[, "default"]
pred_cls  <- predict(dt_fit, newdata = test_df)

test_acc <- mean(pred_cls == test_df$y_default)

test_auc <- pROC::auc(
  response  = test_df$y_default,
  predictor = pred_prob,
  levels    = c("no_default", "default"),
  direction = "<"
)

cat(sprintf("Test AUC: %.3f | Test Accuracy: %.3f\n",
            as.numeric(test_auc), test_acc))

conf_mat <- caret::confusionMatrix(pred_cls, test_df$y_default, positive = "default")
print(conf_mat)

# ROC Curve
roc_obj <- pROC::roc(
  test_df$y_default, pred_prob,
  levels    = c("no_default", "default"),
  direction = "<"
)

plot(roc_obj,
     col  = "blue",
     lwd  = 2,
     main = paste("ROC Curve (AUC =", round(pROC::auc(roc_obj), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "red")

# Lift & Gain Chart
lift_df <- data.frame(
  actual = ifelse(test_df$y_default == "default", 1, 0),
  prob   = pred_prob
) %>%
  dplyr::mutate(decile = dplyr::ntile(-prob, 10)) %>%
  dplyr::group_by(decile) %>%
  dplyr::summarise(defaults = sum(actual), .groups = "drop") %>%
  dplyr::mutate(
    gain = 100 * cumsum(defaults) / sum(defaults),
    lift = gain / (decile * 10)
  )

cat(sprintf("Top10%% gain: %.1f%% | Lift: %.2fx\n",
            lift_df$gain[1], lift_df$lift[1]))

# Gain Chart
ggplot(lift_df, aes(decile, gain)) +
  geom_line(col = "violet") + geom_point(col = "violet") +
  geom_abline(slope = 10, intercept = 0, lty = 2, col = "black") +
  labs(
    title = "Cumulative Gain Chart - Decision Tree",
    x = "Decile",
    y = "% Defaults Captured"
  ) +
  theme_minimal()

# Lift Chart
ggplot(lift_df, aes(decile, lift)) +
  geom_line(col = "darkgreen") + geom_point(col = "darkgreen") +
  geom_hline(yintercept = 1, lty = 2, col = "black") +
  labs(
    title = "Lift Chart - Decision Tree",
    x = "Decile",
    y = "Lift vs Random"
  ) +
  theme_minimal()

# -----------------------------------------
# Logistic Regression 
# -----------------------------------------

set.seed(42)

# Predictors 
candidate_feats <- c(
  "FICO_orig_time", "LTV_orig_time", "Interest_Rate_orig_time",
  "balance_time", "balance_orig_time",
  "LTV_time", "interest_rate_time",
  "balance_change", "LTV_change", "loan_age",
  "econ_PC1", "econ_PC2", "econ_PC3",
  "REtype_CO_orig_time", "REtype_PU_orig_time",
  "REtype_SF_orig_time", "investor_orig_time"
)

# Keep only predictors that actually exist in train
feats <- intersect(names(train), candidate_feats)

# Modeling frames (same columns train/test)
train_df <- train %>%
  dplyr::select(all_of(feats), y_default) %>%
  mutate(
    balance_change = ifelse(is.na(balance_change), 0, balance_change),
    LTV_change     = ifelse(is.na(LTV_change),     0, LTV_change)
  ) %>%
  tidyr::drop_na()

test_df <- test %>%
  dplyr::select(all_of(feats), y_default) %>%
  mutate(
    balance_change = ifelse(is.na(balance_change), 0, balance_change),
    LTV_change     = ifelse(is.na(LTV_change),     0, LTV_change)
  ) %>%
  tidyr::drop_na()

cat("Rows kept in training after NA fix:", nrow(train_df), "\n")

# CV folds + seeds (with upsampling)
set.seed(42)
folds <- createFolds(train_df$y_default, k = 5, returnTrain = TRUE)

seed_list <- vector("list", length(folds) + 1)
for (i in seq_along(folds)) seed_list[[i]] <- 42L
seed_list[[length(seed_list)]] <- 42L

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  index           = folds,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "up",
  savePredictions = "final",
  seeds           = seed_list
)

# Fit Logistic Regression
form <- reformulate(feats, response = "y_default")

logit_fit <- train(
  form,
  data      = train_df,
  method    = "glm",
  family    = binomial(),
  metric    = "ROC",
  trControl = ctrl
)

# Validation metrics
val_auc <- max(logit_fit$results$ROC)

val_acc <- logit_fit$pred %>%
  group_by(Resample) %>%
  summarise(acc = mean(pred == obs), .groups = "drop") %>%
  summarise(val_acc = mean(acc)) %>%
  pull(val_acc)

cat(sprintf("Validation AUC: %.3f | Validation Accuracy: %.3f\n",
            val_auc, val_acc))

# Test metrics
pred_prob <- predict(logit_fit, newdata = test_df, type = "prob")[, "default"]
pred_cls  <- predict(logit_fit, newdata = test_df)

test_acc <- mean(pred_cls == test_df$y_default)

test_auc <- pROC::auc(
  response  = test_df$y_default,
  predictor = pred_prob,
  levels    = c("no_default", "default"),
  direction = "<"
)

cat(sprintf("Test AUC: %.3f | Test Accuracy: %.3f\n",
            as.numeric(test_auc), test_acc))

print(confusionMatrix(pred_cls, test_df$y_default, positive = "default"))

# ROC curve
roc_obj <- pROC::roc(
  test_df$y_default, pred_prob,
  levels    = c("no_default", "default"),
  direction = "<"
)

plot(roc_obj,
     col  = "blue",
     lwd  = 2,
     main = paste("ROC Curve (AUC =", round(pROC::auc(roc_obj), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "red")

# Lift & Gain
lift_df <- data.frame(
  actual = ifelse(test_df$y_default == "default", 1, 0),
  prob   = pred_prob
) %>%
  mutate(decile = ntile(-prob, 10)) %>%
  group_by(decile) %>%
  summarise(defaults = sum(actual), n = n(), .groups = "drop") %>%
  mutate(
    gain = 100 * cumsum(defaults) / sum(defaults),
    lift = gain / (decile * 10)
  )

cat(sprintf("Top10%% gain: %.1f%% | Lift: %.2fx\n",
            lift_df$gain[1], lift_df$lift[1]))

# Gain Chart
ggplot(lift_df, aes(decile, gain)) +
  geom_line(col = "blue") + geom_point(col = "blue") +
  geom_abline(slope = 10, intercept = 0, lty = 2, col = "red") +
  labs(
    title = "Cumulative Gain Chart - Logistic Regression",
    x = "Decile",
    y = "% Defaults Captured"
  ) +
  theme_minimal()

# Lift Chart
ggplot(lift_df, aes(decile, lift)) +
  geom_line(col = "darkgreen") + geom_point(col = "darkgreen") +
  geom_hline(yintercept = 1, lty = 2, col = "red") +
  labs(
    title = "Lift Chart - Logistic Regression",
    x = "Decile",
    y = "Lift vs Random"
  ) +
  theme_minimal()

#-----------------------------
# Multiple Linear Regression
# ----------------------------

# Target: early_close_periods (how many periods early the loan prepaid)
#    - Only borrowers who fully prepaid
#    - Create target: early_close_periods = mat_time - time

reg_df <- mortgage_final %>%
  filter(payoff_flag == 1) %>%                     # only prepaid borrowers
  mutate(
    early_close_periods = pmax(mat_time - time, 0) # periods prepaid early
  )

# Quick checks
cat("Total prepaid borrowers:", nrow(reg_df), "\n")
summary(reg_df$early_close_periods)

# Predictor list for prepayment modelling
reg_feats <- c(
  "FICO_orig_time", "LTV_orig_time", "Interest_Rate_orig_time",
  "balance_time", "balance_orig_time",
  "LTV_time", "interest_rate_time",
  "balance_change", "LTV_change", "loan_age",
  "econ_PC1", "econ_PC2", "econ_PC3"
)

# Final regression modelling frame (for description / structure)
reg_model_df <- reg_df %>%
  dplyr::select(early_close_periods, all_of(reg_feats)) %>%
  drop_na()

cat("Rows in full regression frame:", nrow(reg_model_df), "\n")
str(reg_model_df)

# 2. Train/Test split for regression
#    (based on existing borrower-level partition: train & test)
#    We restrict to prepaid loans inside each partition.

# Training regression data: prepaid loans in TRAIN partition
reg_train <- train %>%
  filter(payoff_flag == 1) %>%
  mutate(
    early_close_periods = pmax(mat_time - time, 0)
  ) %>%
  dplyr::select(early_close_periods,
                all_of(intersect(names(.), reg_feats))) %>%
  drop_na()

# Test regression data: prepaid loans in TEST partition
reg_test <- test %>%
  filter(payoff_flag == 1) %>%
  mutate(
    early_close_periods = pmax(mat_time - time, 0)
  ) %>%
  dplyr::select(early_close_periods,
                all_of(intersect(names(.), reg_feats))) %>%
  drop_na()

cat("Regression rows - train:", nrow(reg_train),
    "| test:", nrow(reg_test), "\n")

# 3. Fit Multiple Linear Regression model
mlr_fit <- lm(early_close_periods ~ ., data = reg_train)

# Model summary (coefficients, significance, etc.)
summary(mlr_fit)

# 4. Model Performance: RMSE and R² on Train and Test

## Train performance
train_pred <- predict(mlr_fit, newdata = reg_train)

train_rmse <- sqrt(mean((train_pred - reg_train$early_close_periods)^2))
train_r2   <- 1 - sum((reg_train$early_close_periods - train_pred)^2) /
  sum((reg_train$early_close_periods -
         mean(reg_train$early_close_periods))^2)

## Test performance
test_pred <- predict(mlr_fit, newdata = reg_test)

test_rmse <- sqrt(mean((test_pred - reg_test$early_close_periods)^2))
test_r2   <- 1 - sum((reg_test$early_close_periods - test_pred)^2) /
  sum((test_pred - mean(reg_test$early_close_periods))^2)

cat(sprintf("Train RMSE: %.2f | Train R²: %.3f\n", train_rmse, train_r2))
cat(sprintf("Test  RMSE: %.2f | Test  R²: %.3f\n", test_rmse, test_r2))

#-----------------------------------------------------------------------
# Multiple Linear Regression - Reduced predictor list (all significant)
#-----------------------------------------------------------------------

sig_feats <- c(
  "balance_time",
  "LTV_time",
  "interest_rate_time",
  "balance_orig_time",
  "FICO_orig_time",
  "Interest_Rate_orig_time",
  "loan_age",
  "LTV_orig_time"
)

# Fit reduced MLR
mlr_reduced <- lm(
  as.formula(paste("early_close_periods ~", paste(sig_feats, collapse = " + "))),
  data = reg_train
)

summary(mlr_reduced)

# Performance on Train
train_pred_red <- predict(mlr_reduced, newdata = reg_train)
train_rmse_red <- sqrt(mean((train_pred_red - reg_train$early_close_periods)^2))
train_r2_red   <- 1 - sum((reg_train$early_close_periods - train_pred_red)^2) /
  sum((reg_train$early_close_periods - mean(reg_train$early_close_periods))^2)

# Performance on Test
test_pred_red <- predict(mlr_reduced, newdata = reg_test)
test_rmse_red <- sqrt(mean((test_pred_red - reg_test$early_close_periods)^2))
test_r2_red   <- 1 - sum((reg_test$early_close_periods - test_pred_red)^2) /
  sum((reg_test$early_close_periods - mean(reg_test$early_close_periods))^2)

cat(sprintf("Reduced Model - Train RMSE: %.2f | Train R²: %.3f\n",
            train_rmse_red, train_r2_red))
cat(sprintf("Reduced Model - Test  RMSE: %.2f | Test  R²: %.3f\n",
            test_rmse_red, test_r2_red))

#-------------------------------------------------------------
# XGBoost Regression – Prepayment Timing (early_close_periods)
#-------------------------------------------------------------

reg_feats <- c(
  "FICO_orig_time", "LTV_orig_time", "Interest_Rate_orig_time",
  "balance_time", "balance_orig_time",
  "LTV_time", "interest_rate_time",
  "balance_change", "LTV_change", "loan_age",
  "econ_PC1", "econ_PC2", "econ_PC3"
)

reg_train <- train %>%
  filter(payoff_flag == 1) %>%
  mutate(
    early_close_periods = pmax(mat_time - time, 0)
  ) %>%
  dplyr::select(
    early_close_periods,
    all_of(intersect(names(.), reg_feats))  # keep only columns that exist
  ) %>%
  drop_na()

reg_test <- test %>%
  filter(payoff_flag == 1) %>%
  mutate(
    early_close_periods = pmax(mat_time - time, 0)
  ) %>%
  dplyr::select(
    early_close_periods,
    all_of(intersect(names(.), reg_feats))
  ) %>%
  drop_na()

cat("XGB regression rows - train:", nrow(reg_train),
    "| test:", nrow(reg_test), "\n")

# Prepare X matrices and y vectors for xgboost

x_vars <- setdiff(names(reg_train), "early_close_periods")  # all predictors

x_train <- as.matrix(reg_train[, x_vars])
y_train <- reg_train$early_close_periods

x_test  <- as.matrix(reg_test[, x_vars])
y_test  <- reg_test$early_close_periods

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

watchlist <- list(train = dtrain, test = dtest)

# Set XGBoost hyperparameters (regression)

params <- list(
  objective = "reg:squarederror",  # regression
  eval_metric = "rmse",
  eta = 0.05,           # learning rate
  max_depth = 4,        # tree depth
  subsample = 0.8,      # row subsample
  colsample_bytree = 0.8
)

set.seed(42)

xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = watchlist,
  early_stopping_rounds = 30,
  print_every_n = 20
)

# Predictions and performance metrics (RMSE, R²)

# Train predictions (using best iteration)
train_pred_xgb <- predict(xgb_fit, newdata = dtrain)
test_pred_xgb  <- predict(xgb_fit, newdata = dtest)

# RMSE
train_rmse_xgb <- sqrt(mean((train_pred_xgb - y_train)^2))
test_rmse_xgb  <- sqrt(mean((test_pred_xgb  - y_test)^2))

# R²
train_r2_xgb <- 1 - sum((y_train - train_pred_xgb)^2) /
  sum((y_train - mean(y_train))^2)

test_r2_xgb  <- 1 - sum((y_test - test_pred_xgb)^2) /
  sum((y_test - mean(y_test))^2)

cat(sprintf("XGBoost - Train RMSE: %.2f | Train R²: %.3f\n",
            train_rmse_xgb, train_r2_xgb))
cat(sprintf("XGBoost - Test  RMSE: %.2f | Test  R²: %.3f\n",
            test_rmse_xgb, test_r2_xgb))

# Feature importance 
importance_mat <- xgb.importance(model = xgb_fit)
print(importance_mat)

# If available:
xgb.plot.importance(importance_mat, top_n = 10)

# ----------------------------------------------------
# K-means clustering on ALL 50,000 borrowers
# (one row per borrower: mortgage_final)
# ----------------------------------------------------

library(dplyr)
library(cluster)

# 1) Select numeric risk features (already normalized in preprocessing)
risk_feats <- c(
  "FICO_orig_time",
  "LTV_orig_time",
  "Interest_Rate_orig_time",
  "balance_time",
  "LTV_time",
  "interest_rate_time",
  "loan_age"
)

# Keep only features that exist, just in case
risk_feats <- intersect(names(mortgage_final), risk_feats)

# Feature matrix for clustering
X <- mortgage_final %>%
  dplyr::select(all_of(risk_feats))

# Standardize again – harmless even if already z-scored
X_s <- scale(X)

# Elbow Method to choose k
set.seed(42)

k_vals <- 2:8
wss    <- numeric(length(k_vals))

for (i in seq_along(k_vals)) {
  km_tmp <- kmeans(X_s, centers = k_vals[i], nstart = 10, iter.max = 50)
  wss[i] <- km_tmp$tot.withinss
}

plot(k_vals, wss, type = "b",
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method - K-Means (50,000 Borrowers)")

# Fit final K-means (choosing k = 3 from elbow)
k <- 3   

set.seed(42)
km <- kmeans(X_s, centers = k, nstart = 25, iter.max = 100)

# Silhouette analysis on a subsample (for stability)

set.seed(42)
idx_sil <- sample.int(nrow(X_s), min(20000, nrow(X_s)))

sil_obj <- silhouette(
  km$cluster[idx_sil],
  dist(X_s[idx_sil, , drop = FALSE])
)

cat("Average silhouette width (sample):",
    round(mean(sil_obj[, "sil_width"]), 3), "\n")

plot(sil_obj,
     main = "Silhouette Plot - K-Means (Sample of Borrowers)",
     col  = rainbow(k),
     border = NA)

# Attach cluster labels back to main dataset

mortgage_final$kmeans_cluster <- km$cluster

# Quick cluster profiling
cluster_profile <- mortgage_final %>%
  group_by(kmeans_cluster) %>%
  summarise(
    across(all_of(risk_feats), mean, .names = "avg_{.col}"),
    .groups = "drop"
  )

cluster_profile

#---------------------------
# Hierarchical Clustering
#---------------------------

set.seed(42)

# 1) Select numeric risk features (same as K-means)

risk_feats <- c(
  "FICO_orig_time",
  "LTV_orig_time",
  "Interest_Rate_orig_time",
  "balance_time",
  "LTV_time",
  "interest_rate_time",
  "loan_age"
)

# Keep only columns that exist and drop any remaining NAs
X <- mortgage_final %>%
  dplyr::select(all_of(intersect(names(.), risk_feats))) %>%
  tidyr::drop_na()

# Standardize (z-score)
X_s <- scale(X)

# Hierarchical clustering on a SUBSAMPLE
#    (distance matrix is too big for all 50k)

sub_n   <- min(20000, nrow(X_s))      # up to 20,000 borrowers
idx_sub <- sample.int(nrow(X_s), sub_n)
X_sub   <- X_s[idx_sub, , drop = FALSE]

# Distance + Ward's method
d_sub  <- dist(X_sub, method = "euclidean")
hc_sub <- hclust(d_sub, method = "ward.D2")

# Dendrogram
plot(hc_sub,
     labels = FALSE,
     main   = "Dendrogram - Hierarchical Clustering (Sample)",
     xlab   = "Borrowers",
     sub    = "")

# Cut tree into k clusters (choose k = 3)
k <- 3
rect.hclust(hc_sub, k = k, border = "red")

cl_sub <- cutree(hc_sub, k = k)

# Silhouette on the hierarchical sample
sil_sub <- silhouette(cl_sub, d_sub)
cat("Hierarchical sample silhouette:",
    round(mean(sil_sub[, "sil_width"]), 3), "\n")

plot(sil_sub,
     main   = "Silhouette Plot - Hierarchical Clustering (Sample)",
     col    = rainbow(k),
     border = NA)

# Compute centroids in scaled space
#    and assign ALL 50,000 borrowers
#    to nearest hierarchical centroid
centers <- rowsum(X_sub, cl_sub) / as.vector(table(cl_sub))  # k x p matrix

# Squared distance from each point to each centroid
d_all <- sapply(1:k, function(j) {
  rowSums(
    (X_s - matrix(centers[j, ], nrow(X_s),
                  ncol(X_s), byrow = TRUE))^2
  )
})

# Cluster assignment = nearest centroid
hc_clusters_all <- max.col(-d_all)   # 1..k

# Attach hierarchical cluster labels back to main data
mortgage_final$hclust_cluster <- hc_clusters_all

# 5) Quick cluster profiling 
hc_profile <- mortgage_final %>%
  group_by(hclust_cluster) %>%
  summarise(
    across(all_of(risk_feats), mean, .names = "avg_{.col}"),
    .groups = "drop"
  )

hc_profile

