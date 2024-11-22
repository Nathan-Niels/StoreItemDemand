library(tidymodels)
library(tidyverse)
library(vroom)
library(patchwork)
library(forecast)
library(timetk)
library(modeltime)

# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_submission <- vroom("./sample_submission.csv")


# Facebook Prophet
# Store-Item Combo 2
# Filter to store/item
storeItemTrain2 <- train_data %>% 
  filter(store == 5,
         item == 2)

storeItemTest2 <- test_data %>% 
  filter(store == 5,
         item == 2)

# Create the CV split for time series
cv_split2 <- time_series_split(storeItemTrain2,
                              assess = "3 months",
                              cumulative = TRUE)

# Define FBProphet model
FBP_model2 <- prophet_reg() %>% 
  set_engine(engine = "prophet") %>% 
  fit(sales ~ date, data = training(cv_split2))

# Calibrate (tune) the models
cv_results2 <- modeltime_calibrate(FBP_model2,
                                   new_data = testing(cv_split2))

# Visualize results
cv_plot2 <- cv_results2 %>% 
  modeltime_forecast(new_data = testing(cv_split2),
                     actual_data = training(cv_split2)) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

# Refit to whole dataset
fullfit2 <- cv_results2 %>% 
  modeltime_refit(data = storeItemTrain2)

# Predict for all the obseravtions in storeItemTest1
fullfit_plot2 <- fullfit2 %>% 
  modeltime_forecast(new_data = storeItemTest2,
                     actual_data = storeItemTrain2) %>% 
  plot_modeltime_forecast(.interactive = FALSE)


# Store-Item Combo 1
storeItemTrain1 <- train_data %>% 
  filter(store == 4,
         item == 12)

storeItemTest1 <- test_data %>% 
  filter(store == 4,
         item == 12)

# Create the CV split for time series
cv_split <- time_series_split(storeItemTrain1,
                              assess = "3 months",
                              cumulative = TRUE)


# Define Facebook Prophet model
FBP_model <- prophet_reg() %>% 
  set_engine("prophet") %>% 
  fit(sales ~ date, data = training(cv_split))

# Calibrate (tune) the models
cv_results <- modeltime_calibrate(FBP_model,
                                  new_data = testing(cv_split))

# Visualize results
cv_plot <- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = training(cv_split)) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

# Refit to whole dataset
fullfit <- cv_results %>% 
  modeltime_refit(data = storeItemTrain1)

# Predict for all the obseravtions in storeItemTest1
fullfit_plot <- fullfit %>% 
  modeltime_forecast(new_data = storeItemTest1,
                     actual_data = storeItemTrain1) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

plotly::subplot(cv_plot, cv_plot2,
                fullfit_plot, fullfit_plot2,
                nrows = 2)
