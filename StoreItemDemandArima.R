library(tidymodels)
library(tidyverse)
library(vroom)
library(patchwork)
library(forecast)
library(naivebayes)
library(discrim)
library(embed)
library(timetk)
library(modeltime)

# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_submission <- vroom("./sample_submission.csv")


# ARIMA
# Store-Item Combo 2
# Filter to store/item
storeItemTrain2 <- train_data %>% 
  filter(store == 5,
         item == 2)

storeItemTest2 <- test_data %>% 
  filter(store == 5,
         item == 2)


# Create the CV split for time series
cv_split <- time_series_split(storeItemTrain2,
                              assess = "3 months",
                              cumulative = TRUE)

# Create recipe
arima_recipe2 <- recipe(sales ~ ., data = storeItemTrain2) %>% 
  step_date(date, features = "decimal") %>% 
  step_date(date, features = "doy") %>% 
  step_date(date, features = "month") %>% 
  step_rm(store, item) %>% 
  step_date(date, features = "dow") %>% 
  step_mutate_at(date_dow, fn = factor) %>% 
  step_mutate_at(date_month, fn = factor) %>% 
  step_range(date_doy, min = 0, max = pi) %>% 
  step_mutate(sinDOY = sin(date_doy),
              cosDOY = cos(date_doy))
prepped_arima_recipe2 <- prep(arima_recipe2)
bake(prepped_arima_recipe2, new_data = storeItemTrain2)

# Define ARIMA model
arima_model2 <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5,
                         seasonal_ar = 2,
                         seasonal_ma = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>% 
  set_engine("auto_arima")

# Merge into single workflow and fit to training data
arima_wf2 <- workflow() %>% 
  add_recipe(arima_recipe2) %>% 
  add_model(arima_model2) %>% 
  fit(data = training(cv_split))

# Calibrate (tune) the models
cv_results2 <- modeltime_calibrate(arima_wf2,
                                  new_data = testing(cv_split))

# Visualize results
cv_plot2 <- cv_results2 %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = training(cv_split)) %>% 
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

view(storeItemTest1)

# Create the CV split for time series
cv_split <- time_series_split(storeItemTrain1,
                              assess = "3 months",
                              cumulative = TRUE)

# Create recipe
arima_recipe <- recipe(sales ~ ., data = storeItemTrain1) %>% 
  step_date(date, features = "decimal") %>% 
  step_date(date, features = "doy") %>% 
  step_date(date, features = "month") %>% 
  step_rm(store, item) %>% 
  step_date(date, features = "dow") %>% 
  step_mutate_at(date_dow, fn = factor) %>% 
  step_mutate_at(date_month, fn = factor) %>% 
  step_range(date_doy, min = 0, max = pi) %>% 
  step_mutate(sinDOY = sin(date_doy),
              cosDOY = cos(date_doy))
prepped_arima_recipe <- prep(arima_recipe)
bake(prepped_arima_recipe, new_data = storeItemTrain1)

# Define ARIMA model
arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5,
                         seasonal_ar = 2,
                         seasonal_ma = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>% 
  set_engine("auto_arima")

# Merge into single workflow and fit to training data
arima_wf <- workflow() %>% 
  add_recipe(arima_recipe) %>% 
  add_model(arima_model) %>% 
  fit(data = training(cv_split))

# Calibrate (tune) the models
cv_results <- modeltime_calibrate(arima_wf,
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
