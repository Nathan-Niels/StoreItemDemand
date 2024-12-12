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
train_data <- vroom("/kaggle/input/demand-forecasting-kernels-only/train.csv")
test_data <- vroom("/kaggle/input/demand-forecasting-kernels-only/test.csv")
sample_submission <- vroom("/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv")

nStores <- max(train_data$store)
nItems <- max(train_data$item)
for (s in 1:nStores){
  for (i in 1:nItems){
    storeItemTrain <- train_data %>%
      filter(store == s, item == i)
    storeItemTest <- test_data %>%
      filter(store == s, item == i)
    
    
    # Create recipe for single store/product combo
    SID_recipe <- recipe(sales ~ ., data = storeItemTrain) %>% 
      step_date(date, features = "decimal") %>% 
      step_date(date, features = "doy") %>% 
      step_date(date, features = "month") %>%
      step_date(date, features = "year") %>% 
      step_date(date, features = "dow") %>% 
      step_mutate_at(date_dow, fn = factor) %>% 
      step_mutate_at(date_month, fn = factor) %>% 
      step_range(date_doy, min = 0, max = pi) %>% 
      step_mutate(sinDOY = sin(date_doy),
                  cosDOY = cos(date_doy)) %>% 
      step_rm(store, item, date) %>% 
      step_lencode_mixed(all_nominal_predictors(),
                         outcome = vars(sales)) %>% 
      step_normalize(all_numeric_predictors())
    prepped_SID_recipe <- prep(SID_recipe)
    bake(prepped_SID_recipe, new_data = storeItemTrain)
    # 
    
    # Random Forest WF & CV
    rf_model <- rand_forest(mtry = 4,
                            min_n = 30,
                            trees = 500) %>% 
      set_engine("ranger") %>% 
      set_mode("regression")
    
    # Create workflow with model & recipe
    rf_wf <- workflow() %>% 
      add_recipe(SID_recipe) %>% 
      add_model(rf_model) %>%
      fit(data=storeItemTrain)
    
    # Generate Predictions
    rf_preds <- predict(rf_wf, new_data = storeItemTest) %>%
      bind_cols(storeItemTest) %>%
      select(id, .pred) %>%
      rename(sales=.pred)
    
    if (s == 1 & i == 1){
      all_preds <- rf_preds
    } else {
      all_preds <- bind_rows(all_preds, rf_preds)
    }
  }
}

# Format output to match sample submission
vroom_write(all_preds, file = "/kaggle/working/submission.csv", delim = ",")