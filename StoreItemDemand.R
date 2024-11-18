library(tidymodels)
library(tidyverse)
library(vroom)
library(patchwork)
library(forecast)

# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_submission <- vroom("./sample_submission.csv")

# EDA Plots
storeItem1 <- train_data %>% 
  filter(store == 2,
         item == 34)

storeItem2 <- train_data %>% 
  filter(store == 8,
         item == 3)

plot1 <- storeItem1 %>% 
  ggplot(mapping = aes(x = date,
                       y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

plot2 <- storeItem1 %>% 
  pull(sales) %>% 
  ggAcf(., lag.max = 31)

plot3 <- storeItem1 %>% 
  pull(sales) %>% 
  ggAcf(., lag.max = 2*365)

plot4 <- storeItem2 %>% 
  ggplot(mapping = aes(x = date,
                       y = sales)) +
  geom_line() +
  geom_smooth(se = FALSE)

plot5 <- storeItem2 %>% 
  pull(sales) %>% 
  ggAcf(., lag.max = 31)

plot6 <- storeItem2 %>% 
  pull(sales) %>% 
  ggAcf(., lag.max = 2*365)

# Create patchwork of plots
(plot1 + plot2 + plot3) / (plot4 + plot5 + plot6)


# Create recipe for single store/product combo
SID_recipe <- recipe(sales ~ ., data = storeItem1) %>% 
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
prepped_SID_recipe <- prep(SID_recipe)
bake(prepped_SID_recipe, new_data = storeItem1)
  

# Random Forest WF & CV

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

# Create workflow with model & recipe
rf_wf <- workflow() %>% 
  add_recipe(SID_recipe) %>% 
  add_model(rf_model)

# Finalize parameters
final_par <- extract_parameter_set_dials(rf_model) %>% 
  finalize(storeItem1)

# Set tuning grid
rf_tune_grid <- grid_regular(final_par,
                            levels = 5)

# Set up K-fold CV
rf_folds <- vfold_cv(storeItem1, v = 5, repeats = 1)

rf_CV_results <- rf_wf %>% 
  tune_grid(resamples = rf_folds,
            grid = rf_tune_grid,
            metrics = metric_set(smape))

rf_CV_results %>% collect_metrics() %>%
  slice(which.min(mean))

rf_CV_results %>% show_best(metric = "smape")

# Find best tuning parameters
rf_best_tune <- rf_CV_results %>% 
  select_best(metric = "smape")
rf_best_tune

  