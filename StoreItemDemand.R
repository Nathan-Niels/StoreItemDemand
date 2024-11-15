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
