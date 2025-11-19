library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

## Read in all data

trainData <- vroom('train.csv') %>%
  mutate(Store = factor(Store), Dept = factor(Dept))
testData <- vroom('test.csv') %>%
  mutate(Store = factor(Store), Dept = factor(Dept))
stores <- vroom('stores.csv') %>%
  mutate(Store = factor(Store), Type = factor(Type))
features <- vroom('features.csv') %>%
  mutate(Store = factor(Store))

## EDA
#####
# 
# ### Features Data
# 
# head(features)
# 
# ### Unemployment Rate
# 
# gplot(features, mapping=aes(x=Date, y=Unemployment)) + geom_point()
# 
# ### The Stores
# 
# stores_to_predict <- interaction(testData$Store, testData$Dept)
# levels(stores_to_predict)
# 
# stores_we_have <- interaction(trainData$Store, trainData$Dept)
# levels(stores_we_have)
# 
# unique_to_predict <- setdiff(as.character(stores_to_predict), 
#                              as.character(stores_we_have))
# unique_to_predict
# 
#####

## Data Cleaning
#####

features <- features %>%
  mutate(across(c(MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5), ~
                  replace_na(.x, 0))) %>%
  mutate(TotalMarkdown = MarkDown1 + MarkDown2 + MarkDown3 + 
           MarkDown4 + MarkDown5) %>%
  mutate(MarkdownFlag = ifelse(TotalMarkdown > 0, 1, 0)) %>%
  mutate(SuperBowl = ifelse(Date %in% c(dmy("12-2-10"), dmy("11-2-11"), 
                                        dmy("10-2-12"), dmy("8-2-13")), 
                            1, 0)) %>%
  mutate(LaborDay = ifelse(Date %in% c(dmy("10-9-10"), dmy("9-9-11"),
                                       dmy("7-9-12"), dmy("6-9-13")),
                           1, 0)) %>%
  mutate(Thanksgiving = ifelse(Date %in% c(dmy("26-11-10"), dmy("25-11-11"),
                                           dmy("23-11-12"), dmy("29-11-13")),
                               1, 0)) %>%
  mutate(Christmas = ifelse(Date %in% c(dmy("31-12-10"), dmy("30-12-11"),
                                        dmy("28-12-12"), dmy("27-12-13")),
                            1, 0)) %>%
  select(-MarkDown1,-MarkDown2,-MarkDown3,-MarkDown4,-MarkDown5,-IsHoliday)

feature_recipe <- recipe(~., data=features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment, impute_with = imp_vars(DecDate, Store))
imputed_features <- juice(prep(feature_recipe))

trainData <- left_join(trainData, imputed_features, by=c("Store", "Date")) %>%
  select(-IsHoliday)

testData <- left_join(testData, imputed_features, by=c("Store", "Date")) %>%
  select(-IsHoliday)

#####

## Store-Department Combos for Testing
#####

combo1_1 <- trainData %>%
  filter(Store == '1', Dept == '1')
  
combo30_60 <- trainData %>%
  filter(Store == '30', Dept == '60')
  
combo23_32 <- trainData %>%
  filter(Store == '23', Dept == '32')

#####

## Recipes for Testing
#####

walmart_recipe <- recipe(Weekly_Sales~., data=combo1_1) %>%
  step_date(Date, features="week") %>%
  step_range(Date_week, min=0, max=pi) %>%
  step_mutate(sinweek=sin(Date_week), cosweek=cos(Date_week)) %>%
  step_rm(Date) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(walmart_recipe)
bake(prepped_recipe, new_data = combo1_1)

walmart_recipe <- recipe(Weekly_Sales~., data=combo30_60) %>%
  step_date(Date, features="week") %>%
  step_range(Date_week, min=0, max=pi) %>%
  step_mutate(sinweek=sin(Date_week), cosweek=cos(Date_week)) %>%
  step_rm(Date) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(walmart_recipe)
bake(prepped_recipe, new_data = combo30_60)

walmart_recipe <- recipe(Weekly_Sales~., data=combo23_32) %>%
  step_date(Date, features="week") %>%
  step_range(Date_week, min=0, max=pi) %>%
  step_mutate(sinweek=sin(Date_week), cosweek=cos(Date_week)) %>%
  step_rm(Date) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(walmart_recipe)
bake(prepped_recipe, new_data = combo23_32)

#####

## Random Forest - 1;1: 7404, 30;60: 169, 23;32: 2988
#####

rand_for <- rand_forest(mtry = tune(),
                        min_n=tune(),
                        trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

walmart_workflow <- workflow() %>%
  add_recipe(walmart_recipe) %>%
  add_model(rand_for)

grid_of_tuning_params <- grid_regular(mtry(range=c(1,9)),
                                      min_n(),
                                      levels=5)

folds <- vfold_cv(combo1_1, v = 10, repeats=1)

folds <- vfold_cv(combo30_60, v = 10, repeats=1)

folds <- vfold_cv(combo23_32, v = 10, repeats=1)

CV_results <- walmart_workflow %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse))

CV_results %>%
  show_best(metric = "rmse", n = 1)

#####

## Boosted Trees - 1;1: 8088, 30;60: 175, 23;32: 3366
#####

library(bonsai)
library(lightgbm)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

walmart_workflow <- workflow() %>%
  add_recipe(walmart_recipe) %>%
  add_model(boost_model)

grid_of_tuning_params <- grid_regular(tree_depth(),
                                      trees(),
                                      learn_rate(),
                                      levels=5)

folds <- vfold_cv(combo1_1, v = 10, repeats=1)

folds <- vfold_cv(combo30_60, v = 10, repeats=1)

folds <- vfold_cv(combo23_32, v = 10, repeats=1)

CV_results <- walmart_workflow %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse))

CV_results %>%
  show_best(metric = "rmse", n = 1)

#####

## BART - 1;1: 7932, 30;60: 162, 23;32: 2469
#####

bart_model <- bart(trees=tune()) %>% 
  set_engine("dbarts") %>%
  set_mode("regression")

walmart_workflow <- workflow() %>%
  add_recipe(walmart_recipe) %>%
  add_model(bart_model)

grid_of_tuning_params <- grid_regular(trees(),
                                      levels=5)

folds <- vfold_cv(combo1_1, v = 10, repeats=1)

folds <- vfold_cv(combo30_60, v = 10, repeats=1)

folds <- vfold_cv(combo23_32, v = 10, repeats=1)

CV_results <- walmart_workflow %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse))

CV_results %>%
  show_best(metric = "rmse", n = 1)

#####