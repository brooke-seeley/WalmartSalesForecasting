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

### Features Data

head(features)

### Unemployment Rate

gplot(features, mapping=aes(x=Date, y=Unemployment)) + geom_point()

### The Stores

stores_to_predict <- interaction(testData$Store, testData$Dept)
levels(stores_to_predict)

stores_we_have <- interaction(trainData$Store, trainData$Dept)
levels(stores_we_have)

unique_to_predict <- setdiff(as.character(stores_to_predict), 
                             as.character(stores_we_have))
unique_to_predict