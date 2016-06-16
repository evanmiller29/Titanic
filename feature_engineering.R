library(readr)
library(dplyr)
library(feather)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)

basepath <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(basepath)

train <- read_csv('train.csv') %>%
        mutate(set = 'train')

order_cols <- colnames(train)

test <- read_csv('test.csv') %>%
        mutate(Survived = 0,
               set = 'test')

test <- test[order]
full <- bind_rows(train, test)

format_cols <- function(x){
  
  x <- gsub(' ', '', x)
  x <- tolower(x)

}

feature_gen <- function(df, pred_missing){
  
  df$child  <- 0
  df$child[df$age < 18]  <- 1
  
  df$fare_band <- '30+'
  df$fare_band[df$fare < 30 & df$fare >= 20] <- '20-30'
  df$fare_band[df$fare < 20 & df$fare >= 10] <- '10-20'
  df$fare_band[df$fare < 10] <- '<10'
  
  df$title <- sapply(df$name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
  df$title <- sub(' ', '', df$title)
  df$title[df$title %in% c('Mme', 'Mlle')] <- 'Mlle'
  df$title[df$title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
  df$title[df$title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  
  df$familysize <- df$sibsp + df$parch + 1
  
  df$surname <- sapply(df$name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  df$familyid <- paste(as.character(df$familysize), df$surname, sep="")
  df$familyid[df$familysize <= 2] <- 'Small'
  
  famIDs <- data.frame(table(df$familyid))
  df$familyid[df$familyid %in% famIDs$Var1] <- 'Small'
  famIDs <- famIDs[famIDs$Freq <= 2,]
  
  df$familyid[df$familyid %in% famIDs$Var1] <- 'Small'
  
  df$familyid2 <- df$familyid
  df$familyid2 <- as.character(df$familyid2)
  df$familyid2[df$familysize <= 3] <- 'Small'

  embark_blank <- which(df$embarked == '')
  df$embarked[embark_blank] <- "S"
    
  if (pred_missing == TRUE){
  
    agefit <- rpart(age ~ pclass + sex + sibsp + parch + fare + title + embarked + familysize,
                      data=df[!is.na(df$age),], 
                      method="anova")
    df$age[is.na(df$age)] <- predict(agefit, df[is.na(df$age),])

    miss_fare <- which(is.na(df$fare))
    df$fare[miss_fare] <- median(df$fare, na.rm=TRUE)
    
    embark_na <- which(is.na(df$embarked))
    df$embarked[embark_na] <- median(df$embarked, na.rm=TRUE)
    
  }
  return(df)
}

colnames(full) <- format_cols(colnames(full))
colnames(train) <- format_cols(colnames(train))
colnames(test) <- format_cols(colnames(test))

train <- feature_gen(train, TRUE)
test <- feature_gen(test, TRUE)
full <- feature_gen(full, TRUE)

levels(test$title)

factorVars <- c('sex', 'fare_band', 'title', 'familyid', 'familyid2', 'embarked')

full[factorVars] <- lapply(full[factorVars], function(x) as.factor(x))
train[factorVars] <- lapply(train[factorVars], function(x) as.factor(x))
test[factorVars] <- lapply(test[factorVars], function(x) as.factor(x))

aggregate(survived ~ title + sex, data = train, FUN = function(x) {sum(x) / length(x)})
aggregate(survived ~ title + sex, data = train, FUN = function(x) length(x))

table(train$survived)
prop.table(table(train$survived))

summary(as.factor(train$sex))

### Total proportions
prop.table(table(train$sex, train$survived))

### Row by row proportions
prop.table(table(train$sex, train$survived), 1)

summary(train$age)

aggregate(survived ~ child + sex, data = train, FUN = sum) 
aggregate(survived ~ child + sex, data = train, FUN = length)
aggregate(survived ~ child + sex, data = train, FUN = function(x) {sum(x) / length(x)})

aggregate(survived ~ fare_band + pclass + sex, data = train, FUN=function(x) {sum(x) / length(x)})

# Much lower odds of survival if you're in class # 3
aggregate(survived ~ pclass + sex, data = train, FUN=function(x) {sum(x) / length(x)})

set.seed(415)

vars <- c('pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
            'embarked', 'title', 'familysize', 'familyid2')

fit <- randomForest(as.factor(survived) ~ pclass + sex + age + sibsp + parch + fare +
                      embarked + title + familysize + familyid2,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)

varImpPlot(fit)

vars %in% colnames(test)

prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$passengerid, Survived = prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)