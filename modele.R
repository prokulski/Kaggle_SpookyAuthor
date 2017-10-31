library(data.table) # for fread()

library(tidyverse)
library(tidytext)
library(stringr)

library(tm)
library(e1071) # SVM & Naive Bayes

# load data
train <- fread("./input/train.csv")


#### PRZYGOTOWANIE DANYCH ####

# split into words
train_dtm <- train %>%
  sample_frac(0.75) %>%
  rename(document_id = id, author_name = author) %>%
  unnest_tokens(word, text) %>%
  mutate(word = gsub(".", "", word, fixed = TRUE)) %>%
  mutate(word = gsub("'", "", word, fixed = TRUE)) %>%
  count(author_name, document_id, word) %>%
  ungroup() %>%
  filter(n > quantile(n, 0.25)) %>%
  # spread into wide
  spread(key=word, value=n, fill = 0) 

# change colnames
colnames(train_dtm) <- paste0(colnames(train_dtm), "_word")
train_dtm <- rename(train_dtm, author_name = author_name_word, document_id = document_id_word)

# train and test probes
ids <- sample(1:nrow(train_dtm), 0.8*nrow(train_dtm))

# test
test_dtm <- train_dtm[-ids, ]
org_test <- test_dtm[, 1:2] %>% unique()
test_dtm <- test_dtm[, -c(1:2)]

# train
train_dtm <- train_dtm[ids, ]
train_dtm$document_id <- NULL
train_dtm$author_name <- as.factor(train_dtm$author_name)



#### MODELE ####

# model NB
model_nb <- naiveBayes(author_name ~ ., data = train_dtm)

# predict
org_test$pred_bayes <- predict(model_nb, newdata = test_dtm)

# compare
table(org = org_test$author_name, pred = org_test$pred_bayes)

# accuracy 33%
sum(diag(table(org = org_test$author_name, pred = org_test$pred_bayes)))/nrow(org_test)



# model SVM
model_svm <- svm(author_name ~ ., data = train_dtm)

# predict
org_test$pred_svm <- predict(model_svm, newdata = test_dtm)

# compare
table(org = org_test$author_name, pred = org_test$pred_svm)

# accuracy 46%
sum(diag(table(org = org_test$author_name, pred = org_test$pred_svm)))/nrow(org_test)



# model nnet
library(nnet)
model_net <- nnet(author_name ~ .,
                  data = train_dtm,
                  size = 10,
                  rang = 0.5, 
                  maxit = 200,
                  MaxNWts = 20000)

org_test$pred_nnet <- predict(model_net, newdata = test_dtm, type = "class")

# compare
table(org = org_test$author_name, pred = org_test$pred_nnet)

# accuracy
sum(diag(table(org = org_test$author_name, pred = org_test$pred_nnet)))/nrow(org_test)

#  5 neurons = 48%
# 10 neurons = 




# KERAS
library(keras)

keras_train_x <- as.matrix(train_dtm[, -1])
keras_train_y <- to_categorical(train_dtm$author_name)

keras_test_x <- as.matrix(test_dtm)


model <- keras_model_sequential()

model %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = ncol(keras_train_x)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 4, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)


epochs_number <- 50
batch_size_number <- 128

history <- model %>%
  fit(keras_train_x,
      keras_train_y, 
      epochs = epochs_number,
      batch_size = batch_size_number, 
      validation_split = 0.2,
      verbose = 1)

plot(history)


# predict
org_test$pred_keras <- model %>% predict_classes(keras_test_x)

# compare
table(org = org_test$author_name, pred = org_test$pred_keras)

# accuracy
sum(diag(table(org = org_test$author_name, pred = org_test$pred_keras)))/nrow(org_test)
