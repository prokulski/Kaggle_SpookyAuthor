# https://www.kaggle.com/jayjay75/text2vec-glmnet

library(data.table) # for fread()

library(tidyverse)
library(tidytext)
library(stringr)


library(text2vec)

library(Matrix) #cBind



train = read_csv("./input/train.csv")
test = read_csv("./input/test.csv")


######################################################################
# extra features ....
######################################################################

train_extra1 <- train %>%
  mutate(Nchar = str_count(text)) %>%
  mutate(Ncommas = str_count(text, ",")) %>%
  mutate(Nsemicolumns = str_count(text, ";")) %>%
  mutate(Ncolumns = str_count(text, ":")) %>%
  mutate(Nother = str_count(text, "[?!\\.]")) %>%
  mutate(Ncapital = str_count(text, "[A-Z]"))

test_extra1 <- test %>%
  mutate(Nchar = str_count(text)) %>%
  mutate(Ncommas = str_count(text, ",")) %>%
  mutate(Nsemicolumns = str_count(text, ";")) %>%
  mutate(Ncolumns = str_count(text, ":")) %>%
  mutate(Nother = str_count(text, "[?!\\.]")) %>%
  mutate(Ncapital = str_count(text, "[A-Z]"))


######################################################################
# Vocabulary-based vectorization
# Let’s first create a vocabulary-based DTM. Here we collect unique terms from all documents and mark each of them with a unique ID using the create_vocabulary() function. We use an iterator to create the vocabulary.
# define preprocessing function and tokenization function
######################################################################
prep_fun  = function(x) {
  stringr::str_replace_all(tolower(x), "[^[:alpha:]]", " ")
}

tok_fun = word_tokenizer

it_train = itoken(train$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$id, 
                  progressbar = FALSE)

vocab = create_vocabulary(it_train)

#Now that we have a vocabulary, we can construct a document-term matrix.
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

#extra features bind
dtm_train <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train)




# KERAS
library(keras)

train_y <- to_categorical(as.factor(train$author))

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 2, activation = 'relu', input_shape = ncol(dtm_train)) %>% 
#  layer_dense(units = 10, activation = 'relu') %>%
#  layer_dropout(rate = 0.3) %>%
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
  fit(dtm_train,
      train_y, 
      epochs = epochs_number,
      batch_size = batch_size_number, 
      validation_split = 0.2,
      verbose = 1)

plot(history)





#We have successfully fit a model to our DTM. Now we can check the model’s performance on test data. Note that we use exactly the same functions from prepossessing and tokenization. Also we reuse/use the same vectorizer - function which maps terms to indices.
it_test = test$text %>% 
  prep_fun %>% 
  tok_fun %>% 
  itoken(ids = test$id,  progressbar = FALSE)

dtm_test = create_dtm(it_test, vectorizer)
dtm_test <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test)




pred_keras <- model %>% predict_proba(dtm_test)

pred <- data_frame(id = rownames(dtm_test), EAP = pred_keras[,2], HPL= pred_keras[,3], MWS=pred_keras[,4])

head(pred)
