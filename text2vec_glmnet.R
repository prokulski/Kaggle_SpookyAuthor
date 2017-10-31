# https://www.kaggle.com/jayjay75/text2vec-glmnet

library(text2vec)
library(data.table)
library(readr)
library(stringr)
library(tidyverse)
library(Matrix)

set.seed(2016)

TokenizeBook <- function(book) {
  itoken(book, tolower, 
         tokenizer = word_tokenizer, chunks_number = 10, progessbar = F)
}
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

train_tokens = train$text %>% 
  prep_fun %>% 
  tok_fun

it_train = itoken(train_tokens, 
                  ids = train$id,
                  # turn off progressbar because it won't look nice
                  progressbar = FALSE)

vocab = create_vocabulary(it_train)
vocab

#Now that we have a vocabulary, we can construct a document-term matrix.
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

#extra features bind
dtm_train <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train)

############################################################################################################################################
#Now we are ready to fit our first model. Here we will use the glmnet package to fit a logistic regression model with an L1 penalty and 4 fold cross-validation.
############################################################################################################################################

library(glmnet)
NFOLDS = 8
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['author']], 
                              family = 'multinomial', 
                              # L1 penalty
                              alpha = 1,
                              type.measure = "class",
                              # n-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)

plot(glmnet_classifier)
title("vocabulary",line=2.5)

#We have successfully fit a model to our DTM. Now we can check the model’s performance on test data. Note that we use exactly the same functions from prepossessing and tokenization. Also we reuse/use the same vectorizer - function which maps terms to indices.
it_test = test$text %>% 
  prep_fun %>% 
  tok_fun %>% 
  itoken(ids = test$id,  progressbar = FALSE)

dtm_test = create_dtm(it_test, vectorizer)
dtm_test <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test)


preds = data.frame(id=test$id, predict(glmnet_classifier, dtm_test, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_vocab.csv")

######################################################################
#Pruning vocabulary
######################################################################
stop_words <- c(tm::stopwords("en"),"it","s","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours")
vocab = create_vocabulary(it_train, stopwords = stop_words)

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 2, 
                                doc_proportion_max = 0.9,
                                doc_proportion_min = 0.0001)
vectorizer = vocab_vectorizer(pruned_vocab)
# create dtm_train with new pruned vocabulary vectorizer
dtm_train  = create_dtm(it_train, vectorizer)
dtm_train <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train)

dtm_test   = create_dtm(it_test, vectorizer)
dtm_test <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test)

glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['author']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "class",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

plot(glmnet_classifier)
title("pruned",line=2.5)
preds = data.frame(id=test$id,predict(glmnet_classifier, dtm_test, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_vocab_pruned.csv")

######################################################################
#N-grams
######################################################################
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))
#vocab = vocab %>% prune_vocabulary(term_count_min = 2, doc_proportion_max = 0.9, doc_proportion_min = 0.001)

bigram_vectorizer = vocab_vectorizer(vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_train <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train)

dtm_test = create_dtm(it_test, bigram_vectorizer)
dtm_test <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test)

glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['author']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "class",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

plot(glmnet_classifier)
title("2N-grams",line=2.5)
preds = data.frame(id=test$id,predict(glmnet_classifier, dtm_test, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_vocab_2N-grams.csv")

######################################################################
#3N-grams
######################################################################
vocab = create_vocabulary(it_train, ngram = c(1L, 3L))
#vocab = vocab %>% prune_vocabulary(term_count_min = 2,   doc_proportion_max = 0.9,doc_proportion_min = 0.001)

trigram_vectorizer = vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, trigram_vectorizer)
dtm_train <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train)

dtm_test <- create_dtm(it_test, trigram_vectorizer)
dtm_test <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test)

glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['author']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "class",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

plot(glmnet_classifier)
title("3N-grams",line=2.5)
preds = data.frame(id=test$id,predict(glmnet_classifier, dtm_test, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_vocab_3N-grams.csv")

############################################################################################################################################  
#Normalization
############################################################################################################################################

dtm_train_l1_norm = normalize(dtm_train, "l1")
dtm_train_l1_norm <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train_l1_norm)

dtm_test_l1_norm = normalize(dtm_test, "l1")
dtm_test_l1_norm <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test_l1_norm)

glmnet_classifier = cv.glmnet(x = dtm_train_l1_norm, y = train[['author']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "class",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)




plot(glmnet_classifier)
title("3N-grams Normalization",line=2.5)
preds = data.frame(id=test$id,predict(glmnet_classifier, dtm_test_l1_norm, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_vocab_Normalization.csv")


######################################################################
#Feature hashing
######################################################################
h_vectorizer = hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L, 3L))
dtm_train = create_dtm(it_train, h_vectorizer)

glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['author']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "class",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

plot(glmnet_classifier)
title("Feature hashing",line=2.5)

dtm_test = create_dtm(it_test, h_vectorizer)

preds = data.frame(id=test$id,predict(glmnet_classifier, dtm_test, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_Feature hashing.csv")

######################################################################
#TF-IDF
######################################################################
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# define tfidf model
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
dtm_train_tfidf <- cBind(train_extra1$Nother,train_extra1$Nchar,train_extra1$Ncommas,train_extra1$Nsemicolumns,train_extra1$Ncolumns,train_extra1$Ncapital  , dtm_train_tfidf)



# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>% 
  transform(tfidf)

dtm_test_tfidf <- cBind(test_extra1$Nother,test_extra1$Nchar,test_extra1$Ncommas,test_extra1$Nsemicolumns,test_extra1$Ncolumns,test_extra1$Ncapital  , dtm_test_tfidf)

glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[['author']], 
                              family = 'multinomial', 
                              alpha = 1,
                              type.measure = "class",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

plot(glmnet_classifier)
title("TF-IDF",line=2.5)


preds = data.frame(id=test$id,predict(glmnet_classifier, dtm_test_tfidf, type = 'response'))
names(preds)[2] <- "EAP"
names(preds)[3] <- "HPL"
names(preds)[4] <- "MWS"

write_csv(preds, "glmnet_benchmark_TF-IDF.csv")






