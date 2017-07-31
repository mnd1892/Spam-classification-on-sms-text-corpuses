# load library for text mining
library("tm")

# load corpus for spam in train data
train_spam <- VCorpus(DirSource("train/spam"))

# load corpus for ham in train data
train_ham <- VCorpus(DirSource("train/ham"))

#combine ham and spam corpus for training set
all_data_train <- c(train_spam,train_ham)

#inspect element of training set
inspect(all_data_train[5])

#convert text to lower case
all_data_train_clean <- tm_map(all_data_train,content_transformer(tolower))

#remove numeric values from corpus 
all_data_train_clean <- tm_map(all_data_train_clean,removeNumbers)

all_data_train_clean <- tm_map(all_data_train_clean,removeWords,stopwords())

#create function to remove punctuation to avoid caveat problem
replacePunctuation <- function(x) {
gsub("[[:punct:]]+", " ", x)
}

#apply function
all_data_train_clean <- tm_map(all_data_train_clean,replacePunctuation)

#load SnowballC package for stemming operation
library(SnowballC)

#remove stemming 
all_data_train_clean <- tm_map(all_data_train_clean,stemDocument)

#remove white space
all_data_train_clean <- tm_map(all_data_train_clean,stripWhitespace)

#because of stripWhitespace function we need to convert raw text into plain text again to cast corpus in document term matrix
all_data_train_clean <- tm_map(all_data_train_clean,PlainTextDocument)

#check raw text
as.character(all_data_train_clean[[1]])

#cast corpus to document term matrix
all_data_train_dtm <- DocumentTermMatrix(all_data_train_clean)

#another way to create document term matrix
all_data_dtm2 <- DocumentTermMatrix(all_data_train,control=list(tolower=TRUE,removeNumbers=TRUE,stopwords=TRUE,removePunctuation=TRUE,stemming=TRUE))

#create class labels and assign values in class labels
spam_label <- vector(mode = "character",length = 123)

for(i in 1:length(spam_label)){
spam_label[i] <- "spam"}

ham_label <- vector(mode = "character",length = 340)

for(i in 1:length(ham_label)){
ham_label[i] <- "ham"}

all_label <- factor(c(spam_label,ham_label))

#load test dataset and apply each and every function which was applied to train data set
all_data_test <- c(VCorpus(DirSource("test/spam")),VCorpus(DirSource("test/ham")))

all_data_test_clean <- tm_map(all_data_test,content_transformer(tolower))

all_data_test_clean <- tm_map(all_data_test_clean,removeNumbers)

all_data_test_clean <- tm_map(all_data_test_clean,removeWords,stopwords())

all_data_test_clean <- tm_map(all_data_test_clean,replacePunctuation)

all_data_test_clean <- tm_map(all_data_test_clean,stemDocument)

all_data_test_clean <- tm_map(all_data_test_clean,stripWhitespace)

all_data_test_clean <- tm_map(all_data_test_clean,PlainTextDocument)

all_data_test_dtm <- DocumentTermMatrix(all_data_test_clean)

#create class labels for test data
test_spam_label <- vector(mode = "character",length = 130)

for(i in 1:length(test_spam_label)){
    test_spam_label[i] <- "spam"}


test_ham_label <- vector(mode = "character",length = 348)

for(i in 1:length(test_ham_label)){
   test_ham_label[i] <- "ham"}

test_label <- factor(c(test_spam_label,test_ham_label))

#load library for wordcloud
library(wordcloud)

#generate word cloud for train ham and spam data


#generate word cloud for train data
wordcloud(all_data_train_clean,min.freq = 50,random.order = FALSE)

wordcloud(all_data_train_clean,min.freq = 50)

#generate word clouds for spam and ham train data

train_spam <- VCorpus(DirSource("train/spam"))
train_spam_clean <- tm_map(train_spam,content_transformer(tolower))
train_spam_clean <- tm_map(train_spam_clean,removeNumbers)
train_spam_clean <- tm_map(train_spam_clean,removeWords,stopwords())
replacePunctuation <- function(x) {
     gsub("[[:punct:]]+", " ", x)
}
train_spam_clean <- tm_map(train_spam_clean,replacePunctuation)
train_spam_clean <- tm_map(train_spam_clean,stripWhitespace)
train_spam_clean <- tm_map(train_spam_clean,PlainTextDocument)
wordcloud(train_spam_clean,min.freq = 50,random.order = FALSE)
wordcloud(train_spam_clean,min.freq = 50)

#word cloud for ham data
train_ham <- VCorpus(DirSource("train/ham"))
train_ham_clean <- tm_map(train_ham,content_transformer(tolower))
train_ham_clean <- tm_map(train_ham_clean,removeNumbers)
train_ham_clean <- tm_map(train_ham_clean,removeWords,stopwords())
replacePunctuation <- function(x) {
     gsub("[[:punct:]]+", " ", x)
}
train_ham_clean <- tm_map(train_ham_clean,replacePunctuation)
train_ham_clean <- tm_map(train_ham_clean,stripWhitespace)
train_ham_clean <- tm_map(train_ham_clean,PlainTextDocument)
wordcloud(train_ham_clean,min.freq = 50,random.order = FALSE)
wordcloud(train_ham_clean,min.freq = 50)

#create document term matrix with most frequent words means words with frequency > 5 for train data

train_frequent_words <- findFreqTerms(all_data_train_dtm,5)

freq_train_dtm <- all_data_train_dtm[,train_frequent_words]

#create document term matrix with most frequent words means words with frequency > 5 for test data

test_frequent_words <- findFreqTerms(all_data_test_dtm,5)

freq_test_dtm <- all_data_test_dtm[,test_frequent_words]

#convert data from categorical to numerical

convert_counts <- function(x){x <- ifelse(x > 0,"Yes","No")}

#generate train data and test data for classification

train_data <- apply(freq_train_dtm,MARGIN = 2,convert_counts)

test_data <- apply(freq_test_dtm,MARGIN = 2,convert_counts)

#load library for classification

library("e1071")

#create training model(classification model)

spam_classifier <- naiveBayes(train_data,all_label)

#get all prediction

spam_prediction <- predict(spam_classifier,test_data)

library(gmodels)

#check for accuracy of classification

CrossTable(spam_prediction,test_label,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))

#improve classification performance with laplace value

sms_classifier2 <- naiveBayes(train_data, all_label,laplace = 1)

sms_test_pred2 <- predict(sms_classifier2,test_data)

CrossTable(sms_test_pred2,test_label,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))

