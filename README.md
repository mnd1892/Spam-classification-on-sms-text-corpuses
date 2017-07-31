Report:
Steps:
1) create corpues from spam and ham data for training and testing models both
2) transform text to find most frequent words
3) transformation process includes : removing of punctuation, stopwords(e.g.: the,that,this,for), whitespace, stemmed words,
 numbers, convert text to lower case,
4) generated wordcloud to find most frequent words
5) generated new document term matrices for test and train data with frequency > 5
6) created function to convert categarical(ordinal data) to numerical format
7) applied naive bayes classification to check accuracy of model.

From comparision of word clouds of spam and ham it's clear that spam and ham data have different frequent words by counting
probability of these words we can assign different class labels for spam and ham

To improve efficiency of classification model we can apply laplacian correction we can get less misclassified instances
