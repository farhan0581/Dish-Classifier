from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
from numpy import array


handle=open('Farhan_Sample Dishname file for vegnonveg indicator.csv','rt')
reader=csv.DictReader(handle)

#training data
full_set=[]
#response matrix
response_full=[]
#count of veg dishes
c1=0
#count of non-veg dishes
c2=0
#dish which is both veg and non-veg
c3=0

for line in reader:
	full_set.append(line['dish name'])
	if line['veg_ind']==line['nonveg_ind']:
		response_full.append(2)
		c3=c3+1
	elif line['veg_ind']=='1':
		response_full.append(1)
		c1=c1+1
	elif line['nonveg_ind']=='1':
		response_full.append(0)
		c2=c2+1

# print c1,c2,c3
# print response_full

response_full=array(response_full)
# print response_full.shape


#building the vocabulary from the words of dishes
count_vectorizer = CountVectorizer(stop_words='english')
full_set_fit=count_vectorizer.fit_transform(full_set)
# print full_set_fit.shape

#finding the term frequency
tfidf_transformer = TfidfTransformer()
full_set_fit_tfidf = tfidf_transformer.fit_transform(full_set_fit)


#splitting the dataset into train and test
X_train, X_test, y_train, y_test=train_test_split(full_set_fit_tfidf,response_full,test_size=0.3)


#training on the training dataset naeive baiyes
clf = MultinomialNB().fit(X_train, y_train)

#training using k neighbour
knn=KNeighborsClassifier(n_neighbors=5)


#k-fold testing on the dataset
clf_kfold=MultinomialNB().fit(full_set_fit_tfidf, response_full)

scores=cross_val_score(clf_kfold,full_set_fit_tfidf,response_full,cv=10,scoring='accuracy')
print scores
print 'mean score='+str(scores.mean()*100)


#testing on some new dataset
docs_new = ['fish rice', 'shrimp curry','fish fry','chily chicken','dal punjabi','bhaji tarkari',
				'murg musallam','Fried Rice','birthday cake','mutton curry','lamb champ']

docs_new_s = count_vectorizer.transform(docs_new)
docs_new_tfidf = tfidf_transformer.transform(docs_new_s)

predicted = clf.predict(X_test)
knn.fit(X_train,y_train)

knn_predict=knn.predict(X_test)


accuracy=metrics.accuracy_score(y_test,predicted)

accuracy_knn=metrics.accuracy_score(y_test,knn_predict)

print accuracy*100
print accuracy_knn*100

doc_pre=clf.predict(docs_new_tfidf)

print "======================================="
for i in range(len(docs_new)):
	if doc_pre[i]==1:
		print docs_new[i]+"--> Veg"
	elif doc_pre[i]==0:
		print docs_new[i]+"--> Non Veg"

print "========================================"

# precision=metrics.precision_score(y_test, predicted)
# recall=metrics.recall_score(y_test, predicted)
# print recall
# print precision


# print "Vocabulary:", count_vectorizer.vocabulary_
# count_vectorizer.get_feature.names()
# freq_term_matrix = count_vectorizer.transform(train_set)
# print freq_term_matrix.todense()