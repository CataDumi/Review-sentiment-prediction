# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

labels = ['Positive Review', 'Negative Review']


def make_graph(name, y_test, y_pred, position):
    plt.subplot(position)

    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                cmap='Reds',
                xticklabels=labels,
                yticklabels=labels)
    plt.xticks(rotation=0)
    plt.title(f'Confusion Matrix based on {name} model')
    plt.xlabel('Predicted Reviews')
    plt.ylabel('Actual Reviews')


def confusion_matrix_and_acc_score(y_pred, y_test, name):
    print(f'{name}')
    print(f'Confusion matrix :\n{confusion_matrix(y_test, y_pred)}')
    print(f'Accuracy score :{accuracy_score(y_test, y_pred):.3f}')
    print(f'\n')

### Importing the dataset
dataset = pd.read_csv('IMDB Dataset.csv')

### Cleaning the texts

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# print(len(dataset.review))
print(f'Number of features allocated :{len(dataset.review) // 20}')
corpus = []
for i in range(0, len(dataset.review) // 20):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in all_stopwords]
    review = ' '.join(review)
    corpus.append(review)

# print(corpus)

### Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:2500, -1].values

### Converting the 'sentiment' column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
# print(y,len(y))

### Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

### Training the Naive Bayes model on the training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

### Predicting the test set results
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

### Making the confusion matrix and calc the accuracy score
confusion_matrix_and_acc_score(y_pred=y_pred, y_test=y_test, name='Results for Naive Bayes model')
graph_1 = make_graph(name='Naive Bayes', y_test=y_test, y_pred=y_pred, position=221)

################################### Models
### Training the Decision Tree model on the training set
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)

### Predicting the test set results

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

### Making the confusion matrix and calc the accuracy score
confusion_matrix_and_acc_score(y_pred=y_pred, y_test=y_test, name='Results for Decision Tree model')
graph_2 = make_graph(name='Decision Tree', y_test=y_test, y_pred=y_pred, position=222)

######################################
### Training the Random Forest Classifier model on the training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion='entropy', n_estimators=100)
classifier.fit(X_train, y_train)

### Predicting the test set results

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

### Making the confusion matrix and calc the accuracy score
confusion_matrix_and_acc_score(y_pred=y_pred, y_test=y_test, name='Results for Random Forest Classifier model')
graph_3 = make_graph(name='Random Forest Classifier', y_test=y_test, y_pred=y_pred, position=223)

######################################
### Training the SVM model on the training set
from sklearn.svm import SVC

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

### Predicting the test set results
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

### Making the confusion matrix and calc the accuracy score
confusion_matrix_and_acc_score(y_pred=y_pred, y_test=y_test, name='Results for SVM model')
graph_4 = make_graph(name='Support Vector Machine', y_test=y_test, y_pred=y_pred, position=224)

### Spacing graphs
plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.show()
