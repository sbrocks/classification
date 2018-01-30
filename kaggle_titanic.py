# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the training dataset
dataset1 = pd.read_csv('train.csv')
X_train=dataset1.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']].values 
y_train=dataset1.loc[:,'Survived'].values 

# Importing the test dataset
dataset2 = pd.read_csv('test.csv')
X_test=dataset2.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']].values 

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [2,5]])
X_train[:, [2,5]] = imputer.transform(X_train[:, [2,5]])

imputer = imputer.fit(X_test[:, [2,5]])
X_test[:, [2,5]] = imputer.transform(X_test[:, [2,5]])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train=X_train[:,1:]

X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder.fit_transform(X_test).toarray()
X_test=X_test[:,1:]

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
np.savetxt('titanic.csv',y_pred,delimiter=",")
"""
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)  """