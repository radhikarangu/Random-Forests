# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:51:49 2020

@author: RADHIKA
"""
###########################company data Assignment################


import pandas as pd
import numpy as np
company_data=pd.read_csv("D:\\ExcelR Data\\Assignments\\Random Forests\\Company_Data.csv",encoding = "ISO-8859-1")
company_data.columns
company_data.head()
company_data.shape
company_data.describe()
company_data.describe()
company_data.hist()
company_data.isnull().sum()
company_data['Urban'],Urban = pd.factorize(company_data['Urban'])
company_data['US'],US = pd.factorize(company_data['US'])
company_data.columns
company_data.head()
##Converting the sales  variable to bucketing. 
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in company_data.columns:
    if company_data[column_name].dtype == object:
        company_data[column_name] = le.fit_transform(company_data[column_name])
    else:
        pass
X=company_data.drop(['Sales'],axis=1)
X
company_data['Sales_levels']=np.where(company_data['Sales']>=7.5,'high','low')
y=company_data['Sales_levels']
y
y.value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(X_train,y_train)
model.estimators_
model.classes_
model.n_features_
model.n_classes_
model.n_outputs_

model.oob_score_
#####0.7214285714285714
##Predictions on train data
prediction = model.predict(X_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
accuracy
#####0.9964285714285714
##Accuracy
pred_test = model.predict(X_test)
acc_test =accuracy_score(y_test,pred_test)
###0.8083333333333333

## test data is less and train data is more means over fitting go for bagging and booting technique

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(X_train,y_train) #fitting the model 
bg.score(X_test,y_test) #test accuracy#0.81
bg.score(X_train,y_train) #train accuracy  0.96

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=5,learning_rate=1)
ada.fit(X_train,y_train)
bg.score(X_test,y_test) #test accuracy0.82
bg.score(X_train,y_train) #test accuracy0.9607

from sklearn.linear_model import LogisticRegression #importing logistc regression
from sklearn.svm import SVC
lr = LogisticRegression() 
dt = DecisionTreeClassifier()
svm = SVC(kernel= 'poly', degree=2)
evc = VotingClassifier(estimators=[('lr',lr),('dt',dt),('svm',svm)],voting='hard')

evc.fit(X_train,y_train)
evc.score(X_test,y_test)#0.74
evc.score(X_train,y_train)#0.89





##############fraud check data Assignment##############
import pandas as pd
import numpy as np
fraud= pd.read_csv("D:\\ExcelR Data\\Assignments\\Random Forests\\Fraud_check.csv")
fraud.columns
fraud.describe()
fraud.shape
fraud.isnull().sum
##Converting the Taxable income variable to bucketing. 
fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"
fraud.drop(["Taxable.Income"],axis=1,inplace=True)
fraud.columns
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
  
##Splitting the data into featuers and labels
features = fraud.iloc[:,0:5]
features
labels = fraud.iloc[:,5]
labels
## Collecting the column names
colnames = list(fraud.columns)
predictors = colnames[0:5]
target = colnames[5]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)
##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)

model.estimators_
model.classes_
model.n_features_
model.n_classes_
model.n_outputs_

model.oob_score_
#73%
##Predictions on train data
prediction = model.predict(x_train)

##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
accuracy
#0.9916666666666667
##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)
##Prediction on test data
pred_test = model.predict(x_test)

##Accuracy
acc_test =accuracy_score(y_test,pred_test)
##0.7666666666666667

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,max_features=1.0, n_estimators=20)
bg.fit(x_train,y_train) #fitting the model 
bg.score(x_test,y_test) #test accuracy#0.775
bg.score(x_train,y_train) #train accuracy  0.8895833333333333

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=5,learning_rate=1)
ada.fit(x_train,y_train)
bg.score(x_test,y_test) #test accuracy0.775
bg.score(x_train,y_train) #test accuracy0.8895



###################################################################
