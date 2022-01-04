import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('diabetes.csv')
columns = ['Glucose', 'BloodPressure','SkinThickness','BMI','Insulin']

# Removeing zeros from the data

for column in columns :
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna= True))
    dataset[column] = dataset[column].replace(np.NaN,mean)

# splitting the data

x = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.3)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Training the Knn model

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classifier.fit(x_train,y_train)

# predecting the results

y_pred = classifier.predict(x_test)

# model evaluation

cm = confusion_matrix(y_test,y_pred)
print ('The confusion matrix : ')
print (cm)

print ('The F1 score : ')
print(f1_score(y_test,y_pred))

print('The Accuracy : ')
print(accuracy_score(y_test,y_pred))



