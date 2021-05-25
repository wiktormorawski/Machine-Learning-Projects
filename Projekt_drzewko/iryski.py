#                  ZADANIE 1
import pandas

pandas.set_option('display.max_rows', 150)
iryski = pandas.read_csv("iris.csv")


# print(iryski['sepal.length'])
def prediction_iryski_moje_kochane(sepal_length, sepal_width, petal_length, petal_width):
    if petal_width < 1.0:
        return "Setosa"
    else:
        if petal_length > 4.9:
            return "Virginica"
        else:
            return "Versicolor"


def myPredict():
    count = 0
    for i in iryski.values:
        result = prediction_iryski_moje_kochane(i[0], i[1], i[2], i[3])
        if result == i[4]:
            count += 1
    return (count / 150)


print(myPredict())

# DZIALA NA POZIOMIE 95 PROCENT

import pandas
pandas.set_option('display.max_rows', 150)
iryski = pandas.read_csv("iris.csv")

#%%

from sklearn import tree
x = iryski.iloc[0:, 0:4].values
y = iryski.iloc[0:, 4].values

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train, y_train, x_train.size, y_train.size)
print(x_test, y_test,x_test.size, y_test.size)

#%%

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
tree.plot_tree(clf)

#%%

from sklearn import metrics
y_pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

#%%
import matplotlib
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
