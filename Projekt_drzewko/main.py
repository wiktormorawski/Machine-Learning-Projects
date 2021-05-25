import random

import pandas
import numpy
from sklearn import preprocessing, metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt

# pandas.set_option('display.max_rows', 10)


def change_NA_to_values(db):
    categories = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "affordable"]
    for i in categories:
        db[i].fillna("UNKNOWN", inplace=True)


def missing_values(db):
    missing_data = db.isnull().sum()
    print("MISSING DATA IN DB |:\n", missing_data)
    cnt = 0
    for i in missing_data:
        cnt += i
    return cnt


def checking_values_with_categories(value, category):
    if (category == "buying" or category == "maint"):
        buying_maint_values = ["vhigh", "high", "med", "low"]
        if value not in buying_maint_values:
            return False
        else:
            return True
    if (category == "doors"):
        doors_values = ["2", "3", "4", "5more"]
        if value not in doors_values:
            return False
        else:
            return True
    if (category == "persons"):
        persons_values = ["2", "4", "more"]
        if value not in persons_values:
            return False
        else:
            return True
    if (category == "lug_boot"):
        persons_values = ["small", "med", "big"]
        if value not in persons_values:
            return False
        else:
            return True
    if (category == "safety"):
        safety_values = ["low", "med", "high"]
        if value not in safety_values:
            return False
        else:
            return True
    if (category == "affordable"):
        afford_values = ["unacc", "acc", "good", "vgood"]
        if value not in afford_values:
            return False
        else:
            return True


#                 db.loc[count, category] = numpy.nan
def check_if_valid_values(db):
    categories = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "affordable"]
    list_of_invalid = []
    for category in categories:
        count = 0
        last_value = ""
        for value in db[category]:
            result = checking_values_with_categories(value, category)
            if not result:
                list_of_invalid.append((count, category))
                db.loc[count, category] = last_value
                count += 1
            else:
                last_value = value
                count += 1
    return list_of_invalid


def classes_to_values(classes, columns):
    result = []
    ind = 0
    for i in classes:
        result.append(columns[ind])
        a = 0
        res = []
        for l in i:
            res.append((l, a))
            a += 1
        result.append(res)
        ind += 1
    return result


def to_numeric_change(base):
    le = preprocessing.LabelEncoder()
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "affordable"]
    classes = []
    for column in columns:
        new_name = le.fit_transform(base[column])
        base[column] = new_name
        classes.append(le.classes_)

    print(classes_to_values(classes, columns))


def values_to_array_of_absence(db):
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "affordable"]
    result = []
    for category in columns:
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count_0 = 0
        other_value = 0
        for value in db[category]:
            if value == 0:
                count_0 += 1
            elif value == 1:
                count_1 += 1
            elif value == 2:
                count_2 += 1
            elif value == 3:
                count_3 += 1
            else:
                other_value += 1

        result.append((category, count_0, count_1, count_2, count_3, other_value))
    return result


def training_set(db):
    cars_without_labels = db.iloc[0:, 0:6].values
    target = db.iloc[0:, 6].values
    X = preprocessing.scale(cars_without_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.5,
                                                        random_state=21)  # 50% training and 50% test
    return X_train, X_test, y_train, y_test


def Knn(X_train, X_test, y_train, y_test, k):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred_knn), metrics.confusion_matrix(y_test, y_pred_knn)


def NaiveBayes(X_train, X_test, y_train, y_test):
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred_nb), metrics.confusion_matrix(y_test, y_pred_nb)


def shuffle(db):
    return db.sample(frac=1).reset_index(drop=True)


def delete_too_much_classes(db):
    data1 = db[db["affordable"] == 2][0:600]
    data2 = db[db["affordable"] == 0]
    data3 = db[db["affordable"] == 1]
    data4 = db[db["affordable"] == 3]
    return pandas.concat([data1, data2, data3, data4])


def create_tree(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, metrics.accuracy_score(y_test, y_pred), metrics.confusion_matrix(y_test, y_pred)


def main():
    cars = pandas.read_csv("car.data", na_values=["-", "NA"])
    print("ILOSC WSZYSTKICH BLEDNYCH DANYCH TYPU NULL  \n--->>>  ", missing_values(cars))
    print("ILOSC WYSTAPIEN W KTORYM REKORDZIE BLEDNYCH DANYCH : \n", check_if_valid_values(cars))
    cars = shuffle(cars)
    change_NA_to_values(cars)
    to_numeric_change(cars)
    print("ILOSC WYSTĄPIEN WARTOSCI OD 0 DO 3 W DANYCH KATEGORIACH \n", values_to_array_of_absence(cars))
    X_train, X_test, y_train, y_test = training_set(cars)
    cars = delete_too_much_classes(cars)
    print(values_to_array_of_absence(cars))
    nb_accuracy, nb_matrix = NaiveBayes(X_train, X_test, y_train, y_test)
    knn3_accuracy, knn3_matrix = Knn(X_train, X_test, y_train, y_test, 3)
    knn11_accuracy, knn11_matrix = Knn(X_train, X_test, y_train, y_test, 11)
    knn19_accuracy, knn19_matrix = Knn(X_train, X_test, y_train, y_test, 11)
    clf, clf_accuracy, clf_matrix = create_tree(X_train, X_test, y_train, y_test)
    tree.plot_tree(clf, max_depth=2, fontsize=8, feature_names=cars.columns)
    plt.show()
    plt.style.use('classic')
    plot_y = [nb_accuracy, knn3_accuracy, knn11_accuracy, knn19_accuracy, clf_accuracy]
    plot_x = ["NB","KNN_3", "KNN_11", "KNN_19", "CLF_TREE"]
    plt.bar(plot_x, plot_y, color='red')
    plt.show()


main()
