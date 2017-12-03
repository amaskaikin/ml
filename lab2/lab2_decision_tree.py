# coding=utf-8
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# загрузка датасета
from sklearn.tree import DecisionTreeClassifier


def load_data(filename):
    return pd.read_csv(filename, header=None).values


# разделение датасета на тестовую и обучающую выборку
def split_dataset(test_size):
    dataset = load_data('fs.dataset.csv')
    site_attr = dataset[:, :-1]  # список атрибутов для каждого сайта
    site_class = dataset[:, -1]  # класс (результат) сайта (норм, подозрительный, фишинговый)
    site_class = site_class.astype(np.int64, copy=False)
    data_train, data_test, class_train, class_test = \
        train_test_split(site_attr, site_class, test_size=test_size, random_state=55)

    return data_train, class_train, data_test, class_test


def main():
    max_size = 0.4
    min_size = step = 0.1

    for size in np.arange(min_size, max_size, step):
        data_train, class_train, data_test, class_test = split_dataset(size)
        decision_forest = DecisionTreeClassifier()
        decision_forest = decision_forest.fit(data_train, class_train)
        decision_accuracy = decision_forest.score(data_test, class_test)

        random_forest = RandomForestClassifier()
        random_forest = random_forest.fit(data_train, class_train)
        random_accuracy = random_forest.score(data_test, class_test)

        print("Size: ", round(size, 1))
        print('DecisionTree accuracy: ', round(decision_accuracy, 10))
        print('RandomTree accuracy: ', round(random_accuracy, 10))


if __name__ == '__main__':
    main()
