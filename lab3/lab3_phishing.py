# coding=utf-8
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import metrics


def load_data(filename):
    return pd.read_csv(filename, header=None).values


# разделение датасета на тестовую и обучающую выборку
def process_dataset(name):
    dataset = load_data(name)
    site_attr = dataset[:, :-1]  # список атрибутов для каждого сайта
    site_class = dataset[:, -1]  # класс (результат) сайта (норм, подозрительный, фишинговый)
    site_class = site_class.astype(np.int64, copy=False)

    return site_attr, site_class


def train_split_dataset(occ_attr, occ_class, test_size, rnd_state):
    data_train, data_test, class_train, class_test = \
        train_test_split(occ_attr, occ_class, test_size=test_size, random_state=rnd_state)

    print_dataset_info(class_train, data_train)
    print_dataset_info(class_test, data_test)

    # train_test_visualization(data_train, data_test, class_train, class_test)
    return data_train, data_test, class_train, class_test


def visualize_data(is2d, is3d, is2plots, site_attr=None, site_class=None, data_train=None, data_test=None,
                   class_train=None, class_test=None):
    if is2d is True:
        data_2d_visualization(site_attr, site_class)
    if is3d is True:
        data_3d_visualization(site_attr, site_class)
    if is2plots is True:
        train_test_visualization(data_train, data_test, class_train, class_test)


def data_2d_visualization(site_attr, site_class):
    plt.figure(figsize=(6, 5))
    for label, marker, color in zip(
            range(-1, 2), ('x', 'o', '^'), ('blue', 'red', 'green')):
        # Вычисление коэффициента корреляции Пирсона
        r = pearsonr(site_attr[:, 5][site_class == label], site_attr[:, 6][site_class == label])
        plt.scatter(x=site_attr[:, 5][site_class == label],
                    y=site_attr[:, 6][site_class == label],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {:}, R={:.2f}'.format(label, r[0])
                    )

    plt.title('Phishing Website Data Set')
    plt.xlabel('Web Traffic')
    plt.ylabel('URL Length')
    plt.legend(loc='upper right')
    plt.show()


def data_3d_visualization(site_attr, site_class):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label, marker, color in zip(
            range(-1, 2), ('x', 'o', '^'), ('blue', 'red', 'green')):
        # Вычисление коэффициента корреляции Пирсона
        ax.scatter(site_attr[:, 1][site_class == label],
                   site_attr[:, 5][site_class == label],
                   site_attr[:, 6][site_class == label],
                   marker=marker,
                   color=color,
                   s=40,
                   alpha=0.7,
                   label='class {:}'.format(label)
                   )

    ax.set_xlabel('popUpWidnow')
    ax.set_ylabel('Web Traffic')
    ax.set_zlabel('URL Length')

    plt.title('Phishing Website Data Set')
    plt.legend(loc='upper right')

    plt.show()


def train_test_visualization(data_train, data_test, class_train, class_test):
    std_scale = preprocessing.StandardScaler().fit(data_train)
    data_train = std_scale.transform(data_train)
    data_test = std_scale.transform(data_test)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

    for a, x_dat, y_lab in zip(ax, (data_train, data_test), (class_train, class_test)):

        for label, marker, color in zip(
                range(-1, 2), ('x', 'o', '^'), ('blue', 'red', 'green')):
            a.scatter(x=x_dat[:, 5][y_lab == label],
                      y=x_dat[:, 6][y_lab == label],
                      marker=marker,
                      color=color,
                      alpha=0.7,
                      label='class {}'.format(label)
                      )

        a.legend(loc='upper right')

    ax[0].set_title('Training Dataset')
    ax[1].set_title('Test Dataset')
    f.text(0.5, 0.04, 'Web Traffic (standardized)', ha='center', va='center')
    f.text(0.08, 0.5, 'URL Length (standardized)', ha='center', va='center', rotation='vertical')
    plt.show()


def linear_discriminant_analysis(data_train, class_train):
    sklearn_lda = LDA()
    sklearn_transf = sklearn_lda.fit(data_train, class_train).transform(data_train)

    plt.figure(figsize=(8, 8))
    for label, marker, color in zip(
            range(-1, 2), ('x', 'o', '^'), ('blue', 'red', 'green')):
        plt.scatter(x=sklearn_transf[class_train == label],
                    y=sklearn_transf[class_train == label],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {}'.format(label))

    plt.xlabel('vector 1')
    plt.ylabel('vector 2')

    plt.legend()
    # Визуализация разбиения классов после линейного преобразования LDA
    plt.title('Most significant singular vectors after linear transformation via LDA')

    plt.show()


def discriminant_analysis(fanalysis, data_train, data_test, class_train, class_test, label):
    fanalysis.fit(data_train, class_train)
    pred_train = fanalysis.predict(data_train)

    print(label)
    print('The accuracy of the classification on the training set of data')
    print('{:.2%}'.format(metrics.accuracy_score(class_train, pred_train)))

    pred_test = fanalysis.predict(data_test)

    print('The accuracy of classification on the test data set')
    print('{:.2%}'.format(metrics.accuracy_score(class_test, pred_test)))


def print_dataset_info(site_class, site_attr):
    print('Number of records:', site_class.shape[0])
    print('Number of characters:', site_attr.shape[1])

    print('Class 0 (Normal): {:.2%}'.format(list(site_class).count(-1) / site_class.shape[0]))
    print('Class 1 (Suspicious): {:.2%}'.format(list(site_class).count(0) / site_class.shape[0]))
    print('Class 2 (Phishing): {:.2%}'.format(list(site_class).count(1) / site_class.shape[0]))


def init(name):
    site_attr, site_class = process_dataset(name)
    print_dataset_info(site_class, site_attr)

    return site_attr, site_class


def main():
    site_attr, site_class = init('fs.dataset.csv')
    data_train, data_test, class_train, class_test = train_split_dataset(site_attr, site_class, 0.3, 55)

    visualize_data(is2d=True, is3d=True, is2plots=True, site_attr=site_attr, site_class=site_class,
                   data_train=data_train, data_test=data_test, class_train=class_train, class_test=class_test)

    linear_discriminant_analysis(data_train, class_train)
    discriminant_analysis(LDA(), data_train, data_test, class_train, class_test, 'Linear discriminant analysis')
    discriminant_analysis(QDA(), data_train, data_test, class_train, class_test, 'Quadratic discriminant analysis')


if __name__ == '__main__':
    main()
