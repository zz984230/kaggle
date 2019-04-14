import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import linear_model


class Lazy(object):
    def __init__(self, method):
        self.method = method
        self.method_name = method.__name__

    def __get__(self, instance, owner):
        value = self.method(instance)
        setattr(instance, self.method_name, value)
        return value


class CreditOriData(object):
    def __init__(self,
                 ori_train_path="../data/ori/cs-training.csv",
                 ori_test_path="../data/ori/cs-test.csv"):
        self.__ori_train_path = ori_train_path
        self.__ori_test_path = ori_test_path
        self.__ori_df = pd.DataFrame()
        self.__ori_test_df = pd.DataFrame()

    def load_train_data(self):
        """
        加载原始训练数据
        """
        df = pd.read_csv(self.__ori_train_path)
        self.__ori_df = df.drop(columns="Unnamed: 0", axis=1)

    def load_test_data(self):
        df = pd.read_csv(self.__ori_test_path)
        self.__ori_test_df = df.drop(columns="Unnamed: 0", axis=1)

    @property
    def ori_test_data(self):
        return self.__ori_test_df

    @property
    def ori_train_data(self):
        return self.__ori_df

    def __print_data_info(self):
        """
        打印原始数据相关信息
        """
        print(self.__ori_df.columns)
        # print(self.__ori_df.describe())
        # print(self.__ori_df.dropna().describe())

    def checkout_data(self):
        """
        观察原始数据
        """
        self.__print_data_info()


class CreditCleanedData(object):
    def __init__(self,
                 ori_train_data,
                 ori_test_data,
                 feature_distribution_pic_path="../data/pic/feature_distribution",
                 heatmap_pic_path="../data/pic/heatmap",
                 cleaned_test_data_path="../data/preprocess/cleaned_test.csv"):
        self.__ori_train_df = ori_train_data
        self.__ori_test_df = ori_test_data
        self.__feature_distribution_pic_path = feature_distribution_pic_path
        self.__heatmap_pic_path = heatmap_pic_path
        self.__cleaned_test_path = cleaned_test_data_path

    def __clean_na(self):
        """
        去除NA行
        """
        self.__ori_train_df = self.__ori_train_df.dropna()
        self.__ori_test_df = self.__ori_test_df.dropna()

    def __fill_na(self):
        self.__ori_train_df[self.__ori_train_df.isna()] = 0
        self.__ori_test_df[self.__ori_test_df.isna()] = 0

    def __clean_debt_ratio_abnormal(self):
        """
        去除异常数据
        """
        self.__ori_train_df = self.__ori_train_df[self.__ori_train_df["MonthlyIncome"] < 2]

    def __save_cleaned_test_data(self):
        self.__ori_test_df.to_csv(self.__cleaned_test_path, index=False)

    def clean_data(self):
        """
        数据清洗
        """
        self.__fill_na()
        # self.__clean_debt_ratio_abnormal()
        self.__save_cleaned_test_data()

    @property
    def cleaned_train_data(self):
        return self.__ori_train_df

    @property
    def cleaned_test_data(self):
        return self.__ori_test_df

    def draw_feature_distribution(self):
        """
        画特征分布图、特征散点图
        """
        if os.path.exists(self.__feature_distribution_pic_path):
            return

        g = sns.PairGrid(self.__ori_train_df)
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter)
        plt.savefig(self.__feature_distribution_pic_path)

    def draw_heatmap(self):
        """
        画特征间相关性热力图
        """
        if os.path.exists(self.__heatmap_pic_path):
            return

        plt.figure(figsize=(18, 12))
        heat_corr = self.__ori_train_df.corr()
        sns.heatmap(heat_corr)
        plt.savefig(self.__heatmap_pic_path)


class CreditOversamplingData(object):
    def __init__(self, ori_cleaned_data, oversampling_data_path="../data/preprocess"):
        self.__ori_cleaned_data = ori_cleaned_data
        self.__oversampling_data_path = oversampling_data_path
        self.__oversampling_df = None

    def oversampling_data_by_smote(self):
        """
        SMOTE过采样
        """
        smote_file_path = os.path.join(self.__oversampling_data_path, "SMOTE_train.csv")
        if os.path.exists(smote_file_path):
            return

        ori_X = self.__ori_cleaned_data.drop(columns="SeriousDlqin2yrs", axis=1)
        ori_y = self.__ori_cleaned_data["SeriousDlqin2yrs"]
        s = SMOTE()
        oversampling_X, oversampling_y = s.fit_resample(ori_X, ori_y)
        self.__oversampling_df = pd.concat([pd.DataFrame(oversampling_y),
                                            pd.DataFrame(oversampling_X)], axis=1)
        self.__oversampling_df.columns = ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
                                          'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                                          'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                                          'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                                          'NumberOfDependents']
        self.__oversampling_df.to_csv(smote_file_path, index=False)

    @Lazy
    def smote_data(self):
        """
        预处理后的训练集
        """
        if self.__oversampling_df is None:
            self.__oversampling_df = pd.read_csv(os.path.join(self.__oversampling_data_path, "SMOTE_train.csv"))
        return self.__oversampling_df


class CreditPreprocessData(object):
    def __init__(self,
                 oversampling_data_path="../data/preprocess/SMOTE_train.csv",
                 cleaned_test_path="../data/preprocess/cleaned_test.csv",
                 oversampling_data=None):
        self.__oversampling_data_path = oversampling_data_path
        self.__cleaned_test_path = cleaned_test_path
        self.__oversampling_data = oversampling_data
        self.__df = None
        self.__df_X = None
        self.__df_y = None
        self.__df_test_X = None
        self.__scaler = None

    def __scale_train_data(self):
        self.__scaler = preprocessing.StandardScaler().fit(self.__df_X)
        self.__df_X = self.__scaler.transform(self.__df_X)

    def __scale_test_data(self):
        self.__df_test_X = self.__scaler.transform(self.__df_test_X)

    def load_oversampling_train_data(self):
        """
        加载过采样后的训练数据
        """
        if self.__oversampling_data is None:
            self.__df = pd.read_csv(self.__oversampling_data_path)
        else:
            self.__df = self.__oversampling_data

        self.__df_y = self.__df['SeriousDlqin2yrs']
        self.__df_X = self.__df.drop("SeriousDlqin2yrs", axis=1)
        self.__scale_train_data()

    def load_cleaned_test_data(self):
        df = pd.read_csv(self.__cleaned_test_path)
        self.__df_test_X = df.drop("SeriousDlqin2yrs", axis=1)
        self.__scale_test_data()

    def chi2_selected_train_data(self):
        self.__df_X = feature_selection.SelectKBest(feature_selection.chi2, k=5).fit_transform(self.__df_X, self.__df_y)

    def rfe_selected_train_data(self):
        self.__df_X = feature_selection.RFE(estimator=linear_model.LogisticRegression(solver="lbfgs"), n_features_to_select=9).fit_transform(self.__df_X, self.__df_y)

    @property
    def selected_train_feature_label_data(self):
        return self.__df_X, self.__df_y

    @property
    def selected_test_feature_data(self):
        return self.__df_test_X


if __name__ == "__main__":
    ori_credit = CreditOriData()
    ori_credit.load_train_data()
    ori_credit.load_test_data()

    cleaned_credit = CreditCleanedData(ori_credit.ori_train_data, ori_credit.ori_test_data)
    cleaned_credit.clean_data()
    cleaned_credit.draw_heatmap()

    preprocessed_credit = CreditOversamplingData(cleaned_credit.cleaned_train_data)
    preprocessed_credit.oversampling_data_by_smote()

    feature_select_credit = CreditPreprocessData()
    feature_select_credit.load_oversampling_train_data()
    feature_select_credit.load_cleaned_test_data()
    # feature_select_credit.chi2_selected_data()
    train_X, train_y = feature_select_credit.selected_train_feature_label_data
    print(np.shape(train_X), np.shape(train_y))
    test_X = feature_select_credit.selected_test_feature_data
    print(np.shape(test_X))
