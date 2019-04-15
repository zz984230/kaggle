from sklearn import linear_model, metrics
from dl_group.credit_score.code.credit_data import CreditPreprocessData
from dl_group.credit_score.code.ann_model import ANN
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, svm, ensemble
from xgboost import XGBClassifier
import pandas as pd


class ClassifiedModel(object):
    def __init__(self,
                 ori_X,
                 ori_y,
                 test_X=None,
                 cls_name="logicalRegression",
                 saved_predict_path="../data/result/result.csv"):
        self.__ori_X = ori_X
        self.__ori_y = ori_y
        self.__train_X = None
        self.__train_y = None
        self.__val_X = None
        self.__val_y = None
        self.__test_X = test_X
        self.__cls_name = cls_name
        self.__saved_predict_path = saved_predict_path
        self.__cls = None
        self.__predict_result = None

    def __simple_split_train_data(self):
        """
        划分训练集
        """
        self.__train_X, self.__val_X, self.__train_y, self.__val_y = model_selection.train_test_split(self.__ori_X, self.__ori_y, test_size=0.3, random_state=0)

    def __logical_regression_model(self):
        self.__cls = linear_model.LogisticRegression().fit(self.__ori_X, self.__ori_y)

    def __svc_model(self):
        self.__cls = svm.SVC().fit(self.__ori_X, self.__ori_y)

    def __xgboost_model(self):
        self.__cls = XGBClassifier(learning_rate=0.2, reg_alpha=1, subsample=0.9)

        # score = model_selection.cross_val_score(self.__cls, self.__train_X, self.__train_y, cv=5, scoring="roc_auc")
        # print(score)
        # pred_y = model_selection.cross_val_predict(self.__cls, self.__val_X, self.__val_y, cv=5)
        # acc = metrics.accuracy_score(self.__val_y, pred_y)
        # print("accuracy: %f" % acc)

        self.__cls.fit(self.__ori_X, self.__ori_y)

    def __random_forest_model(self):
        self.__cls = ensemble.RandomForestClassifier(n_estimators=100)
        self.__cls.fit(self.__ori_X, self.__ori_y)

    def __ann_model(self):
        self.__cls = ANN(epoch=30)
        # self.__cls.fit(self.__ori_X, self.__ori_y)
        self.__cls.fit_by_saved_model(self.__ori_X, self.__ori_y)

    def train(self):
        self.__simple_split_train_data()
        if self.__cls_name == "logicalRegression":
            self.__logical_regression_model()
        elif self.__cls_name == "svm":
            self.__svc_model()
        elif self.__cls_name == "xgboost":
            self.__xgboost_model()
        elif self.__cls_name == "randomForest":
            self.__random_forest_model()
        elif self.__cls_name == "ANN":
            self.__ann_model()

    def predict(self):
        self.__predict_result = self.__cls.predict_proba(self.__test_X)

    def save_predict_result(self):
        l = np.reshape([i + 1 for i in range(np.shape(self.__predict_result)[0])], [-1, 1])
        result = np.hstack((np.array(l, dtype=np.int32), np.reshape(self.__predict_result[:, 1], [-1, 1])))
        r = pd.DataFrame(result)
        r.columns = ["Id", "Probability"]
        r["Id"] = r["Id"].astype(int)
        r.to_csv(self.__saved_predict_path, index=False)


if __name__ == "__main__":
    feature_select_credit = CreditPreprocessData("../data/preprocess/SMOTE_train.csv")
    feature_select_credit.load_oversampling_train_data()
    feature_select_credit.load_cleaned_test_data()
    # feature_select_credit.xgboost_selected_data(is_plot=True)
    train_X, train_y = feature_select_credit.selected_train_feature_label_data
    test_X = feature_select_credit.selected_test_feature_data

    cls_model = ClassifiedModel(train_X, train_y, test_X=test_X, cls_name="ANN")
    cls_model.train()
    cls_model.predict()
    cls_model.save_predict_result()
