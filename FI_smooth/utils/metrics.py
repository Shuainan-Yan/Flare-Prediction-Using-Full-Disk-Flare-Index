import numpy as np
from sklearn.metrics import confusion_matrix
import sys


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))* 100


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

# 计算参数
class Class_Metric(object):
    def __init__(self, y_true, y_pred):
        self.__matrix = confusion_matrix(y_true, y_pred)

    def Matrix(self):
        return self.__matrix

    # TSS=TP/TP+FN - FP/FP+TN
    def TP(self):
        tp = np.diag(self.__matrix)
        return tp.astype(float)

    def TN(self):
        tn = self.__matrix.sum() - (self.FP() + self.FN() + self.TP())
        return tn.astype(float)

    def FP(self):
        fp = self.__matrix.sum(axis=0) - np.diag(self.__matrix)
        return fp.astype(float)

    def FN(self):
        fn = self.__matrix.sum(axis=1) - np.diag(self.__matrix)
        return fn.astype(float)

    def TPRate(self):
        return self.TP() / (self.TP() + self.FN()) 

    def TNRate(self):
        return self.TN() / (self.TN() + self.FP())

    def FPRate(self):
        return 1 - self.TNRate()

    def FNRate(self):
        return 1 - self.TPRate()

    def Accuracy(self):
        ALL = self.TP() + self.FP() + self.TN() + self.FN()
        RIGHT = self.TP() + self.TN()
        return RIGHT / ALL 

    def Recall(self):
        return self.TP() / (self.TP() + self.FN() ) 

    def Precision(self):
        return self.TP() / (self.TP() + self.FP() )  

    def TSS(self):
        return self.TPRate() - self.FPRate()

    def HSS(self):
        P = self.TP() + self.FN()
        N = self.TN() + self.FP()
        up = 2 * (self.TP() * self.TN() - self.FN() * self.FP())
        below = P * (self.FN() + self.TN()) + N * (self.TP() + self.FP())
        return up / below 

    # 补充FAR
    def FAR(self):
        return self.FP() / (self.FP() + self.TP()) 