import numpy as np
import pandas as pd
from sklearn.metrics.scorer import f1_score,recall_score,precision_score

class get_evaluate():
    def __init__(self,y_pred,y_true,type):
        self.y_pred = y_pred
        self.y_true = y_true
        self.type = type
        self.allevaluate = []

    def predict2top(self):
        '''
        np.where(condition, x, y)
        满足条件(condition)，输出x，不满足输出y

        :param predictions:模型对测试集预测的结果
        :return:
        '''

        one_hots = []
        for prediction in self.y_pred:
            one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
            one_hots.append(one_hot)
        return np.array(one_hots)

    def predict2half(self):
        return np.where(self.y_pred > 0.5, 1.0, 0.0)

    def predict2both(self):
        one_hots = []
        for prediction in self.y_pred:
            one_hot = np.where(prediction > 0.5, 1.0, 0.0)
            if one_hot.sum() == 0:
                one_hot = np.where(prediction == prediction.max(), 1.0, 0.0)
            one_hots.append(one_hot)
        return np.array(one_hots)

    def get_accu(self):
        y_pred_top =  self.predict2top()
        y_pred_half = self.predict2half()
        y_pred_both = self.predict2both()
        if self.type == "top":
            accunum = [(self.y_true[i] == y_pred_top[i]).min() for i in range(len(y_pred_top))]
        elif self.type=="half":
            accunum = [(self.y_true[i] == y_pred_half[i]).min() for i in range(len(y_pred_half))]
        elif self.type=="both":
            accunum = [(self.y_true[i] == y_pred_both[i]).min() for i in range(len(y_pred_both))]
        accu = sum(accunum) / len(accunum)
        return accu
    def get_pre(self):
        y_pred_top =  self.predict2top()
        y_pred_half = self.predict2half()
        y_pred_both = self.predict2both()
        if self.type == "top":
            pre_micro = precision_score(y_pred=y_pred_top, y_true=self.y_true, pos_label=1, average='micro')
            pre_macro = precision_score(y_pred=y_pred_top, y_true=self.y_true, pos_label=1, average='macro')
        elif self.type=="half":
            pre_micro = precision_score(y_pred=y_pred_half, y_true=self.y_true, pos_label=1, average='micro')
            pre_macro = precision_score(y_pred=y_pred_half, y_true=self.y_true, pos_label=1, average='macro')
        elif self.type=="both":
            pre_micro = precision_score(y_pred=y_pred_both, y_true=self.y_true, pos_label=1, average='micro')
            pre_macro = precision_score(y_pred=y_pred_both, y_true=self.y_true, pos_label=1, average='macro')
        return (pre_micro, pre_macro)
    def get_recall(self):
        y_pred_top =  self.predict2top()
        y_pred_half = self.predict2half()
        y_pred_both = self.predict2both()
        if self.type == "top":
            recall_micro = recall_score(y_pred=y_pred_top, y_true=self.y_true, pos_label=1, average='micro')
            recall_macro = recall_score(y_pred=y_pred_top, y_true=self.y_true, pos_label=1, average='macro')
        elif self.type == "half":
            recall_micro = recall_score(y_pred=y_pred_half, y_true=self.y_true, pos_label=1, average='micro')
            recall_macro = recall_score(y_pred=y_pred_half, y_true=self.y_true, pos_label=1, average='macro')
        elif self.type == "both":
            recall_micro = recall_score(y_pred=y_pred_both, y_true=self.y_true, pos_label=1, average='micro')
            recall_macro = recall_score(y_pred=y_pred_both, y_true=self.y_true, pos_label=1, average='macro')
        return (recall_micro, recall_macro)
    def get_f1(self):
        y_pred_top =  self.predict2top()
        y_pred_half = self.predict2half()
        y_pred_both = self.predict2both()
        if self.type == "top":
            f1_micro = f1_score(y_pred=y_pred_top, y_true=self.y_true, pos_label=1, average='micro')
            f1_macro = f1_score(y_pred=y_pred_top, y_true=self.y_true, pos_label=1, average='macro')
        elif self.type == "half":
            f1_micro = f1_score(y_pred=y_pred_half, y_true=self.y_true, pos_label=1, average='micro')
            f1_macro = f1_score(y_pred=y_pred_half, y_true=self.y_true, pos_label=1, average='macro')
        elif self.type == "both":
            f1_micro = f1_score(y_pred=y_pred_both, y_true=self.y_true, pos_label=1, average='micro')
            f1_macro = f1_score(y_pred=y_pred_both, y_true=self.y_true, pos_label=1, average='macro')
        return (f1_micro, f1_macro)

    def get_all_evaluate(self):
        accu = self.get_accu()
        pre_micro,pre_macro = self.get_pre()
        recall_micro,recall_macro = self.get_recall()
        f1_micro,f1_macro = self.get_f1()
        self.allevaluate=[accu,pre_micro,recall_micro,f1_micro,pre_macro,recall_macro,f1_macro]
        return self.allevaluate