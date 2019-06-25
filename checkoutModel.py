from keras.models import load_model
import numpy as np
from get_evaluate import get_evaluate
import pandas as pd
result_list=[]
fact_test =np.load('./data_preprocessing/data_deal/data_model_use/fact/data_test_fact_pad_seq_80000_400.npy')
labels_test =np.load('./data_preprocessing/data_deal/data_model_use/labels/data_test_labels_accusation.npy')

model = load_model('./model_save/FastText_No_Enhanced/FastText_epochs_10.h5')

y = model.predict(fact_test[:])
rs = get_evaluate(y_pred=y, y_true=labels_test, type="top")
result_list.append(rs.get_all_evaluate())
print(pd.DataFrame(result_list, columns=["accu", "pre_micro", "recall_micro", "f1_micro","pre_macro","recall_macro","f1_macro"]).to_string(index=True))
