from keras.models import load_model
import numpy as np
from get_evaluate import get_evaluate
import pandas as pd
from keras import layers
from Capsule_Keras import Capsule

num = 3
result_list=[]
#data need to test

# fact_test =np.load('./multi_label/multi_label_data/fact/'+str(num)+'_num_index_test_data.npy')
# labels_test =np.load('./multi_label/multi_label_data/label/'+str(num)+'_num_index_test_label.npy')

fact_test =np.load('./data_preprocessing/data_deal/data_model_use/fact/data_test_fact_pad_seq_80000_400.npy')
labels_test =np.load('./data_preprocessing/data_deal/data_model_use/labels/data_test_labels_accusation.npy')

# fact_test =np.load('./one_label/one_label_data/fact/8_强迫他人吸毒_data.npy')
# labels_test =np.load('./one_label/one_label_data/label/8_强迫他人吸毒_label.npy')


model = load_model('./model_save/Capsule_No_Enhanced/Capsule_epochs_17.h5', custom_objects={'Capsule': Capsule})
# model = load_model('./model_save/FastText_Enhanced/FastText_epochs_30.h5')

# model = load_model('./model_save/CNN_Enhanced/CNN_epochs_20.h5')
y = model.predict(fact_test[:])


rs = get_evaluate(y_pred=y, y_true=labels_test, type="both")
result_list.append(rs.get_all_evaluate())
print(pd.DataFrame(result_list, columns=["accu", "pre_micro", "recall_micro", "f1_micro","pre_macro","recall_macro","f1_macro"]).to_string(index=True))
