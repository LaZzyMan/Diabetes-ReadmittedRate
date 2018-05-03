import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt

data = pd.read_csv('./data/re/com_data.csv', encoding='gb2312')
'''
drop_list_1 = ['gender', 'num_procedures', 'race_1', 'race_3']
drop_list = ['num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
             'number_emergency', 'number_inpatient', 'number_diagnoses']
for feature in drop_list_1:
    data.pop(feature)
'''
train, test = train_test_split(data, test_size=0.3, random_state=0)
train_x = train.drop('readmitted', axis=1)
train_y = train['readmitted']
test_x = test.drop('readmitted', axis=1)
test_y = test['readmitted']
classifier = LogisticRegression(penalty='l2', C=100, random_state=0, max_iter=500, n_jobs=-1)
classifier.fit(train_x, train_y)

scores = cross_val_score(classifier, train_x, train_y, cv=5)
print('准确率：', np.mean(scores), scores)
precisions = cross_val_score(classifier, train_x, train_y, cv=5, scoring='precision')
print('精确率：', np.mean(precisions), precisions)
recalls = cross_val_score(classifier, train_x, train_y, cv=5, scoring='recall')
print('召回率：', np.mean(recalls), recalls)
f1s = cross_val_score(classifier, train_x, train_y, cv=5, scoring='f1')
print('f1：', np.mean(f1s), f1s)

predictions = classifier.predict_proba(test_x)
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
