import pandas as pd
import statsmodels.api as sm
import csv
from scipy import stats
import pylab as pl
import numpy as np
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# data = pd.read_csv('./data/re/metric_data_rc.csv', encoding='gb2312')
data = pd.read_csv('./data/re/com_data.csv', encoding='gb2312')
'''
drop_list = ['num_lab_procedures', 'num_medications', 'number_diagnoses', 'gender', 'num_procedures', 'race_2', 'race_3']
for feature in drop_list:
    data.pop(feature)
'''
rand_data = train_test_split(data, train_size=0.2, random_state=1)[0]
train = data.drop('readmitted', axis=1)
logit = sm.Logit(data['readmitted'], train)
result = logit.fit()
params = result.params
pvalues = result.pvalues
print(params, pvalues)
pd.DataFrame(params, pvalues).to_csv('./data/lr_single.csv', encoding='gb2312')




