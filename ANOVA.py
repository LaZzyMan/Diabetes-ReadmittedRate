from scipy import stats
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.cross_validation import train_test_split
import csv
import math
import scipy.optimize as opt
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def get_relationship(rand_data, f_1, f_2):
    print('%s:%s' % (f_1, f_2))
    pr, pp = stats.pearsonr(rand_data[f_1], rand_data[f_2])
    sr, sp = stats.spearmanr(rand_data[f_1], rand_data[f_2])
    kr, kp = stats.kendalltau(rand_data[f_1], rand_data[f_2])
    print('pearson: r=%f, p=%f' % (pr, pp))
    print('spearman: r=%f, p=%f' % (sr, sp))
    print('kendall : r=%f, p=%f' % (kr, kp))
    return [f_1, f_2, math.fabs(pr), pp, math.fabs(sr), sp, math.fabs(kr), kp]


f_list = ['age', 'discharge_disposition_id', 'admission_source_id',
          'time_in_hospital', 'number_outpatient', 'number_emergency', 'diag1',
          'medical_specialty', 'HdA1c', 'diabetesMed']
data = pd.read_csv('./data/re/com_data.csv', encoding='gb2312')
rand_data = train_test_split(data, train_size=0.1, random_state=0)[0]
'''
formula_str = 'readmitted~'
for f_1 in f_list:
    for f_2 in f_list:
        if not f_1 == f_2:
            formula_str += '%s:%s+' % (f_1, f_2)
formula_str = formula_str[:-1]
model = ols(formula_str, rand_data).fit()
anova = anova_lm(model)
with open('./data/anova.csv', 'w', encoding='gb2312', newline='') as f:
    writter = csv.writer(f)
    writter.writerow(['f1:f2', 'F', 'PR'])
print(anova)

'''
with open('./data/feature_relationship_com.csv', 'w', encoding='gb2312', newline='') as f:
    writter = csv.writer(f)
    writter.writerow(['f_1', 'f_2', 'pr', 'pp', 'sr', 'sp', 'kr', 'kp'])
    for f_1 in rand_data.columns.tolist():
        for f_2 in rand_data.columns.tolist():
            if not f_1 == f_2:
                writter.writerow(get_relationship(rand_data, f_1, f_2))
    f.close()


