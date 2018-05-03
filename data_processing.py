import pandas as pd
import json
import csv
import math
from sklearn.preprocessing import MinMaxScaler


def create_dummy(df, col_name):
    dummy = pd.get_dummies(df[col_name], prefix=col_name)
    df = df.drop(col_name, axis=1).join(dummy)
    return df


def export_numerical_data(filename):
    i = 0
    race = {'AfricanAmerican': 1, 'Asian': 2, 'Caucasian': 3,
            'Hispanic': 4, 'Other': 0}
    gender = {'Female': 1, 'Male': 0, 'Unknown/Invalid': -1}
    age = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2,
           '[30-40)': 3, '[40-50)': 4, '[50-60)': 5,
           '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
    diag = {'呼吸系统疾病': 2, '肌肉骨骼系统及结缔组织疾病': 7, '泌尿生殖系统疾病': 6, '受伤及中毒': 5,
            '糖尿病': 4, '消化系统疾病': 3, '循环系统疾病': 1, '肿瘤': 8, '其他': 0}
    diag_other = ['传染病和寄生虫疾病', '感觉器官疾病', '精神失常', '内分泌、营养、新陈代谢及免疫系统疾病', '皮肤及皮下组织疾病',
                  '妊娠、分娩和产后合并症', '神经系统疾病', '外伤及补充分类', '先天畸形', '血液及造血器官疾病', '症候、征候及不明情况']
    medical_specialty = {}
    max_glu_serum = {'None': -1, 'Norm': 0, '>200': 1, '>300': 2}
    A1Cresult = {'None': -1, 'Norm': 0, '>7': 1, '>8': 2}
    dosage = {'No': -1, 'Down': 0, 'Steady': 1, 'Up': 2}
    change = {'Ch': 1, 'No': 0}
    diabetesMed = {'Yes': 1, 'No': 0}
    readmitted = {'NO': 0, '<30': 1, '>30': 2}
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace(['?', '其他疾病'], [-1, -1])
    data = data.drop_duplicates('patient_nbr')
    # 生成分类统计表
    with open('./data/classification_data.csv', 'w', newline='') as f:
        writter = csv.writer(f)
        writter.writerow(['类别', '描述', '人数', '占比'])
        for key, value in diag.items():
            if value == 0:
                continue
            num = data.loc[data['diag1'] == key].shape[0]
            percent = '%.2f%%' % (num / data.shape[0] * 100)
            writter.writerow([value, key, num, percent])
        for item in diag_other:
            num = data.loc[data['diag1'] == item].shape[0]
            percent = '%.2f%%' % (num / data.shape[0] * 100)
            writter.writerow([0, item, num, percent])
        f.close()
    data['race'] = data['race'].replace(race)
    data['gender'] = data['gender'].replace(gender)
    data['age'] = data['age'].replace(age)
    data['admission_type_id'] = data['admission_type_id'].replace([5, 6, 8], [-1, -1, 0])
    data['discharge_disposition_id'] = data['discharge_disposition_id'].replace([18, 26, 25], [-1, -1, 0])
    data['admission_source_id'] = data['admission_source_id'].replace([9, 15, 17, 21, 20], [-1, -1, -1, -1, 0])
    for item in data['medical_specialty'].unique():
        if not item == -1:
            medical_specialty[item] = i
            i += 1
    data['medical_specialty'] = data['medical_specialty'].replace(medical_specialty)
    data['diag1'] = data['diag1'].replace(diag)
    data['diag1'] = data['diag1'].replace(diag_other, 0)
    data['diag2'] = data['diag2'].replace(diag)
    data['diag2'] = data['diag2'].replace(diag_other, 0)
    data['diag3'] = data['diag3'].replace(diag)
    data['diag3'] = data['diag3'].replace(diag_other, 0)
    data['max_glu_serum'] = data['max_glu_serum'].replace(max_glu_serum)
    data['A1Cresult'] = data['A1Cresult'].replace(A1Cresult)
    data['change'] = data['change'].replace(change)
    data['diabetesMed'] = data['diabetesMed'].replace(diabetesMed)
    data['readmitted'] = data['readmitted'].replace(readmitted)
    data = data.replace(dosage)
    data.to_csv('./data/numerical_data.csv')
    mapping = json.dumps({'race': race, 'gender': gender, 'age': age,
                          'diag': diag, 'medical_specialty': medical_specialty,
                          'max_glu_serum': max_glu_serum, 'A1Cresult': A1Cresult,
                          'dosage': dosage, 'change': change, 'diabetesMed': diabetesMed,
                          'readmitted': readmitted})
    with open('./data/mapping.json', 'w') as f:
        f.write(mapping)
        f.close()


def export_metric_data(filename):
    data = pd.read_csv(filename, encoding='gb2312')
    mapping = json.loads(open('./data/mapping.json').read())
    with open('./data/IDs_mapping.csv', 'r', encoding='gb2312') as f:
        i = 0
        reader = csv.reader(f)
        for line in list(reader):
            i += 1
            line[1].replace(' ', '_')
            if i in range(2, 10):
                if not data.loc[data['admission_type_id'] == int(line[0])].shape[0] == 0:
                    data['admission_type_' + line[1]] = data['admission_type_id'].apply(lambda x: 1 if x == int(line[0]) else 0)
            if i in range(12, 42):
                if not data.loc[data['discharge_disposition_id'] == int(line[0])].shape[0] == 0:
                    data['discharge_disposition_' + line[1]] = data['discharge_disposition_id'].apply(lambda x: 1 if x == int(line[0]) else 0)
            if i in range(44, 69):
                if not data.loc[data['admission_source_id'] == int(line[0])].shape[0] == 0:
                    data['admission_source_' + line[1]] = data['admission_source_id'].apply(lambda x: 1 if x == int(line[0]) else 0)
        f.close()
    for key, value in mapping['medical_specialty'].items():
        data['medical_specialty_' + key] = data['medical_specialty'].apply(lambda x: 1 if x == int(value) else 0)
    with open('./data/classification_data.csv', 'r', encoding='gb2312') as f:
        reader = csv.reader(f)
        for row in list(reader)[1: 10]:
            data['diag1_' + row[1]] = data['diag1'].apply(lambda x: 1 if x == int(row[0]) else 0)
            data['diag2_' + row[1]] = data['diag2'].apply(lambda x: 1 if x == int(row[0]) else 0)
            data['diag3_' + row[1]] = data['diag3'].apply(lambda x: 1 if x == int(row[0]) else 0)
        f.close()
    for item in ['diag1', 'diag2', 'diag3', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'medical_specialty']:
        data.pop(item)
    data.to_csv('./data/metric_data.csv')


def export_prob_distribution(filename, feature):
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace({'readmitted': 2}, 1)
    f_series = data[feature].drop_duplicates()
    f_series = f_series.sort_values()
    f_df = pd.DataFrame(f_series.reset_index(drop=True))
    f_df['num'] = 0
    f_df['prob'] = 0.0
    for i in range(f_df.shape[0]):
        tmp = data.loc[data[feature] == f_df[feature][i]]
        f_df['num'][i] = tmp.shape[0]
        f_df['prob'][i] = tmp.shape[0] / data.shape[0]
    f_df.to_csv('./data/prob_distributions/prob_%s.csv' % feature)


def export_confidence_interval(filename, feature):
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace({'readmitted': 2}, 1)
    diag_series = data[feature].drop_duplicates()
    diag = pd.DataFrame(diag_series.reset_index(drop=True))
    diag['readmitted_rate'] = 0.0
    diag['confidence_interval_95'] = ' '
    for i in range(diag.shape[0]):
        tmp = data.loc[data[feature] == diag[feature][i]]
        s = tmp['readmitted'].std()
        mean = tmp['readmitted'].mean()
        upper = mean + 1.96 * s / math.sqrt(tmp.shape[0])
        lower = mean - 1.96 * s / math.sqrt(tmp.shape[0])
        diag['readmitted_rate'][i] = mean
        diag['confidence_interval_95'][i] = '(%.2f%%, %.2f%%)' % (upper, lower)
    diag.to_csv('./data/confidence/confidence_interval_%s.csv' % feature)


def export_numerical_data_2(filename):
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace({'readmitted': 2}, 1)
    data = data.replace({'admission_type_id': -1}, 1)
    data = data.replace({'diag1': -1}, 0)
    data = data.replace({'gender': -1}, 0)
    data = data[['race', 'gender', 'age', 'discharge_disposition_id', 'admission_source_id',
                 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
                 'number_emergency', 'number_inpatient', 'diag1', 'number_diagnoses', 'medical_specialty',
                 'A1Cresult', 'change', 'diabetesMed', 'readmitted']]
    reclassification_list = ['race', 'age', 'discharge_disposition_id', 'admission_source_id', 'medical_specialty']
    for feature in reclassification_list:
        mapping = pd.read_csv(('./data/map/prob_%s.csv' % feature), encoding='gb2312')
        data[feature] = data[feature].replace(mapping[feature].tolist(), mapping['recollection'].tolist())
    data['HdA1c'] = 0
    for i in range(data.shape[0]):
        hda1c = data['A1Cresult'][i]
        change = data['change'][i]
        if hda1c == -1:
            continue
        if hda1c == 2:
            if change == 0:
                data['HdA1c'][i] = 2
            else:
                data['HdA1c'][i] = 3
        else:
            data['HdA1c'][i] = 1
    data.pop('A1Cresult')
    data.pop('change')
    data.to_csv('./data/re/numerical_data_rc.csv')


def export_metric_data_dt(filename):
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace({'discharge_disposition_id': 2}, 1)
    data = data.replace({'discharge_disposition_id': 4}, 2)
    data = data.replace({'discharge_disposition_id': 5}, 2)
    data = data.replace({'discharge_disposition_id': 6}, 3)
    data.pop('race')
    data.pop('gender')
    data.pop('num_medications')
    data.pop('medical_specialty')
    data.pop('num_lab_procedures')
    data.pop('number_diagnoses')
    data.pop('num_procedures')
    data.pop('number_inpatient')
    data = create_dummy(data, 'age')
    data = create_dummy(data, 'discharge_disposition_id')
    data = create_dummy(data, 'admission_source_id')
    data = create_dummy(data, 'diag1')
    data = create_dummy(data, 'HdA1c')
    num_list = ['time_in_hospital', 'number_outpatient', 'number_emergency', ]
    for feature in num_list:
        data[feature] = MinMaxScaler().fit_transform(data[feature])
    data.to_csv('./data/re/metric_data_dt.csv')


def export_metric_data_2(filename):
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace({'discharge_disposition_id': 2}, 1)
    data = data.replace({'discharge_disposition_id': 4}, 2)
    data = data.replace({'discharge_disposition_id': 5}, 2)
    data = data.replace({'discharge_disposition_id': 6}, 3)
    data = create_dummy(data, 'race')
    data = create_dummy(data, 'age')
    data = create_dummy(data, 'discharge_disposition_id')
    data = create_dummy(data, 'admission_source_id')
    data = create_dummy(data, 'diag1')
    data = create_dummy(data, 'medical_specialty')
    data = create_dummy(data, 'HdA1c')
    data['intercept'] = 1.0
    num_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
                'number_emergency', 'number_inpatient', 'number_diagnoses']
    for feature in num_list:
        data[feature] = MinMaxScaler().fit_transform(data[feature])
    data.to_csv('./data/re/metric_data_rc.csv')


def merge(x, f1, f2):
    if x == f1 and y == f2:
        return 1
    else:
        return 0


def export_combine_data(filename):
    data = pd.read_csv(filename, encoding='gb2312')
    data = data.replace({'discharge_disposition_id': 2}, 1)
    data = data.replace({'discharge_disposition_id': 4}, 2)
    data = data.replace({'discharge_disposition_id': 5}, 2)
    data = data.replace({'discharge_disposition_id': 6}, 3)
    num_list = ['time_in_hospital', 'number_outpatient', 'number_emergency']

    for feature in num_list:
        data[feature] = MinMaxScaler().fit_transform(data[feature])

    com_list = ['age:admission_source_id', 'age:diabetesMed', 'age:discharge_disposition_id',
                'diag1:HdA1c', 'age:HdA1c', 'diabetesMed:HdA1c']
    result = data[num_list]
    result['readmitted'] = data['readmitted']
    for feature in com_list:
        f_1, f_2 = feature.split(':')
        count = 0
        for i in data[f_1].unique().tolist():
            for j in data[f_2].unique().tolist():
                if count == 0:
                    count += 1
                    continue
                name = '%s_%s_%s_%s' % (f_1, i, f_2, j)
                result[name] = data.apply(lambda x: int((x[f_1] == i) and (x[f_2] == j)), axis=1)
                print(name)
                count += 1
    result['intercept'] = 1.0
    result.to_csv('./data/re/com_data.csv', encoding='gb2312')


if __name__ == '__main__':
    # export_numerical_data_2('./data/numerical_data.csv')
    # export_metric_data_2('./data/re/numerical_data_rc_4.csv')
    # export_numerical_data('./data/newdata.csv')
    # export_metric_data('./data/numerical_data.csv')
    # export_combine_data('./data/re/numerical_data_rc.csv')
    export_metric_data_dt('./data/re/numerical_data_rc.csv')
    '''
    feature_list = ['race', 'gender', 'age', 'admission_source_id',
                    'discharge_disposition_id', 'time_in_hospital', 'medical_specialty',
                    'diag1', 'HdA1c', 'diabetesMed']

    for feature in ['A1Cresult', 'change']:
        export_confidence_interval('./data/numerical_data.csv', feature)
        print(feature)
    '''
    '''
    for feature in feature_list:
        export_prob_distribution('./data/numerical_data.csv', feature)
        print(feature)
    '''
