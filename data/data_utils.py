'''
The data processing of mimiciii is adapted from https://github.com/imJiawen/Warpformer.git and https://github.com/sindhura97/STraTS
'''
import os
import re
from data.preprocess_mimic_iii import *

def extract_unq_params_p12(path):
    cnt = 0
    for f in os.listdir(path):
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '.txt':
            df_temp = pd.read_csv(path + file_name + '.txt', sep=",", header=1, names=["time", "param", "value"])
            arr_data_temp = np.array(df_temp)
            params_temp = arr_data_temp[:, 1]  # extract variable names
            if cnt == 0:
                params_all = params_temp
            else:
                params_all = np.concatenate([params_all, params_temp], axis=0)
            cnt += 1
    params_all = list(params_all)
    params_all = [p for p in params_all if str(p) != 'nan']
    param_list = list(np.unique(np.array(params_all)))
    return param_list

def extract_ts_data_p12(path, param_list):
    plist = []
    allfiles = os.listdir(path)
    allfiles.sort()
    for f in allfiles:
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '.txt':
            df = pd.read_csv(path + file_name + '.txt', sep=",", header=1, names=["time", "param", "value"])
            df_demogr = df.iloc[0:5]
            df_data = df.iloc[5:]
            arr_demogr = np.array(df_demogr)
            arr_data = np.array(df_data)
            my_dict = {'id': file_name}
            my_dict['static'] = (
                arr_demogr[0, 2], arr_demogr[1, 2], arr_demogr[2, 2], arr_demogr[3, 2], arr_demogr[4, 2])
            n_pts = arr_data.shape[0]
            ts_list = []
            for i in range(n_pts):  # for each line
                param = arr_data[i, 1]  # the name of variables
                if param in param_list:
                    ts = arr_data[i, 0]  # time stamp
                    hrs, mins = float(ts[0:2]), float(ts[3:5])
                    value = arr_data[i, 2]  # value of variable
                    totalmins = 60.0 * hrs + mins
                    ts_list.append((hrs, mins, totalmins, param, value))
            my_dict['ts'] = ts_list
            plist.append(my_dict)
    return plist

def transform_p12data(plist, path):
    extended_static_list = ['Age', 'Gender=0', 'Gender=1', 'Height', 'ICUType=1', 'ICUType=2', 'ICUType=3', 'ICUType=4',
                            'Weight']
    np.save('./data/processed_data/P12/extended_static_params.npy', extended_static_list)
    df_outcomes_a = pd.read_csv(path + '/Outcomes-a.txt', sep=",", header=0,
                                names=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival",
                                       "In-hospital_death"])
    df_outcomes_b = pd.read_csv(path + '/Outcomes-b.txt', sep=",", header=0,
                                names=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival",
                                       "In-hospital_death"])
    df_outcomes_c = pd.read_csv(path + '/Outcomes-c.txt', sep=",", header=0,
                                names=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival",
                                       "In-hospital_death"])
    print(df_outcomes_a.head(n=5))
    print(df_outcomes_b.head(n=5))
    print(df_outcomes_c.head(n=5))

    arr_outcomes_a = np.array(df_outcomes_a)
    arr_outcomes_b = np.array(df_outcomes_b)
    arr_outcomes_c = np.array(df_outcomes_c)

    arr_outcomes = np.concatenate([arr_outcomes_a, arr_outcomes_b, arr_outcomes_c], axis=0)
    n = arr_outcomes.shape[0]
    print(arr_outcomes.shape)

    y_inhospdeath = arr_outcomes[:, -1]
    print("Percentage of in-hosp death: %.2f%%" % (np.sum(y_inhospdeath) / n * 100))
    print(y_inhospdeath.shape)
    tsdict_list = []
    max_tmins = 48 * 60
    max_len = 215
    ts_params = np.load('./data/processed_data/P12/ts_params.npy')
    num_params = len(ts_params)
    for ind in range(len(plist)):
        ID = plist[ind]['id']
        static = list(plist[ind]['static'])
        ts = plist[ind]['ts']
        # find unique times
        unq_tmins = []
        for sample in ts:
            current_tmin = sample[2]
            if (current_tmin not in unq_tmins) and (current_tmin < max_tmins):
                unq_tmins.append(current_tmin)
        unq_tmins = np.array(unq_tmins)

        # one-hot encoding of categorical static variables
        extended_static = [static[0],0,0,static[2],0,0,0,0,static[4]]
        if static[1]==0:
            extended_static[1] = 1
        elif static[1]==1:
            extended_static[2] = 1
        if static[3]==1:
            extended_static[4] = 1
        elif static[3]==2:
            extended_static[5] = 1
        elif static[3]==3:
            extended_static[6] = 1
        elif static[3]==4:
            extended_static[7] = 1

        # construct array of maximal size
        ts_array = np.zeros((max_len,num_params))
        mask_array = np.zeros((max_len,num_params))
        t_array = np.zeros((max_len,1))

        # for each time measurement find index and store
        for sample in ts:
            tmins = sample[2]
            param = sample[-2]
            value = sample[-1]
            if tmins < max_tmins:
                time_id  = np.where(tmins==unq_tmins)[0][0]
                param_id = np.where(ts_params==param)[0][0]
                ts_array[time_id, param_id] = value
                t_array[time_id, 0] = unq_tmins[time_id]
                mask_array[time_id, param_id] = 1

        length = len(unq_tmins)
        # construct dictionary
        ts_dict = {'id':ID, 'static':static, 'extended_static':extended_static, 'arr':ts_array, 'time':t_array, 'length':length, 'mask':mask_array, 'label':y_inhospdeath[ind]}

        tsdict_list.append(ts_dict)


    blacklist = ['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731', '150309',
                 '155655', '156254']
    i = 0
    n = len(tsdict_list)
    while i < n:
        pid = tsdict_list[i]['id']
        if pid in blacklist:
            tsdict_list = np.delete(tsdict_list, i)
            arr_outcomes = np.delete(arr_outcomes, i, axis=0)
            n -= 1
        i += 1
    np.save('./data/processed_data/P12/outcomes.npy', arr_outcomes)
    print('outcomes.npy saved')
    np.save('./data/processed_data/P12/tsdict_list.npy', tsdict_list)
    return tsdict_list

def process_p12data(path):
    param_list_a = extract_unq_params_p12(path+'/set-a/')
    param_list_b = extract_unq_params_p12(path+'/set-b/')
    param_list_c = extract_unq_params_p12(path+'/set-c/')
    param_list = param_list_a + param_list_b + param_list_c
    param_list = list(np.unique(param_list))
    static_param_list = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    np.save('./data/processed_data/P12/static_params.npy', static_param_list)
    param_list.remove("Gender")
    param_list.remove("Height")
    param_list.remove("Weight")
    param_list.remove("Age")
    param_list.remove("ICUType")

    print("Parameters: ", param_list)
    print("Number of total parameters:", len(param_list))
    np.save('./data/processed_data/P12/ts_params.npy', param_list)
    print('ts_params.npy: the names of 36 variables')

    p_list_a = extract_ts_data_p12(path+'/set-a/', param_list)
    p_list_b = extract_ts_data_p12(path+'/set-b/', param_list)
    p_list_c = extract_ts_data_p12(path+'/set-c/', param_list)
    plist = p_list_a + p_list_b + p_list_c
    print('Length of plist', len(plist))

    tsdict_list = transform_p12data(plist, path)
    return tsdict_list


def process_p19data(path):
    path_a = os.path.join(path, 'training_setA')
    path_b = os.path.join(path, 'training_setB')
    files_a = [os.path.join(path_a, f) for f in os.listdir(path_a) if f.endswith('.psv')]
    files_b = [os.path.join(path_b, f) for f in os.listdir(path_b) if f.endswith('.psv')]
    files = files_a + files_b
    p19_data = []
    arr_outcomes = []
    params_all = []
    tsdict_list = []
    for file in files:
        if params_all == []:
            df_temp = pd.read_csv(file, sep='|')
            params_all = list(df_temp.columns)
            print('params_all:', params_all)
            static_param_list = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
            np.save('./data/processed_data/P19/static_params.npy', static_param_list)
            extended_static_list = ['Age', 'Gender=0','Gender=1', 'Unit1', 'Unit2', 'HospAdmTime']
            np.save('./data/processed_data/P19/extended_static_params.npy', extended_static_list)
            params_all.remove("Gender")
            params_all.remove("Age")
            params_all.remove("Unit1")
            params_all.remove("Unit2")
            params_all.remove("HospAdmTime")
            params_all.remove("SepsisLabel")
            params_all.remove('ICULOS')
            print("Parameters: ", params_all)
            print("Number of total parameters:", len(params_all))
            np.save('./data/processed_data/P19/ts_params.npy', params_all)
        data = pd.read_csv(file, sep='|')
        sepsis_label = int(data['SepsisLabel'].max())  # 1 if sepsis occurs in any row, otherwise 0
        time = data['ICULOS']
        patient_id = int(re.search(r'p(\d+)', file).group(1))
        static = [float(data['Age'].max()), int(data['Gender'].max()), int(data['Unit1'].max(skipna=True) if not data['Unit1'].isna().all() else 0),
    int(data['Unit2'].max(skipna=True) if not data['Unit2'].isna().all() else 0), float(data['HospAdmTime'].max())]
        extended_static = [static[0], 0, 0, static[2], static[3], static[4]]
        if static[1] == 0:
            extended_static[1] = 1
        elif static[1] == 1:
            extended_static[2] = 1
        data = data.drop(columns=['SepsisLabel'])
        data = data.drop(columns=['Age'])
        data = data.drop(columns=['Gender'])
        data = data.drop(columns=['Unit1'])
        data = data.drop(columns=['Unit2'])
        data = data.drop(columns=['HospAdmTime'])
        data = data.drop(columns=['ICULOS'])
        ts_array_original = np.array(data)
        t_array_original = np.array(time)
        t_array = list(range(48 + 1))
        t_array = np.array(t_array).reshape(-1, 1)
        ts_array_temp = [0] * (48 + 1)
        for t, d in zip(t_array_original, ts_array_original):
            if t <= 48:
                ts_array_temp[t] = d
        max_length = max(len(arr) if isinstance(arr, np.ndarray) else 1 for arr in ts_array_original)
        ts_array = np.zeros((len(ts_array_temp), max_length))
        mask_array = np.zeros((len(ts_array_temp), max_length))
        for i, arr in enumerate(ts_array_temp):
            if isinstance(arr, np.ndarray):
                ts_array[i, :len(arr)] = np.nan_to_num(arr, nan=0)
                mask_array[i, :len(arr)] = ~np.isnan(arr)
            elif arr!=0:
                ts_array[i, 0] = arr
                mask_array[i, 0] = 1
            else:
                ts_array[i, 0] = 0
                mask_array[i, 0] = 0
        sample = {'data': data, 'time': time, 'patient_id': patient_id}

        if mask_array.sum(-1).sum(-1) == 0:
            print('empty')
        else:
            p19_data.append(sample)
            arr_outcomes.append([patient_id, sepsis_label])
            ts_dict = {'id': patient_id, 'static': static, 'extended_static': extended_static, 'arr': ts_array, 'time': t_array, 'length': len(t_array_original)+1, 'mask': mask_array, 'label': sepsis_label}
            tsdict_list.append(ts_dict)
    arr_outcomes = np.array(arr_outcomes)
    np.save('./data/processed_data/P19/outcomes.npy', arr_outcomes)
    np.save('./data/processed_data/P19/tsdict_list.npy', tsdict_list)
    return tsdict_list


def process_mimiciii(path):
    path = path + '/'
    if 'mimic_iii_events.csv' in os.listdir(path):
        print('MIMIC-III data already preprocessed')
    else:
        print('Preprocessing MIMIC-III data')
        prepare_mimic_data(path)

    mimi_iii_event = path + 'mimic_iii_events.csv'
    events = pd.read_csv(mimi_iii_event, usecols=['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VALUENUM', 'TABLE', 'NAME'])
    events.drop_duplicates(inplace=True)
    events = events.loc[~(events.CHARTTIME.isna() & events.VALUENUM.isna())]
    pat = pd.read_csv(path + 'mimic_iii_icu.csv')

    icu = pd.read_csv(path + 'ICUSTAYS.csv',
                      usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS'])
    icu.drop_duplicates(inplace=True)
    icu = icu.loc[icu.HADM_ID.isin(events.HADM_ID)]
    icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]
    selected_subject_ids = icu['SUBJECT_ID'].unique()

    adm = pd.read_csv(path + 'ADMISSIONS.csv',
                      usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'])
    adm.drop_duplicates(inplace=True)
    adm = adm.loc[adm.SUBJECT_ID.isin(selected_subject_ids)]


    feature_map, chart_label_dict, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict = create_map(icu, events, adm)

    # This is optional
    # remove_mod_idx = [] # keep 122 features
    remove_mod_idx = [2, 65, 91, 119, 42, 97, 120, 115, 94, 62, 105, 63, 73, 81, 87, 98, 110, 67,
                      93]  # keep 103 features
    # remove_mod_idx = [2, 66, 92, 120, 42, 98, 121, 116, 95, 63, 106, 64, 74, 82, 88, 99, 111, 68, 94]

    tmp_feature_name = sorted(feature_map.items(), key=lambda d: d[1], reverse=False)

    feature_map = {}
    feature_name = []
    new_idx = 0
    for i, j in tmp_feature_name:
        if j not in remove_mod_idx:
            feature_map[i] = new_idx
            new_idx += 1
            feature_name.append(i)
    print("got ", str(len(feature_map)), " features")

    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    adm.ADMITTIME = pd.to_datetime(adm.ADMITTIME)
    adm.DISCHTIME = pd.to_datetime(adm.DISCHTIME)
    adm.DEATHTIME = pd.to_datetime(adm.DEATHTIME)

    icu_id, mor_data, mor_label = create_mor_large(adm, events, adm2subj_dict, feature_map)
    x, m, T = preproc_xy(icu_id, mor_data, mor_label)

    static_param_list = ['GENDER', 'AGE']
    extended_static_list = ['GENDER', 'AGE']
    params_all = feature_map.keys()
    params_all = list(params_all)
    np.save('./data/processed_data/MIMICIII/static_params.npy', static_param_list)
    np.save('./data/processed_data/MIMICIII/extended_static_params.npy', extended_static_list)
    np.save('./data/processed_data/MIMICIII/ts_params.npy', params_all)
    labels = []
    samples = []
    for i in range(len(icu_id)):
        ID = icu_id[i][1]
        import numpy as np

        if ID in adm2subj_dict:
            gender_series = pat.loc[pat['SUBJECT_ID'] == adm2subj_dict[ID], 'GENDER'].map({'F': 0, 'M': 1})
            gender = gender_series.dropna().iloc[0] if not gender_series.dropna().empty else np.nan

            age_series = pat.loc[pat['SUBJECT_ID'] == adm2subj_dict[ID], 'AGE']
            age = age_series.dropna().iloc[0] if not age_series.dropna().empty else np.nan

            if not np.isnan(age):
                print(gender, age)
        else:
            gender = np.nan
            age = np.nan

        static = [gender, age]
        extended_static = [gender, age]

        ts_array = x[i].T
        mask_array = m[i].T
        t_array = T[i].T
        length = len(t_array)
        label = mor_label[i]
        ts_dict = {'id': ID, 'static': static, 'extended_static': extended_static, 'arr': ts_array, 'time': t_array, 'length': length, 'mask': mask_array, 'label': label}
        labels.append(label)
        samples.append(ts_dict)
    labels = np.array(labels)
    np.save('./data/processed_data/MIMICIII/outcomes.npy', labels)
    np.save('./data/processed_data/MIMICIII/tsdict_list.npy', samples)
    return samples





def create_map(icu, events, adm):
    chart_label_dict = {}
    icu_dict = {}
    los_dict = {}
    adm2subj_dict = {}
    adm2deathtime_dict = {}

    for _, p_row in tqdm(icu.iterrows(), total=icu.shape[0]):
        if p_row.HADM_ID not in icu_dict:
            icu_dict.update({p_row.HADM_ID: {p_row.ICUSTAY_ID: [p_row.INTIME, p_row.OUTTIME]}})
            los_dict.update({str(p_row.HADM_ID) + '_' + str(p_row.ICUSTAY_ID): p_row.LOS})

        elif p_row.ICUSTAY_ID not in icu_dict[p_row.HADM_ID]:
            icu_dict[p_row.HADM_ID].update({p_row.ICUSTAY_ID: [p_row.INTIME, p_row.OUTTIME]})
            los_dict.update({str(p_row.HADM_ID) + '_' + str(p_row.ICUSTAY_ID): p_row.LOS})

        if p_row.HADM_ID not in adm2subj_dict:
            adm2subj_dict.update({p_row.HADM_ID: p_row.SUBJECT_ID})

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        if p_row.HADM_ID not in adm2deathtime_dict:
            adm2deathtime_dict.update({p_row.HADM_ID: p_row.DEATHTIME})

    # get feature set
    feature_set = []
    feature_map = {}
    events = events.loc[~(events.CHARTTIME.isna() & events.VALUENUM.isna())]

    idx = 0
    for i in events.NAME:
        if i not in feature_set:
            feature_map[i] = idx
            idx += 1
            feature_set.append(i)

    type_dict = {}
    for i in feature_set:
        tmp_p = events.loc[events.NAME.isin([i])]
        tmp_set = set(tmp_p.TABLE)
        type_dict.update({i: tmp_set})

    idx = 0
    for k in type_dict:
        if 'chart' in type_dict[k] or 'lab' in type_dict[k]:
            if k not in chart_label_dict and k != "Mechanical Ventilation":
                chart_label_dict[k] = idx
                idx += 1

    print("got ", str(len(feature_set)), " features")
    return feature_map, chart_label_dict, icu_dict, los_dict, adm2subj_dict, adm2deathtime_dict


def create_mor_large(adm, events, adm2subj_dict, feature_map):
    mor_adm_icu_id = []
    mor_data = []
    mor_label = []

    for _, p_row in tqdm(adm.iterrows(), total=adm.shape[0]):
        adm_id = int(p_row.HADM_ID)
        p = events.loc[events.HADM_ID.isin([adm_id])]

        in_time = p.CHARTTIME.min()
        p = p.loc[(p.CHARTTIME - in_time) <= pd.Timedelta(48, 'h')]

        if p.shape[0] < 1:
            continue

        patient = [[] for _ in range(len(feature_map))]
        for _, row in p.iterrows():
            if row.NAME in feature_map:
                patient[feature_map[row.NAME]].append((row.CHARTTIME, row.VALUENUM))

        if adm_id in adm2subj_dict:
            mor_adm_icu_id.append((adm2subj_dict[adm_id], adm_id, None))
        else:
            mor_adm_icu_id.append((None, adm_id, None))
        mor_data.append(patient)
        mor_label.append(int(p_row.HOSPITAL_EXPIRE_FLAG))

    return mor_adm_icu_id, mor_data, mor_label


def preproc_xy(adm_icu_id, data_x, data_y,):
    out_value, out_timestamps = trim_los(data_x)

    x, m, T, ts_len = fix_input_format(out_value, out_timestamps)
    print("timestamps format processing success")

    return x, m, T


def fix_input_format(x, T):
    """Return the input in the proper format
    x: observed values
    M: masking, 0 indicates missing values
    delta: time points of observation
    """
    timestamp = 200
    num_features = 122

    M = np.zeros_like(x)
    # x[x > 500] = 0.0
    x[x < 0] = 0.0
    M[x > 0] = 1

    x, M, T = remove_missing_dim(x, M, T)

    x = x[:, :, :timestamp]
    M = M[:, :, :timestamp]

    delta = np.zeros((x.shape[0], 1, x.shape[-1]))

    ts_len = []
    for i in range(len(T)):
        for j in range(1, len(T[i])):
            if j >= timestamp:
                break
            delta[i, 0, j] = (T[i][j] - T[i][0]).total_seconds() / 3600.0
        ts_len.append(len(T[i]))

    return x, M, delta, ts_len


def remove_missing_dim(x, M, T):
    new_x = np.zeros((len(x), len(x[0]), len(x[0][0])))
    new_M = np.zeros((len(M), len(M[0]), len(M[0][0])))
    new_T = [[] for _ in range(len(x))]

    tmp_x = x.sum(1).squeeze()  # [B 1 L]
    for b in range(len(tmp_x)):
        new_l = 0
        for l in range(len(tmp_x[b])):
            if tmp_x[b][l] > 0:
                new_x[b, :, new_l] = x[b, :, l]
                new_M[b, :, new_l] = M[b, :, l]
                # new_T[b,:,new_l] = T[b,:,l]
                new_T[b].append(T[b][l])
                new_l += 1

    return new_x, new_M, new_T


def trim_los(data):
    """Used to build time set
    """
    num_features = len(data[0])  # final features (excluding EtCO2)
    max_length = 300  # maximum length of time stamp(48 * 60)
    a = np.zeros((len(data), num_features, max_length))
    timestamps = []

    for i in range(len(data)):

        TS = set()
        for j in range(num_features):
            for k in range(len(data[i][j])):
                TS.add(data[i][j][k][0].to_pydatetime())

        TS = list(TS)
        TS.sort()
        timestamps.append(TS)

        for j in range(len(data[i])):
            for t, v in data[i][j]:
                idx = TS.index(t.to_pydatetime())
                if idx < max_length:
                    a[i, j, idx] = v

    print("feature extraction success")
    print("value processing success ")
    return a, timestamps

