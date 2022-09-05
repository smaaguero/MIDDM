import pandas as pd
import numpy as np
import sklearn
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing

def split_patients(X, ratio=0.7, idPatient='Admissiondboid', seed=42):
    """
    This function split all the samples of the patients in train and test.
    """
    patients = np.array(X[[idPatient]].drop_duplicates())
    patients_to_train = pd.DataFrame(data=patients).sample(frac=ratio, random_state=seed).values[:, 0]
    X_train = X[X.Admissiondboid.isin(patients_to_train)]
    X_test = X[~X.Admissiondboid.isin(patients_to_train)]
    return X_train, X_test


def my_standard_scaler(
    X_train, X_test,
    non_scalable_features,
    apply_masking=False, mask_value=666
):
    """
    This function implements a standard scaler.
    """
    if apply_masking:
        X_train_norm = X_train[X_train["mask"] != mask_value]
        X_train_nonorm = X_train[X_train["mask"] == mask_value]
        X_test_norm = X_test[X_test["mask"] != mask_value]
        X_test_nonorm = X_test[X_test["mask"] == mask_value]
    else:
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
    
    # Scale in train
    scaler = preprocessing.StandardScaler()
    df_aux = X_train_norm[non_scalable_features]
    X_train_norm = X_train_norm.drop(columns=non_scalable_features)
    mapper = DataFrameMapper([(X_train_norm.columns, scaler)])
    scaled_features = mapper.fit_transform(X_train_norm.copy(), 4)
    scaled_X_train = pd.DataFrame(scaled_features, index=X_train_norm.index, columns=X_train_norm.columns)
    scaled_X_train = scaled_X_train.join(df_aux)

    # Scale in test
    df_aux = X_test_norm[non_scalable_features]
    X_test_norm = X_test_norm.drop(columns=non_scalable_features)
    scaled_features = mapper.transform(X_test_norm.copy())                                        
    scaled_X_test = pd.DataFrame(scaled_features, index=X_test_norm.index, columns=X_test_norm.columns)
    scaled_X_test = scaled_X_test.join(df_aux)

    if apply_masking:
        df_final_train = pd.concat([scaled_X_train, X_train_nonorm])
        df_final_test = pd.concat([scaled_X_test, X_test_nonorm])
        return df_final_train, df_final_test
    else:
        return scaled_X_train, scaled_X_test
    
def dataframe_to_tensor(df, y, eliminateColumn, columns, timeStepLength):
    _, idx = np.unique(df.Admissiondboid, return_index=True)
    listPatients = np.array(df.Admissiondboid)[np.sort(idx)]

    index = df.index
    y = y.reindex(index)
    y = y.drop_duplicates(subset="Admissiondboid")
    # y = y.drop(columns=["Admissiondboid"])

    for i in range(len(listPatients)):
        df_trial = df[df.Admissiondboid == listPatients[i]]
        if eliminateColumn:
            df_trial = df_trial.drop(columns=columns)
        if i == 0:
            X = np.array(df_trial)
            X = X.reshape(1, timeStepLength, df.shape[1] - len(columns))
        else:
            X_2 = np.array(df_trial)
            X_2 = X_2.reshape(1, timeStepLength, df.shape[1] - len(columns))
            X = np.append(X, X_2, axis=0)
    return X, y

def reorder_static_data(X, y):
    X_train_static = pd.merge(X, y.reset_index().Admissiondboid, how="right")
    X_train_scaled, _ = my_standard_scaler(
        X_train_static, X_train_static, 
        ['Admissiondboid',  'Origin', 'ReasonAdmission', 'PatientCategory'],
        apply_masking=False
    )
    X_train_scaled = X_train_scaled[[
        'Age', 'Gender', 'SAPSIIIScore', 'MonthOfAdmission', 'YearOfAdmission',
        'Origin', 'ReasonAdmission', 'PatientCategory'
    ]]
    return X_train_scaled