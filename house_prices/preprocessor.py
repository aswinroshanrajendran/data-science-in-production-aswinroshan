import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore')


def sel_fea(train_data, num_cols, cat_cols) -> pd.DataFrame:
    features = train_data[num_cols + cat_cols]
    return features


def sel_tar(train_data, target_col) -> pd.Series:
    target = train_data[target_col]
    return target


def split_dataset(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42)
    return X_train, X_test, y_train, y_test


def scale_num_fea(dataset, numerical_cols, flag):
    if flag:
        scaler_set = scaler.fit(dataset[numerical_cols])
    scaler_set = scaler.transform(dataset[numerical_cols])
    return scaler_set


def encode_cat_fea(dataset, categorical_cols, flag):
    if flag:
        encoded_set = encoder.fit(dataset[categorical_cols])
    encoded_set = encoder.transform(dataset[categorical_cols]).toarray()
    return encoded_set


def comb_fea(dataset, numerical_cols, categorical_cols, flag):
    numerical_features = scale_num_fea(dataset, numerical_cols, flag)
    categorical_features = encode_cat_fea(dataset, categorical_cols, flag)
    processed_feature = np.hstack([numerical_features, categorical_features])
    return processed_feature


def preprocess_features(test_data, numerical_cols, categorical_cols):
    test_feature = test_data[numerical_cols + categorical_cols]
    test_num_fea = scale_num_fea(test_feature, numerical_cols, False)
    test_cat_fea = encode_cat_fea(test_feature, categorical_cols, False)
    processed_test_feature = np.hstack([test_num_fea, test_cat_fea])
    return processed_test_feature
