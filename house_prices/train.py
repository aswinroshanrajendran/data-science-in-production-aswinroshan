import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from house_prices.preprocessor import comb_fea, sel_fea, sel_tar, split_dataset

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore')


def rms(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def Target_Transform(y_train):
    y_train_transformed = np.log(y_train + 1)
    return y_train_transformed


def train_model(X_train_prepared, y_train_transformed):
    model = LinearRegression()
    model.fit(X_train_prepared, y_train_transformed)
    model_path = '../model/model.joblib'
    encoder_path = '../model/encoder.joblib'
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    return model


def model_predict(model, X_test_prepared):
    y_pred_raw = model.predict(X_test_prepared)
    y_pred = np.exp(y_pred_raw) - 1
    return y_pred


def evaluate_model(y_test, y_pred):
    rmsle_score = rms(y_test, y_pred)
    print(f'RMSLE: {rmsle_score}')


training_data_df = pd.read_csv("../data/train.csv")
num_cols = ['LotArea', 'GrLivArea']
cat_col = ['MSZoning', 'Street']
target_col = 'SalePrice'


def model_training(train_data):
    feature = sel_fea(train_data, num_cols, cat_col)
    target = sel_tar(train_data, target_col)
    X_train, X_test, y_train, y_test = split_dataset(feature, target)
    processed_feature_train = comb_fea(X_train, num_cols, cat_col, True)
    processed_feature_test = comb_fea(X_test, num_cols, cat_col, True)
    y_train_transformed = Target_Transform(y_train)
    model = train_model(processed_feature_train, y_train_transformed)
    y_pred = model_predict(model, processed_feature_test)
    evaluate_model(y_test, y_pred)
    return model, processed_feature_test, y_test


def build_model(data: pd.DataFrame) -> dict[str, str]:
    model_training(data)
    pass


model_performance_dict = build_model(training_data_df)
