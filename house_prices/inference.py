import joblib
import numpy as np
from house_prices import CATEGORICAL, MODEL_PATH, NUMERICAL
from house_prices.preprocessor import encode_cat_fea, scale_num_fea, sel_fea


def process_data(test_data, numerical_cols, categorical_cols):
    test_fea = sel_fea(test_data, numerical_cols, categorical_cols)
    test_num_fea = scale_num_fea(test_fea, numerical_cols, False)
    test_cat_fea = encode_cat_fea(test_fea, categorical_cols, False)
    pros_test_fea = np.hstack([test_num_fea, test_cat_fea])
    return pros_test_fea


def make_predictions(test_data):
    loaded_model = joblib.load(MODEL_PATH)
    processed_feature = process_data(test_data, NUMERICAL, CATEGORICAL)
    acutal_prediction = loaded_model.predict(processed_feature)
    return acutal_prediction
