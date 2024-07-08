import PyGRF
from sklearn.model_selection import train_test_split
import pandas as pd


def standarize_data(data, stats):
    return (data - stats['mean']) / stats['std']


def load_data():
    income = pd.read_csv("IncomeData/Data/income_new.csv")
    y = income['Income01']
    X_train_all, X_test_all, y_train, y_test = train_test_split(income, y, test_size=0.2, random_state=42)
    X_train = X_train_all[["UnemrT01", "PrSect01", "Foreig01"]]
    X_test = X_test_all[["UnemrT01", "PrSect01", "Foreig01"]]
    training_stat = X_train.describe().transpose()
    X_scaled_train = standarize_data(X_train, training_stat)
    X_scaled_test = standarize_data(X_test, training_stat)
    coord_train = X_train_all[['X', 'Y']]
    coords_test = X_test_all[['X', 'Y']]

    return X_scaled_train, X_scaled_test, y_train, y_test, coord_train, coords_test


def test_search_bw_lw_ISA():
    data = pd.read_csv("IncomeData/Data/income_train_X.csv")
    bandwidth, local_weight, p_value = PyGRF.search_bw_lw_ISA(data["Income01"], data[['X', 'Y']])

    assert bandwidth == 39


def test_init():
    model = PyGRF.PyGRFBuilder(max_features=1, band_width=39, train_weighted=True,
                              predict_weighted=True, bootstrap=False,
                              resampled=True, random_state=42)

    assert isinstance(model, PyGRF.PyGRFBuilder)


def test_fit():
    X_train, X_test, y_train, y_test, coord_train, coords_test = load_data()
    model = PyGRF.PyGRFBuilder(max_features=1, band_width=39, train_weighted=True,
                               predict_weighted=True, bootstrap=False,
                               resampled=True, random_state=42)
    model.fit(X_train, y_train, coord_train)

    assert model.n_estimators == 100


def test_predict():
    X_train, X_test, y_train, y_test, coord_train, coords_test = load_data()
    model = PyGRF.PyGRFBuilder(max_features=1, band_width=39, train_weighted=True,
                               predict_weighted=True, bootstrap=False,
                               resampled=True, random_state=42)
    model.fit(X_train, y_train, coord_train)
    predict_combined, predict_global, predict_local = model.predict(X_test, coords_test,
                                                                  local_weight=0.46)

    assert len(predict_combined) == len(y_test)


def test_get_local_feature_importance():
    X_train, X_test, y_train, y_test, coord_train, coords_test = load_data()
    model = PyGRF.PyGRFBuilder(max_features=1, band_width=39, train_weighted=True,
                               predict_weighted=True, bootstrap=False,
                               resampled=True, random_state=42)
    model.fit(X_train, y_train, coord_train)
    local_feature_importance = model.get_local_feature_importance()

    assert local_feature_importance.shape[0] == X_train.shape[0]
    assert local_feature_importance.shape[1] == X_train.shape[1] + 1


def test_search_bandwidth():
    X_train, X_test, y_train, y_test, coord_train, coords_test = load_data()
    search_result = PyGRF.search_bandwidth(X=X_train, y=y_train, coords=coord_train, n_estimators=100, max_features=1,
                                           random_state=42)

    df_bandwidth = search_result["bandwidth_search_result"]
    best_bandwidth = search_result["best_bandwidth"]

    assert df_bandwidth['bandwidth'].min() == max(round(X_train.shape[0] * 0.05), X_train.shape[1] + 2, 20)
    assert df_bandwidth['bandwidth'].max() == max(round(X_train.shape[0] * 0.95), X_train.shape[1] + 2)
    assert best_bandwidth == 20