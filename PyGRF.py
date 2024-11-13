import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils import resample
from scipy.spatial import distance
import libpysal
from esda import Moran


class PyGRFBuilder:
    """
    Python implementation of geographic random forest (PyGRF).

    Parameters
    ----------
    band_width: int or float
        - The number (int) of neighbors for fitting local models if kernel is "adaptive".
        - The number (int or float) of distance within which the neighbors are used for model fitting if kernel is "fixed".
    n_estimators: int, default=100
        The number of trees in the forest.
    max_features: {“sqrt”, “log2”, None}, int or float, default=1.0
        The number of input variables to consider at each split.
        More details please refer to the documentation of scikit-learn at the link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.
    kernel: {"adaptive", "fixed"}, default="adaptive"
        The type of kernel used for determining the neighbors of a data point. Two types are available:
        - If "adaptive", a specific number of neighbors to use for fitting local models.
        - If "fixed", neighbors within a fixed distance for fitting local models.
    train_weighted: bool, default = True
        Whether samples are weighted based on distances for training local models. If False, samples are equally weighted.
    predict_weighted: bool, default = True
        Whether the ensemble of local models within the bandwidth is used and spatially weighted for producing local predictions. If False, only closest local model is used for producing local predictions.
    resampled: bool, default = True
        Whether local samples are expanded. If False, the original samples are used for fitting local models.
    n_jobs: int, default=None
        The number of jobs to execute in parallel.
        More details please refer to the documentation of scikit-learn at the link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    bootstrap: bool, default = True
        Whether each tree is built using bootstrap sampling (with replacement) from the original dataset. If False, each tree is built using the entire dataset.
        Note that this parameter should be true if out of bag (OOB) predictions are needed.
        More details please refer to the documentation of scikit-learn at the link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.
    random_state: int, instance of Numpy RandomState or None, default=None
        Determine the randomness within the model fitting. This parameter has to be fixed in order to achieve reproducibility in the model fitting process.
        More details please refer to the documentation of scikit-learn at the link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.
    **kwargs : dict, optional
        Additional parameters for fitting the random forest model. Some common parameters include:        
        - max_depth : int, default=None
            The maximum depth of the tree.

        - min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.
            
        - min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
        
        Additional parameters can be found in the scikit-learn RandomForestRegressor documentation at the link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.
    
    
    """

    def __init__(self, band_width, n_estimators=100, max_features=1.0, kernel="adaptive", train_weighted=True, predict_weighted=True,
                 resampled=True, n_jobs=None, bootstrap=True, random_state=None, **kwargs):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.band_width = band_width
        self.kernel = kernel
        self.train_weighted = train_weighted
        self.predict_weighted = predict_weighted
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self.resampled = resampled
        self.random_state = random_state
        self.global_model = None
        self.local_models = None
        self.train_data_coords = None
        self.distance_matrix = None
        self.train_data_columns = None
        self.rf_add_params = kwargs

    def fit(self, X_train, y_train, coords):
        """
        Fit PyGRF model.

        Parameters
        ----------
        X_train: data frame
            A data frame of the independent variables of training samples.
        y_train: data series
            A data series of the dependent variable of training samples.
        coords: data frame
            A data frame of the two-dimensional coordinates of training samples. It is recommended to use projected coordinates.

        Returns
        -------
        global_oob_prediction: list
            The OOB prediction from the global model
        local_oob_prediction: list
            The OOB prediction from the local models
        """

        # save the columns of the training data
        self.train_data_columns = X_train.columns.tolist()

        # fit and save the global model, and get the OOB predictions from the global model if bootstrap is True
        if self.bootstrap:
            rf_global = RandomForestRegressor(bootstrap=self.bootstrap, oob_score=True, n_estimators=self.n_estimators,
                                              max_features=self.max_features, n_jobs=self.n_jobs,
                                              random_state=self.random_state, **self.rf_add_params)
        else:
            rf_global = RandomForestRegressor(bootstrap=self.bootstrap, n_estimators=self.n_estimators,
                                              max_features=self.max_features, n_jobs=self.n_jobs,
                                              random_state=self.random_state, **self.rf_add_params)
        rf_global.fit(X_train, y_train)
        self.global_model = rf_global
        if self.bootstrap:
            global_oob_prediction = rf_global.oob_prediction_

        # create an empty list for saving local models
        self.local_models = []

        # get the distance matrix between the training samples using their geographic coordinates
        coords_array = np.array(coords, dtype=np.float64)
        self.train_data_coords = coords_array
        self.distance_matrix = distance.cdist(coords_array, coords_array, 'euclidean')

        # build a spatial weight matrix for samples based on the distances if samples are weighted spatially
        if self.train_weighted:
            if self.kernel == "adaptive":
                bandwidth_array = np.partition(self.distance_matrix, int(self.band_width) - 1, axis=1)[:,
                                     int(self.band_width) - 1] * 1.0000001
                weight_matrix = (1 - (self.distance_matrix / bandwidth_array[:, np.newaxis])**2)**2
            elif self.kernel == "fixed":
                weight_matrix = (1 - (self.distance_matrix / self.band_width)**2)**2

        # create an empty list for saving the OOB predictions from local models
        local_oob_prediction = []

        # train local models for each training samples
        for i in range(len(X_train)):
            # get the indexes of samples that are the nearest to the target sample, and only those indices within the band_width are valid
            idx = np.array([])
            distance_array = self.distance_matrix[i]
            if self.kernel == "adaptive":
                idx = np.argpartition(distance_array, self.band_width)
                idx = idx[:self.band_width]
                idx = idx[np.argsort(distance_array[idx])]
            elif self.kernel == "fixed":
                idx = np.where(distance_array < self.band_width)[0]
                idx = idx[np.argsort(distance_array[idx])]

            # get the spatial weights for samples if samples are weighted spatially
            if self.train_weighted:
                weight_array = weight_matrix[i]
                sample_weights = weight_array[idx]

            # the independent and dependent variables of samples for training local models
            local_X_train = X_train.iloc[idx]
            local_y_train = y_train.iloc[idx]

            # build a local model
            if self.bootstrap:
                rf_local = RandomForestRegressor(bootstrap=self.bootstrap, oob_score=True, n_estimators=self.n_estimators,
                                                 max_features=self.max_features, n_jobs=self.n_jobs, random_state=self.random_state, **self.rf_add_params)
            else:
                rf_local = RandomForestRegressor(bootstrap=self.bootstrap,
                                                 n_estimators=self.n_estimators,
                                                 max_features=self.max_features, n_jobs=self.n_jobs, random_state=self.random_state, **self.rf_add_params)

            # fit a local model using local trianing data, which may be expanded with replacement
            if self.train_weighted:
                if self.resampled and len(local_X_train) < 2 * self.n_estimators:
                    resampled_length = min(2 * self.n_estimators, 2*len(local_X_train)) - len(local_X_train)
                    more_X_train_resampled, more_y_train_resampled, more_sample_weights_resampled = resample(
                        local_X_train,
                        local_y_train,
                        sample_weights,
                        replace=True,
                        n_samples=resampled_length,
                        random_state=self.random_state)
                    local_X_train_resampled = pd.concat([local_X_train, more_X_train_resampled], ignore_index=True)
                    local_y_train_resampled = pd.concat([local_y_train, more_y_train_resampled], ignore_index=True)
                    sample_weights_resampled = np.concatenate((sample_weights, more_sample_weights_resampled))
                    rf_local.fit(local_X_train_resampled, local_y_train_resampled, sample_weights_resampled)
                else:
                    rf_local.fit(local_X_train, local_y_train, sample_weights)
            else:
                if self.resampled and len(local_X_train) < 2 * self.n_estimators:
                    resampled_length = min(2 * self.n_estimators, 2*len(local_X_train)) - len(local_X_train)
                    more_X_train_resampled, more_y_train_resampled = resample(local_X_train,
                                                                              local_y_train,
                                                                              replace=True,
                                                                              n_samples=resampled_length,
                                                                              random_state=self.random_state)
                    local_X_train_resampled = pd.concat([local_X_train, more_X_train_resampled], ignore_index=True)
                    local_y_train_resampled = pd.concat([local_y_train, more_y_train_resampled], ignore_index=True)
                    rf_local.fit(local_X_train_resampled, local_y_train_resampled)
                else:
                    rf_local.fit(local_X_train, local_y_train)

            # get the local OOB prediction for the current sample
            if self.bootstrap:
                local_oob_prediction.append(rf_local.oob_prediction_[0])

            # save a local model in a dictionary
            self.local_models.append(rf_local)

        if self.bootstrap:
            return global_oob_prediction, local_oob_prediction

    def predict(self, X_test, coords_test, local_weight):
        """
        Make predictions for test data using fitted model

        Parameters
        ----------
        X_test: data frame
            A data frame of the independent variables of test samples.
        coords_test: data frame
            A data frame of the two-dimensional coordinates of test samples. It is recommended to use projected coordinates.
        local_weight: float
            A number for combining global and local predictions

        Returns
        -------
        predict_combined: list
            A list of predictions combined from global and local predictions.
        predict_global: list
            A list of global predictions.
        predict_local: list
            A list of local predictions.
        """

        # make predictions using the global RF model
        predict_global = self.global_model.predict(X_test).flatten()

        # build a matrix of local predictions derived from local RF models
        local_predict_list = []
        for local_model in self.local_models:
            locl_predict_one = local_model.predict(X_test)
            local_predict_list.append(locl_predict_one[:, np.newaxis])
        local_predict_matrix = np.concatenate(local_predict_list, axis=1)

        # get the distance matrix between test samples and training samples
        coords_test_array = np.array(coords_test, dtype=np.float64)
        distance_matrix_test_to_train = distance.cdist(coords_test_array, self.train_data_coords, 'euclidean')

        # build a spatial weight matrix based on distances between local models and test samples
        if self.predict_weighted:
            if self.kernel == "adaptive":
                bandwidth_array = np.partition(distance_matrix_test_to_train, int(self.band_width) - 1, axis=1)[:,
                                     int(self.band_width) - 1] * 1.0000001
                weight_matrix = (1 - (distance_matrix_test_to_train / bandwidth_array[:, np.newaxis])**2)**2
            elif self.kernel == "fixed":
                weight_matrix = (1 - (distance_matrix_test_to_train / self.band_width)**2)**2

        # create an empty list for saving local predictions
        predict_local = []

        # make predictions using local models
        for i in range(len(X_test)):
            this_local_prediction = 0
            distance_array = distance_matrix_test_to_train[i]
            local_predict_array = local_predict_matrix[i]
            # derive local prediction by spatially weighting the predictions from local models within the band_width
            if self.predict_weighted:
                # get the indexes of local models within the band_width and
                idx = np.array([])
                if self.kernel == "adaptive":
                    idx = np.argpartition(distance_array, self.band_width)
                    idx = idx[:self.band_width]
                    idx = idx[np.argsort(distance_array[idx])]
                elif self.kernel == "fixed":
                    idx = np.where(distance_array < self.band_width)[0]
                    idx = idx[np.argsort(distance_array[idx])]
                weight_array = weight_matrix[i]
                sample_weights = weight_array[idx]

                # compute the spatially weighted sum of local predictions within the band_width
                local_prediction_bandwidth = local_predict_array[idx]
                this_local_prediction = np.sum(local_prediction_bandwidth * sample_weights)
                this_local_prediction = this_local_prediction * 1.0 / np.sum(sample_weights)
            # derive local prediction using only the nearest local model
            else:
                idx = np.argpartition(distance_array, 1)
                this_idx = idx[0]
                this_local_prediction = local_predict_array[this_idx]

            predict_local.append(this_local_prediction)

        # combine global and local predictions
        predict_local_array = np.array(predict_local)
        predict_global_array = np.array(predict_global)
        predict_combined = (predict_local_array * local_weight + predict_global_array * (1 - local_weight)).tolist()

        return predict_combined, predict_global, predict_local

    def get_local_feature_importance(self):
        """
        Get the local feature importance based on local models

        Returns:
        -------
        feature_importance_df: data frame
            A data frame containing all the feature importance from local models.
        """

        if self.local_models == None:
            print("The model has not been trained yet...")
            return None

        # create an empty data frame for saving the local feature importances
        column_list = ["model_index"] + self.train_data_columns
        feature_importance_df = pd.DataFrame(columns=column_list)

        # Extract the feature importances from local models
        for i in range(len(self.local_models)):
            this_row = [i]
            this_row.extend(self.local_models[i].feature_importances_)
            feature_importance_df = pd.concat([feature_importance_df, pd.DataFrame([this_row], columns=column_list)], ignore_index=True)

        return feature_importance_df


def search_bw_lw_ISA(y, coords, bw_min=None, bw_max=None, step=1):
    """
    Search for bandwidth and local model weight using incremental spatial autocorrelation (ISA)

    Parameters
    ----------
    y: data series
        A data series of dependent variable of samples.
    coords: data frame
        A data frame of two-dimentional coordinates of samples. It is recommended to use projected coordinates.
    bw_min: int, default = None
        The minimum band_width for searching.
    bw_max: int, default = None
        The maximum band_width for searching.
    step: int, default = 1
        The step for iterating the band_width between minimum and maximum values.

    Returns
    -------
    found_bandwidth: int
        The found bandwidth using ISA.
    found_moran_I: float
        The Moran's I corresponding to the found bandwidth.
    found_p_value: float
        The p-value corresponding to the Moran's I.
    """

    # compute the default values for bw_min and bw_max
    if bw_min is None:
        bw_min = 1
    if bw_max is None:
        bw_max = len(y)

    # build the k-d tree using spatial coordinates of data records
    coords_list = [tuple(row) for row in coords.to_numpy()]
    kd = libpysal.cg.KDTree(np.array(coords_list))

    # create lists for saving the ISA result
    bandwidth_list, moran_I_list, z_score_list, p_value_list = [], [], [], []

    # compute the moran's I, z-score, and p_value using the sequence of bandwidths
    for current_bw in range(bw_min, bw_max, step):
        kw = libpysal.weights.KNN(kd, current_bw)
        moran_I = Moran(y, kw)
        bandwidth_list.append(current_bw)
        moran_I_list.append(moran_I.I)
        z_score_list.append(moran_I.z_norm)
        p_value_list.append(moran_I.p_norm)

    # search the global peak with p-value smaller than 0.05
    max_index = None
    max_zscore = float('-inf')
    for i in range(len(z_score_list)):
        if z_score_list[i] > max_zscore and p_value_list[i] < 0.05:
            max_zscore = z_score_list[i]
            max_index = i

    found_bandwidth, found_moran_I, found_p_value = bandwidth_list[max_index], moran_I_list[max_index], p_value_list[max_index]
    print("bandwidth: {}, moran's I: {}, p-value: {}".format(found_bandwidth, found_moran_I, found_p_value))

    return found_bandwidth, found_moran_I, found_p_value


def search_bandwidth(X, y, coords, n_estimators, max_features, bw_min=None, bw_max=None, step=1, train_weighted=True, resampled=True, n_jobs=None,
                     random_state=None):
    """
    Optimize the bandwidth using OOB score

    Parameters
    ----------
    X: data frame
        A data frame of independent variables of samples in the data used for searching the optimal bandwidth.
    y: data series
        A data series of dependent variable of samples.
    coords: data frame
        A data frame of two-dimentional coordinates of samples. It is recommended to use projected coordinates.
    n_estimators: int
        The number of trees for the PyGRF model.
    max_features: {“sqrt”, “log2”, None}, int or float
        The number of input variables to consider at each split.
    bw_min: int, default = None
        The minimum band_width for searching.
    bw_max: int, default = None
        The maximum band_width for searching.
    step: int, default = 1
        The step for iterating the band_width between minimum and maximum values.
    train_weighted: bool, default = True
        Whether samples are weighted based on distances in the PyGRF model.
    resampled: bool, default = True
        Whether local samples are expanded in the PyGRF model.
    n_jobs: int, default=None
        The number of jobs to execute in parallel.
    random_state: int, instance of Numpy RandomState or None, default=None
        Determine the randomness within the PyGRF model fitting.

    Returns
    -------
    search_result: dictionary
        The result of searching the optimal band_width
    """

    # compute the default values for bw_min and bw_max
    records_num = X.shape[0]
    variables_num = X.shape[1]
    if bw_min is None:
        bw_min = max(round(records_num * 0.05), variables_num + 2, 20)
    if bw_max is None:
        bw_max = max(round(records_num * 0.95), variables_num + 2)

    # create lists and a data frame for saving the band_width searching result
    band_width_list = []
    # local_list = []
    mixed_list = []
    # low_list = []
    df_search_bw = pd.DataFrame(columns=['bandwidth', 'mixed'])

    # iterate each band_width between minimum and maximum values
    for current_bw in range(bw_min, bw_max + 1, step):
        band_width_list.append(current_bw)

        # fit PyGRF model using the test bandwidth and get the OOB predictions
        grf = PyGRFBuilder(n_estimators=n_estimators, max_features=max_features, band_width=current_bw, random_state=random_state,
                    train_weighted=train_weighted)
        y_oob_local, y_oob_global = grf.fit(X, y, coords)

        # compute R-squred scores using local OOB predictions and global OOB predictions
        # r_oob_local = r2_score(y, y_oob_local)
        # local_list.append(r_oob_local)
        y_oob_mixed = (np.array(y_oob_local) + np.array(y_oob_global)) / 2
        r_oob_mixed = r2_score(y, y_oob_mixed)
        mixed_list.append(r_oob_mixed)
        # y_oob_low = 0.25 * np.array(y_oob_local) + 0.75 * np.array(y_oob_global)
        # r_oob_low = r2_score(y, y_oob_low)
        # low_list.append(r_oob_low)

        print("bandwidth: " + str(current_bw), "mixed: " + str(round(r_oob_mixed, 4)))

    # get the optimal band_width searching result
    df_search_bw["bandwidth"] = band_width_list
    # df_search_bw["local"] = local_list
    df_search_bw["mixed"] = mixed_list
    # df_search_bw["low_local"] = low_list
    optimal_local_row = df_search_bw.loc[df_search_bw['mixed'].idxmax()]
    optimal_bandwidth = optimal_local_row['bandwidth']
    print("Best Bandwidth: ", optimal_bandwidth)
    search_result = {'bandwidth_search_result': df_search_bw, 'best_bandwidth': optimal_bandwidth}

    return search_result
