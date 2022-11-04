"""Wrapper for https://github.com/sebp/scikit-survival + https://github.com/garethjns/IncrementalTrees"""

import time
from typing import Union, Optional

# third party imports
import multiprocessing
import numpy as np
import pandas as pd
import warnings

from joblib import Parallel, delayed
from lifelines import NelsonAalenFitter
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from random_survival_forest.splitting import _find_split
from incremental_trees.add_ins.classifier_additions import ClassifierAdditions
from random_survival_forest.scoring import concordance_index


class RandomSurvivalForest:

    def __init__(self, n_estimators: int = 100, min_leaf: int = 3, unique_deaths: int = 3, n_jobs: int or None = None,
                 oob_score: bool = False, timeline=None, random_state=None):
        """
        A Random Survival Forest is a prediction model especially designed for survival analysis.
        :param n_estimators: The numbers of trees in the forest.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param n_jobs: The number of jobs to run in parallel for fit. None means 1.
        """
        self.n_estimators = n_estimators
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.n_jobs = n_jobs
        self.bootstrap_idxs = None
        self.bootstraps = []
        self.oob_idxs = None
        self.oob_score = oob_score
        self.trees = []
        self.timeline = timeline
        self.random_state = random_state
        self.random_instance = check_random_state(self.random_state)

    def fit(self, x, y):
        """
        Build a forest of trees from the training set (X, y).
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event
        in the second with the shape [n_samples, 2]
        :return: self: object
        """

        try:
            if self.timeline is None:
                self.timeline = y.iloc[:, 1].sort_values().unique()
        except Exception:
            raise (
                "Timeline seems to contain float values. Please provide a custom timeline in the RandomSurvivalForest "
                "constructor. "
                "For example: RandomSurivalForest(timeline=range(y.iloc[:, 1].min(), y.iloc[:, 1].max(), 0.1)")

        self.bootstrap_idxs = self._draw_bootstrap_samples(x)

        num_cores = multiprocessing.cpu_count()

        if self.n_jobs > num_cores or self.n_jobs == -1:
            self.n_jobs = num_cores
        elif self.n_jobs is None:
            self.n_jobs = 1

        trees = Parallel(n_jobs=self.n_jobs)(delayed(self._create_tree)(x, y, i) for i in range(self.n_estimators))

        for i in range(len(trees)):
            if trees[i].prediction_possible:
                self.trees.append(trees[i])
                self.bootstraps.append(self.bootstrap_idxs[i])

        if self.oob_score:
            self.oob_score = self.compute_oob_score(x, y)

        return self

    def _create_tree(self, x, y, i: list):
        """
        Grows a survival tree for the bootstrap samples.
        :param y: label data frame y with survival time as the first column and event as second
        :param x: feature data frame x
        :param i: Indices
        :return: SurvivalTree
        """
        n_features = int(round(np.sqrt(x.shape[1]), 0))
        f_idxs = self.random_instance.permutation(x.shape[1])[:n_features]
        tree = SurvivalTree(x=x.iloc[self.bootstrap_idxs[i], :], y=y.iloc[self.bootstrap_idxs[i], :],
                            f_idxs=f_idxs, n_features=n_features,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            timeline=self.timeline, random_instance=self.random_instance)

        return tree
    def _compute_oob_ensembles(self, xs):
        """
        Compute OOB ensembles.
        :return: List of oob ensemble for each sample.
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_oob_ensemble_chf)(sample_idx, xs, self.trees, self.bootstraps) for sample_idx in
            range(xs.shape[0]))
        oob_ensemble_chfs = [i for i in results if not i.empty]
        return oob_ensemble_chfs

    def compute_oob_score(self, x, y):
        """
        Compute the oob score (concordance-index).
        :return: c-index of oob samples
        """
        oob_ensembles = self._compute_oob_ensembles(x)
        c = concordance_index(y_time=y.iloc[:, 1], y_pred=oob_ensembles, y_event=y.iloc[:, 0])
        return c

    def predict(self, xs):
        """
        Predict survival for xs.
        :param xs: The input samples
        :return: List of the predicted cumulative hazard functions.
        """
        ensemble_chfs = [self._compute_ensemble_chf(sample_idx=sample_idx, xs=xs, trees=self.trees)
                         for sample_idx in range(xs.shape[0])]
        return ensemble_chfs

    def _draw_bootstrap_samples(self, data):
        """
        Draw bootstrap samples
        :param data: Data to draw bootstrap samples of.
        :return: Bootstrap indices for each of the trees
        """
        bootstrap_idxs = []
        for i in range(self.n_estimators):
            no_samples = len(data)
            data_rows = range(no_samples)
            bootstrap_idx = self.random_instance.choice(data_rows, no_samples)
            bootstrap_idxs.append(bootstrap_idx)

        return bootstrap_idxs

    def _compute_ensemble_chf(self, sample_idx: int, xs, trees: list):
        denominator = 0
        numerator = 0
        for b in range(len(trees)):
            sample = xs.iloc[sample_idx].to_list()
            chf = trees[b].predict(sample)
            denominator = denominator + 1
            numerator = numerator + 1 * chf
        ensemble_chf = numerator / denominator
        return ensemble_chf

    def _compute_oob_ensemble_chf(self, sample_idx: int, xs, trees: list, bootstraps: list):
        denominator = 0
        numerator = 0
        for b in range(len(trees)):
            if sample_idx not in bootstraps[b]:
                sample = xs.iloc[sample_idx].to_list()
                chf = trees[b].predict(sample)
                denominator = denominator + 1
                numerator = numerator + 1 * chf
        if denominator != 0:
            oob_ensemble_chf = numerator / denominator
        else:
            oob_ensemble_chf = pd.Series()
        return oob_ensemble_chf


class SurvivalTree:

    def __init__(self, x, y, f_idxs, n_features, random_instance, timeline, unique_deaths=3, min_leaf=3):
        """
        A Survival Tree to predict survival.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.min_leaf = min_leaf
        self.unique_deaths = unique_deaths
        self.score = 0
        self.index = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.prediction_possible = None
        self.timeline = timeline
        self.random_instance = random_instance
        self.grow_tree()

    def grow_tree(self):
        """
        Grow the survival tree recursively as nodes.
        :return: self
        """
        unique_deaths = self.y.iloc[:, 0].reset_index().drop_duplicates().sum()[1]

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = _find_split(self)

        if self.split_var is not None and unique_deaths > self.unique_deaths:
            self.prediction_possible = True
            lf_idxs, rf_idxs = _select_new_feature_indices(self.x, self.n_features, self.random_instance)

            self.lhs = Node(x=self.x.iloc[lhs_idxs_opt, :], y=self.y.iloc[lhs_idxs_opt, :],
                            tree=self, f_idxs=lf_idxs, n_features=self.n_features,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            timeline=self.timeline, random_instance=self.random_instance)

            self.rhs = Node(x=self.x.iloc[rhs_idxs_opt, :], y=self.y.iloc[rhs_idxs_opt, :],
                            tree=self, f_idxs=rf_idxs, n_features=self.n_features,
                            unique_deaths=self.unique_deaths, min_leaf=self.min_leaf,
                            timeline=self.timeline, random_instance=self.random_instance)

            return self
        else:
            self.prediction_possible = False
            return self

    def predict(self, x):
        """
        Predict survival for x.
        :param x: The input sample.
        :return: The predicted cumulative hazard function.
        """
        if x[self.split_var] <= self.split_val:
            self.lhs.predict(x)
        else:
            self.rhs.predict(x)
        return self.chf


class Node:

    def __init__(self, x, y, tree: SurvivalTree, f_idxs: list, n_features: int, timeline, random_instance,
                 unique_deaths: int = 3, min_leaf: int = 3):
        """
        A Node of the Survival Tree.
        :param x: The input samples. Should be a Dataframe with the shape [n_samples, n_features].
        :param y: The target values as a Dataframe with the survival time in the first column and the event.
        :param tree: The corresponding Survival Tree
        :param f_idxs: The indices of the features to use.
        :param n_features: The number of features to use.
        :param unique_deaths: The minimum number of unique deaths required to be at a leaf node.
        :param min_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
        only be considered if it leaves at least min_leaf training samples in each of the left and right branches.
        """
        self.x = x
        self.y = y
        self.tree = tree
        self.f_idxs = f_idxs
        self.n_features = n_features
        self.unique_deaths = unique_deaths
        self.min_leaf = min_leaf
        self.score = 0
        self.split_val = None
        self.split_var = None
        self.lhs = None
        self.rhs = None
        self.chf = None
        self.chf_terminal = None
        self.terminal = False
        self.timeline = timeline
        self.random_instance = random_instance
        self.grow_tree()

    def grow_tree(self):
        """
        Grow tree by calculating the Nodes recursively.
        :return: self
        """
        unique_deaths = self.y.iloc[:, 0].reset_index().drop_duplicates().sum()[1]

        if unique_deaths <= self.unique_deaths:
            self.compute_terminal_node()
            return self

        self.score, self.split_val, self.split_var, lhs_idxs_opt, rhs_idxs_opt = _find_split(self)

        if self.split_var is None:
            self.compute_terminal_node()
            return self

        lf_idxs, rf_idxs = _select_new_feature_indices(self.x, self.n_features, self.random_instance)

        self.lhs = Node(self.x.iloc[lhs_idxs_opt, :], self.y.iloc[lhs_idxs_opt, :], self.tree, lf_idxs,
                        self.n_features, min_leaf=self.min_leaf, timeline=self.timeline,
                        random_instance=self.random_instance)

        self.rhs = Node(self.x.iloc[rhs_idxs_opt, :], self.y.iloc[rhs_idxs_opt, :], self.tree, rf_idxs,
                        self.n_features, min_leaf=self.min_leaf, timeline=self.timeline,
                        random_instance=self.random_instance)

        return self

    def compute_terminal_node(self):
        """
        Compute the terminal node if condition has reached.
        :return: self
        """
        self.terminal = True
        self.chf = NelsonAalenFitter()
        t = self.y.iloc[:, 1]
        e = self.y.iloc[:, 0]
        self.chf.fit(t, event_observed=e, timeline=self.timeline)
        return self

    def predict(self, x):
        """
        Predict the cumulative hazard function if its a terminal node. If not walk through the tree.
        :param x: The input sample.
        :return: Predicted cumulative hazard function if terminal node
        """
        if self.terminal:
            self.tree.chf = self.chf.cumulative_hazard_
            self.tree.chf = self.tree.chf.iloc[:, 0]
            return self.tree.chf.dropna()

        else:
            if x[self.split_var] <= self.split_val:
                self.lhs.predict(x)
            else:
                self.rhs.predict(x)


def _select_new_feature_indices(x, n_features: int, random_instance):
    lf_idxs = random_instance.permutation(x.shape[1])[:n_features]
    rf_idxs = random_instance.permutation(x.shape[1])[:n_features]

    return lf_idxs, rf_idxs


class ForestOverloads:
    def set_params(self,
                   **kwargs):
        """
        Ensure warm_Start is set to true, otherwise set other params as usual.

        :param kwargs: Params to set.
        """
        # Warm start should be true to get .fit() to keep existing estimators.
        kwargs['warm_start'] = True

        for key, value in kwargs.items():
            setattr(self, key, value)

        return self

    def fit(self, *args, pf_call: bool = False, classes_: Optional[np.ndarray] = None,
            sample_weight: Optional[np.array] = None, **kwargs):
        """
        This fit handles calling either super().fit or partial_fit depending on the caller.

        :param pf_call: True if called from partial fit, in this case super.fit() is called, instead of getting stuck in
                        a recursive loop.
        :param classes_: On pf calls, classes is passed from self.classes which will have already been set. These are
                         re-set after the call to super's fit, which will change them based on observed data.
        :param sample_weight: Sample weights. If None, then samples are equally weighted.
        """

        if not self.dask_feeding and not pf_call:
            if self.verbose > 0:
                print('Feeding with spf')
            self._sampled_partial_fit(*args)

        else:

            if self.verbose > 0:
                print('Fitting from a partial_fit call')
            super().fit(*args)
            if classes_ is not None:
                self.classes_ = classes_
                self.n_classes_ = len(classes_)

        return self


class ClassifierOverloads(ForestOverloads):
    """
    Overloaded methods specific to classifiers.
    """

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Call each predict proba from tree, and accumulate. This handle possibly inconsistent shapes, but isn't
        parallel?

        Cases where not all classes are presented in the first or subsequent subsets needs to be
        handled. For the RandomForestClassifier, tree predictions are averaged in
        sklearn.ensemble.forest.accumulate_prediction function. This sums the output matrix with dimensions
        n rows x n classes and fails if the class dimension differs.
        The class dimension is defined at the individual estimator level during the .fit() call, which sets the
        following attributes:
            - self.n_outputs_ = y.shape[1], which is then used by _validate_y_class_weight()), always called in .fit()
              to set:
                - self.classes_
                - self.n_classes_

        The .predict() method (sklearn.tree.tree.BaseDecisionTree.predict()) sets the output shape using:
            # Classification
            if is_classifier(self):
                if self.n_outputs_ == 1:
                    return self.classes_.take(np.argmax(proba, axis=1), axis=0)
                else:
                   [Not considering this yet]

        :param x:
        :return:
        """
        # Prepare expected output shape
        preds = np.zeros(shape=(x.shape[0], self.n_classes_),
                         dtype=np.float32)
        counts = np.zeros(shape=(x.shape[0], self.n_classes_),
                          dtype=np.int16)

        for e in self.estimators_:
            # Get the prediction from the tree
            est_preds = e.predict_proba(x)
            # Get the indexes of the classes present
            present_classes = e.classes_.astype(int)
            # Sum these in to the correct array columns
            preds[:, present_classes] += est_preds
            counts[:, present_classes] += 1

        # Normalise predictions against counts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            norm_prob = preds / counts

        # And remove nans (0/0) and infs (n/0)
        norm_prob[np.isnan(norm_prob) | np.isinf(norm_prob)] = 0

        return norm_prob


class OnlineStreamingRFC(ClassifierAdditions, ClassifierOverloads, RandomSurvivalForest):
    """
    Overload sklearn.ensemble.RandomForestClassifier to add partial fit method and new params.
    Note this init is a slightly different structure to ExtraTressClassifier/Regressor and RandomForestRegressor.
    """

    def __init__(self,
                 bootstrap=True,
                 class_weight=None,
                 criterion='gini',
                 max_depth=None,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 n_estimators_per_chunk: int = 1,
                 n_jobs=1,
                 oob_score=False,
                 random_state=None,
                 verbose=0,
                 warm_start: bool = True,
                 dask_feeding: bool = True,
                 max_n_estimators=10,
                 spf_n_fits=100,
                 spf_sample_prop=0.1) -> None:
        """
        :param bootstrap:
        :param class_weight:
        :param criterion:
        :param max_depth:
        :param max_features:
        :param max_leaf_nodes:
        :param min_impurity_decrease:
        :param min_impurity_split:
        :param min_samples_leaf:
        :param min_samples_split:
        :param min_weight_fraction_leaf:
        :param n_estimators_per_chunk: Estimators per chunk to fit.
        :param n_jobs:
        :param oob_score:
        :param random_state:
        :param verbose:
        :param warm_start:
        :param max_n_estimators: Total max number of estimators to fit.
        :param verb: If > 0 display debugging info during fit
        """

        # Run the super init, which also calls other parent inits to handle other params (like base estimator)
        super().__init__()

        self.max_n_estimators: int = None
        self._fit_estimators: int = 0
        self.classes_: np.array = None  # NB: Needs to be array, not list.
        self.n_classes_: int = None

        self.set_params(bootstrap=bootstrap,
                        class_weight=class_weight,
                        criterion=criterion,
                        max_depth=max_depth,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        min_impurity_split=min_impurity_split,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=min_samples_split,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        n_estimators_per_chunk=n_estimators_per_chunk,
                        n_estimators=n_estimators_per_chunk,
                        n_jobs=n_jobs,
                        oob_score=oob_score,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        _fit_estimators=0,
                        dask_feeding=dask_feeding,
                        max_n_estimators=max_n_estimators,
                        verb=0,
                        spf_n_fits=spf_n_fits,
                        spf_sample_prop=spf_sample_prop)


if __name__ == '__main__':
    from lifelines import datasets
    import numpy as np
    srfc = OnlineStreamingRFC(n_estimators_per_chunk=3,
                              max_n_estimators=np.inf,
                              spf_n_fits=30,
                              spf_sample_prop=0.3)
    rossi = datasets.load_rossi()
    y = rossi.loc[:, ["arrest", "week"]]
    X = rossi.drop(["arrest", "week"], axis=1)
    rsf=srfc.fit(X,y)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
    start_time = time.time()
    print(f'--- {round(time.time() - start_time, 3)} seconds ---')
    y_pred = rsf.predict(X_test)
    print(y_pred)
    c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
    print(f'C-index {round(c_val, 3)}')
