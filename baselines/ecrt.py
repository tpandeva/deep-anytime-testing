import numpy as np
from enum import Enum
import json
from sklearn.linear_model import LassoCV, Lasso
from warnings import simplefilter
from data import get_cit_data
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
import os
import pickle


def get_data_statistics(X):
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)
    params_dict = {"X_mu": mu,
                   "X_sigma": sigma}
    return params_dict


def create_conditional_gauss(X, j, mu, sigma):
    """"
    This function learns the conditional distribution of X_j|X_-j

    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param mu, sigma: The mean and covariance of X.
    :return: The mean and covariance of the conditional distribution.
    To learn more about the implementation see: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    """
    a = np.delete(X, j, 1)
    mu_1 = np.array([mu[j]])
    mu_2 = np.delete(mu, j, 0)
    sigma_11 = sigma[j, j]
    sigma_12 = np.delete(sigma, j, 1)[j, :]
    sigma_21 = np.delete(sigma, j, 0)[:, j]
    sigma_22 = np.delete(np.delete(sigma, j, 0), j, 1)
    mu_bar_vec = []
    sigma12_22 = sigma_12 @ np.linalg.inv(sigma_22)
    sigma_bar = sigma_11 - sigma12_22 @ sigma_21
    for a_i in a:
        mu_bar = mu_1 + sigma12_22 @ (a_i - mu_2)
        mu_bar_vec.append(mu_bar)

    return mu_bar_vec, np.sqrt(sigma_bar)


def sample_from_gaussian(X, j, X_mu, X_sigma):
    """
    This function samples the dummy features for gaussian distribution.
    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param X_mu, X_sigma: The mean and covariance of X.
    :return: A copy of the batch X, with the dummy features in the j-th column.
    """
    mu_tilde, sigma_tilde = create_conditional_gauss(X, j, X_mu, X_sigma)
    n = X.shape[0]
    X_tilde = X.copy()
    Xj_tilde = np.random.normal(mu_tilde, sigma_tilde, (n, 1))
    X_tilde[:, j] = Xj_tilde.ravel()
    return X_tilde


def get_hiv_prob(X, clf):
    return clf.predict_proba(X)[:, 1]


def sample_hiv_data(X, j, clf):
    """
    This function sample the dummy features for binary original features.
    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param clf: A trained classifier, trained to predict the j-th feature given all other features.
    :return: A copy of the batch X, with the dummy features in the j-th column.
    """
    features_idx = np.arange(X.shape[1])
    train_features = np.setdiff1d(features_idx, j)
    prob = get_hiv_prob(X[:, train_features], clf)
    n = X.shape[0]
    X_tilde = X.copy()
    u_sample = np.random.uniform(0, 1, n)
    Xj_tilde = np.zeros((n,))
    Xj_tilde[u_sample < prob] = 1
    X_tilde[:, j] = Xj_tilde.ravel()
    return X_tilde


class BettingFunction(Enum):
    sign = lambda a, b: np.sign(b - a)
    tanh = lambda a, b: np.tanh(20 * (b - a) / np.max((a, b)))


class TestStatistic(Enum):
    mse = lambda a, b: ((a - b) ** 2).mean()


def get_martingale_values(martingale_dict):
    b_last_used_list = []
    st_list = []
    for b in martingale_dict.keys():
        st_list.append(martingale_dict[b]["St"])
        b_last_used_list.append(martingale_dict[b]["last_used_idx"])
    return np.array(st_list).mean(), np.array(b_last_used_list).max()


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def lasso_cv_online_learning(X, y, models_dict, val_prcg=0.2):
    """
    Online hyper-parameter tuning using ensemble of Lasso models
    :param X: The data matrix with size (n, d).
    :param y: A vector of labels with size (n, 1).
    :param models_dict: A dictionary contains M models.
    The keys are the values of the tuned parameter (the regularization constant the multiplies the L1 term
    in the loss function).
    The values in the dictionary are the lasso models with the corresponding parameter.
    :param val_prcg: The percentage of data to be used for validation.
    :return: The regularization constant of the model that got the best score on the validation data.
    """
    train_idx = int(len(y) * (1 - val_prcg))
    X_train = X[:train_idx, :]
    y_train = y[:train_idx]
    X_val = X[train_idx:, :]
    y_val = y[train_idx:]
    alpha_vec = list(models_dict.keys())
    score = -1
    best_alpha = alpha_vec[0]
    for alpha in alpha_vec:
        models_dict[alpha].fit(X_train, y_train.ravel())
        model_score = models_dict[alpha].score(X_val, y_val.ravel())
        if model_score > score:
            best_alpha = alpha
            score = model_score
    return best_alpha


class EcrtTester:
    """
    Conditional Testing with e-CRT
    """

    def __init__(self, batch_list=[2, 5, 10], n_init=50, K=20, j=0,
                 g_func=BettingFunction.sign, test_statistic=TestStatistic.mse, offline=False,
                 path="../results", load_name="", save_name="martingale_dict",
                 learn_conditional_distribution=get_data_statistics,
                 sampling_func=sample_from_gaussian, sampling_args={}
                 ):
        """
        :param batch_list: A list of batch sizes for the batch-ensemble.
        All batches must be divisors of the maximal one.
        :param n_init: The number of samples for the initial training.
        :param K: The de-randomization parameter. Number of dummy copies to be used for the wealth computation.
        :param j: The index of the tested feature. If you wish to test a different feature,
        you should create a new instance.
        :param g_func: The betting score function. Must be antisymmetric.
        :param test_statistic: The test statistic function, used to compare between the original and the dummy features.
        :param offline: Train offline LassoCV instead of online Lasso.
        :param path: Folder path to save and load martingales data.
        :param load_name: File name to load old martingales data.
        If given, the online updates start from the last saved wealth, and the last used point.
        If not given, the test starts from initial wealth 1.
        If you choose to load previous data, make sure to run with the same batch list, and on the same feature j.
        :param save_name: File name to save martingales data.
        :param learn_conditional_distribution: This function get X, the dataset,
        and returns learned arguments that are needed for the sampling of X_tilde.
        The returned arguments are saved to the dictionary sampling_args, and passed to the sampling function.
        :param sampling_func: This function gets X, j, and the additional arguments in sampling_args,
        and returns the dummy features X_tilde.
        :param sampling_args: A dictionary with all the non-learned arguments to pass to the sampling functions.
        """
        max_b = np.max(batch_list)
        for b in batch_list:
            assert max_b % b == 0
        self.batch_list = batch_list
        self.n_init = n_init
        self.K = K
        self.j = j
        self.g_func = g_func
        self.test_statistic = test_statistic
        self.offline = offline
        self.path = path
        self.load_name = load_name
        self.save_name = save_name
        self.integral_vector = np.linspace(0, 1, 1001, endpoint=False)[1:]
        self._initialize_martingales()
        self.model = None
        self.models_dict = {}
        self.sampling_args = sampling_args
        self.sampling_func = sampling_func
        self.learn_conditional_distribution = learn_conditional_distribution

    def _initialize_martingales(self):
        if self.load_name:
            with open(f"{self.path}/{self.load_name}.json") as json_file:
                self.martingale_dict = json.load(json_file)
            for b in self.batch_list:
                self.martingale_dict[b] = self.martingale_dict.pop(str(b))
        else:
            self.martingale_dict = {}
            for b in self.batch_list:
                self.martingale_dict[b] = {"St": 1,
                                           "St_v": np.ones((1000,)),
                                           "last_used_idx": self.n_init}

    def save_martingales(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        with open(f"{self.path}/{self.save_name}.json", 'w') as json_file:
            json.dump(self.martingale_dict, json_file, default=default, indent=4)

    def _sample_dummy(self, X):
        X_tilde = self.sampling_func(X, self.j, **self.sampling_args)
        return X_tilde

    def _initialize_online_lasso(self, X, y):
        # In the online learning we use 20 Lasso models to choose the best eta hyper-parameter.
        # The grid of eta is set as follows:
        eps = 5e-3
        train_size = X.shape[0]
        Xy = np.dot(X.T, y)
        eta_max = np.sqrt(np.sum(Xy ** 2, axis=1)).max() / (train_size * 1)
        eta_vec = np.logspace(np.log10(eta_max * eps), np.log10(eta_max), num=20)
        # The models are trained using 80% of the train set. The hold out points are used to choose the best eta.
        models_dict = {}
        for eta in eta_vec:
            models_dict[eta] = Lasso(alpha=eta, warm_start=True)
        best_eta = lasso_cv_online_learning(X, y, models_dict, val_prcg=0.2)
        # We set the best eta to the main model, and fit the main model on the train set.
        model = Lasso(alpha=best_eta, warm_start=True)
        model.fit(X, y.ravel())
        model.max_iter = 50
        self.model = model
        self.models_dict = models_dict

    def _update_martingale(self, X, y, batch, test_idx, St_v):
        """
        The online update of one batch of samples (a single update).
        :param X: The full data set with d features.
        :param y: The full labels set.
        :param batch: The batch size b
        :param test_idx: The index of the first new point to be used in the update.
        The evaluation will be applied on the batch [test_idx : test_idx + batch].
        :param St_v: A vector with 1000 values, hold the martingales from the previous update.
        :return: A scalar St, the result of the integral over the 1000 martingales, with uniform density.
        :return A vector St_v, with the updated martingales.
        """
        wealth = 0
        # Compute the MSE (or any other test statistic) of the original features once.
        y_predict = self.model.predict(X[test_idx:test_idx + batch, :])
        q = self.test_statistic(y_predict.ravel(), y[test_idx:test_idx + batch].ravel())
        # For K iterations, sample the dummy features, compute the dummy MSE
        # and update the wealth using the betting score function, g_func.
        for k in range(self.K):
            X_tilde = self._sample_dummy(X[test_idx:test_idx + batch, :])
            y_tilde = self.model.predict(X_tilde)
            q_tilde = self.test_statistic(y_tilde.ravel(), y[test_idx:test_idx + batch].ravel())
            wealth += self.g_func(q, q_tilde)
        # Update the martingales using the average betting score.
        St_v = St_v * (1 + self.integral_vector * wealth / self.K)
        St = np.mean(St_v)  # integral with uniform density.
        return St, St_v

    def run(self, X, y, start_idx=None, alpha=0.05):
        """
        :param X: The data matrix with size (n, d).
        Note that even if you run using old martingales data, and start_idx is not None,
        you should provide the old data. The data will not be used to update the martingales,
        but will be used to train the learning model.
        :param y: A vector of labels with size (n, 1)
        :param start_idx: The first sample that will be used to update the martingales.
        All points before it will be used for training only. If None, the first sample will be n_init.
        :param alpha: The target level. The null will be rejected when the martingale will reach 1/alpha.
        :return: Whether the null is rejected or not, i.e., whether the tested feature is important or not.
        """
        rejected = -1
        if start_idx is None:
            start_idx = self.n_init
        # Train the model on the available data points, that are not used for the martingales update.
        # If you wish to use a different predictive model, please replace the Lasso model here.
        if self.offline:
            self.model = LassoCV(max_iter=10000, eps=5e-3).fit(X[:start_idx, :], y[:start_idx].ravel())
        else:
            self._initialize_online_lasso(X[:start_idx, :], y[:start_idx])
        n = X.shape[0]
        # Run the sequential updates
        St_running = []
        for new_points in np.arange(start_idx, n, 1):
            b_last_used_list = []
            st_list = []
            update = False
            # Ensemble over batches
            for b in self.batch_list:
                # Once we use a point to update the martingale, we can not use it again for updates, only for training.
                # In this condition we validate that we train the model using old samples, and that we update the
                # martingale using new unseen points.
                if new_points == self.martingale_dict[b]["last_used_idx"]:
                    # This step is skipped if we already updated the model in a previous iteration,
                    # for a smaller batch size.
                    if not update:
                        # Train the model on the valid training data.
                        if not self.offline:
                            best_eta = lasso_cv_online_learning(X[:new_points, :], y[:new_points], self.models_dict,
                                                                  val_prcg=0.2)
                            self.model.alpha = best_eta
                        self.model.fit(X[:new_points, :], y[:new_points].ravel())
                    update = True
                    # Update the martingale using the new batch of samples.
                    self.martingale_dict[b]["St"], self.martingale_dict[b]["St_v"] = self._update_martingale(
                        X, y, b, new_points, self.martingale_dict[b]["St_v"])
                    self.martingale_dict[b]["last_used_idx"] = min(new_points + b, n)
                b_last_used_list.append(self.martingale_dict[b]["last_used_idx"])
                st_list.append(self.martingale_dict[b]["St"])
            # If all the martingales used the same number of points, update the ensemble.

            st_vec = np.array(st_list)
            St = st_vec.mean()
            St_running.append(St)
            if len(set(b_last_used_list)) == 1 and update:
                # If the ensemble martingale passes the test level, we can safely reject the null.
                if St > 1 / alpha:
                    if rejected == -1: rejected = new_points
        return  rejected, St_running


if __name__ == "__main__":
    j = 0  # In this setting, the tested feature is always the first one.
    n_exp = 100  # We run the test 100 times on different realizations, in order to evaluate the power and the error.
    seed_vec = np.arange(n_exp)
    tests_list = ["type2", "type1"]
    results_dict = {}
    d = 20
    for test in tests_list:
        results_dict[test] = {
            "sts": []
        }

    for test in tests_list:
        rejected_vec = np.zeros((n_exp,))

        for ii, seed in enumerate(seed_vec):
            X_total, Y_total = None, None
            for seq in range(50):
                print(f"Seed: {ii}, seq: {seq}")
                X, Y, mu = get_cit_data(test=test, n=100, seed=(seed + 1) * 100 + seq)
                n_samples = X_total.shape[0] if X_total is not None else X.shape[0]
                X_total = X if X_total is None else np.concatenate((X_total, X), axis=0)
                Y_total = Y if Y_total is None else np.concatenate((Y_total, Y), axis=0)

                sampling_args = {"X_mu": np.array([0] * d),
                                 "X_sigma": np.eye(d)}
                ecrt_tester = EcrtTester(n_init=n_samples, j=j,
                                         sampling_args=sampling_args)  # In this simple run, almost all the input parameters are the default ones.
                rejected, Sts = ecrt_tester.run(X_total, Y_total)
                print("Rejected at: ", rejected)
                results_dict[test]["sts"].append(Sts)

    if not os.path.exists("logs/cit/"):
        os.makedirs("logs/cit/")
    pickle.dump(results_dict, open("logs/cit/crt.pkl", "wb"))

