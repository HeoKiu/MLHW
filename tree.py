import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    p = np.sum(y, axis=0) / y.shape[0]
    H = -np.sum(p * np.log2(p + EPS))

    return H

def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    return 1 - (y.mean(axis=0) ** 2).sum()



def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE
    Var = np.mean((y - np.mean(y)) ** 2)

    return Var


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE

    return (np.abs(y - np.mean(y))).mean()


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        X_left = X_subset[X_subset[:, feature_index] < threshold]
        y_left = y_subset[X_subset[:, feature_index] < threshold]
        X_right = X_subset[X_subset[:, feature_index] >= threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold
        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        y_left, y_right = y_subset[X_subset[:, feature_index] < threshold], y_subset[X_subset[:, feature_index] >= threshold]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with
        threshold : float
            Threshold value to perform split
        """
        # YOUR CODE HERE
        best_criterion_value = np.inf
        feature_index = 0
        threshold = 0
        sample_size, feature_count = X_subset.shape

        for id in range(feature_count):
            for cur_threshold in np.sort(np.unique(X_subset.T[id]))[1:-1]:
                y_l, y_r = self.make_split_only_y(id, cur_threshold, X_subset, y_subset)
                L = len(y_l) / len(X_subset) * self.criterion(y_l) + len(y_r) / len(X_subset) * self.criterion(y_r)

                if L < best_criterion_value:
                    best_criterion_value = L
                    feature_index = id
                    threshold = cur_threshold

        return feature_index, threshold
        return feature_index, threshold

    def make_tree(self, X_subset, y_subset, depth = 0):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset
        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if X_subset.shape[0] > self.min_samples_split and depth < self.max_depth:
            new_node = Node(*self.choose_best_split(X_subset, y_subset))
            (X_left, y_left), (X_right, y_right) = self.make_split(new_node.feature_index, new_node.value, X_subset,
                                                                   y_subset)
            new_node.left_child, new_node.right_child = self.make_tree(X_left, y_left, depth + 1), self.make_tree(
                X_right, y_right, depth + 1)
            return new_node
        else:
            return y_subset.mean(axis=0)



    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on
        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for
        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        # YOUR CODE HERE
        y_predicted = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            current_node = self.root

            while type(current_node) is Node:
                if X[i, current_node.feature_index] < current_node.value:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

            if self.all_criterions[self.criterion_name][1]:
                y_predicted[i] = np.argmax(current_node)
            else:
                y_predicted[i] = current_node

        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for
        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))

        for i in range(X.shape[0]):
            current_node = self.root

            while type(current_node) is Node:
                if X[i, current_node.feature_index] < current_node.value:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

            y_predicted_probs[i] = current_node

        return y_predicted_probs



