import numpy as np
from collections import Counter



# Tree Node

class TreeNode:
   

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.value = None
        self.samples = None
        self.impurity = None


# Decision Tree Classifier


class DecisionTreeClassifier:
    

    def __init__(self, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        # Set after fit()
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self._rng = np.random.default_rng(random_state)

    # Public API
  

    def fit(self, X, y):
       
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.feature_importances_ = np.zeros(self.n_features_)

        self.root = self._build_tree(X, y, depth=0)

        # Normalise importances so they sum to 1
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def predict(self, X):
       
        X = np.array(X, dtype=float)
        return np.array([self._predict_sample(x, self.root) for x in X])

    def predict_proba(self, X):
        
        X = np.array(X, dtype=float)
        return np.array([self._predict_proba_sample(x, self.root) for x in X])

    def get_feature_importance(self):
        
        return self.feature_importances_

    
    # Impurity Measures


    def _gini(self, y):
       
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
        
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y)          # ignore zero-count classes
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y):
        
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: '{self.criterion}'. Use 'gini' or 'entropy'.")

    def _information_gain(self, y, y_left, y_right):
       
        n = len(y)
        if n == 0:
            return 0.0
        parent_impurity = self._impurity(y)
        weighted_child = (len(y_left) / n) * self._impurity(y_left) + \
                         (len(y_right) / n) * self._impurity(y_right)
        return parent_impurity - weighted_child

    
    # Split Search
   

    def _find_best_split(self, X, y):
       
        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        # Determine which features to evaluate
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            k = min(self.max_features, n_features)
            feature_indices = self._rng.choice(n_features, size=k, replace=False)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_vals = np.unique(feature_values)

            if len(unique_vals) < 2:
                continue  # No split possible on a constant feature

            # Use midpoints between consecutive unique values as candidate thresholds
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Enforce min_samples_leaf
                if left_mask.sum() < self.min_samples_leaf or \
                   right_mask.sum() < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    
    # Tree Construction
    

    def _stopping_condition(self, y, depth):
        
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if len(np.unique(y)) == 1:
            return True
        return False

    def _majority_class(self, y):
        
        return Counter(y).most_common(1)[0][0]

    def _build_tree(self, X, y, depth):
        
        node = TreeNode()
        node.samples = len(y)
        node.impurity = self._impurity(y)

        # --- Leaf node ---
        if self._stopping_condition(y, depth):
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node

        # --- Find best split ---
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        # No valid split found → leaf
        if best_feature is None or best_gain == 0:
            node.is_leaf = True
            node.value = self._majority_class(y)
            return node

        # --- Partition data ---
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # --- Accumulate feature importance ---
        # Weighted impurity decrease: (n/N) * gain  (will normalise later)
        self.feature_importances_[best_feature] += (node.samples / 1) * best_gain

        # --- Populate internal node ---
        node.feature = best_feature
        node.threshold = best_threshold

        # --- Recurse ---
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    # Prediction Helpers
  

    def _predict_sample(self, x, node):
       
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def _predict_proba_sample(self, x, node):
       
        # Collect training labels at this leaf to estimate probabilities
        leaf_node = self._reach_leaf(x, self.root)
        # We store value (majority class) but not distribution by default.
        # For proper proba, fall back to one-hot of majority class here.
        # (Full leaf distributions are tracked by RandomForest via vote counts.)
        proba = np.zeros(self.n_classes_)
        proba[int(leaf_node.value)] = 1.0
        return proba

    def _reach_leaf(self, x, node):
        """Walk the tree and return the leaf node reached by sample x."""
        if node.is_leaf:
            return node
        if x[node.feature] <= node.threshold:
            return self._reach_leaf(x, node.left)
        else:
            return self._reach_leaf(x, node.right)

 
    # Utility
   

    def get_depth(self):
        """Return the actual maximum depth of the fitted tree."""
        return self._tree_depth(self.root)

    def _tree_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))

    def get_n_leaves(self):
        """Return the total number of leaf nodes."""
        return self._count_leaves(self.root)

    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def __repr__(self):
        return (f"DecisionTreeClassifier("
                f"criterion='{self.criterion}', "
                f"max_depth={self.max_depth}, "
                f"min_samples_split={self.min_samples_split})")



# Quick sanity check


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    for criterion in ('gini', 'entropy'):
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=5,
                                      random_state=42)
        tree.fit(X_train, y_train)
        train_acc = (tree.predict(X_train) == y_train).mean()
        test_acc = (tree.predict(X_test) == y_test).mean()
        print(f"[{criterion}] depth={tree.get_depth()}  "
              f"leaves={tree.get_n_leaves()}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}")
    print("Feature importances:", tree.get_feature_importance().round(3))
