
import numpy as np
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 criterion='gini', min_samples_split=2, min_samples_leaf=1,
                 bootstrap=True, oob_score=False, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Populated during fit()
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_classes_ = None
        self.classes_ = None
        self._rng = np.random.default_rng(random_state)

 
    # Public API
   
    def fit(self, X, y):
      
        X = np.array(X, dtype=float)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Resolve max_features to an integer
        max_feats = self._resolve_max_features(n_features)

        # OOB accumulator: shape (n_samples, n_classes)
        if self.oob_score:
            oob_votes = np.zeros((n_samples, self.n_classes_), dtype=float)
            oob_counts = np.zeros(n_samples, dtype=int)

        self.estimators_ = []

        for i in range(self.n_estimators):
            # --- Bootstrap sampling ---
            if self.bootstrap:
                indices = self._rng.integers(0, n_samples, size=n_samples)
                X_boot, y_boot = X[indices], y[indices]
            else:
                indices = np.arange(n_samples)
                X_boot, y_boot = X, y

            # --- Build tree with feature randomness ---
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feats,
                random_state=int(self._rng.integers(0, 2**31))
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

            # --- OOB predictions ---
            if self.oob_score and self.bootstrap:
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[np.unique(indices)] = False
                if oob_mask.any():
                    oob_preds = tree.predict(X[oob_mask])
                    for sample_idx, pred in zip(np.where(oob_mask)[0], oob_preds):
                        class_idx = np.where(self.classes_ == pred)[0][0]
                        oob_votes[sample_idx, class_idx] += 1
                        oob_counts[sample_idx] += 1

        # --- Aggregate feature importances ---
        self.feature_importances_ = np.mean(
            [tree.get_feature_importance() for tree in self.estimators_], axis=0)

        # --- OOB score ---
        if self.oob_score:
            # Only samples that appeared in at least one OOB set
            valid = oob_counts > 0
            if valid.sum() == 0:
                self.oob_score_ = None
            else:
                oob_predictions = self.classes_[np.argmax(oob_votes[valid], axis=1)]
                self.oob_score_ = (oob_predictions == y[valid]).mean()

        return self

    def predict(self, X):
        
        X = np.array(X, dtype=float)
        # Collect votes: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority vote for each sample
        predictions = []
        for sample_votes in all_preds.T:               # shape (n_estimators,)
            vote_counts = Counter(sample_votes)
            predictions.append(vote_counts.most_common(1)[0][0])
        return np.array(predictions)

    def predict_proba(self, X):
       
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        vote_matrix = np.zeros((n_samples, self.n_classes_), dtype=float)

        for tree in self.estimators_:
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                class_idx = np.where(self.classes_ == pred)[0][0]
                vote_matrix[i, class_idx] += 1

        return vote_matrix / self.n_estimators

    def get_feature_importance(self):
       
        return self.feature_importances_


    # Internal helpers


    def _resolve_max_features(self, n_features):
        
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif self.max_features == 'all' or self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            raise ValueError(f"Unknown max_features: '{self.max_features}'. "
                             "Use 'sqrt', 'log2', 'all', an int, or a float.")

    def __repr__(self):
        return (f"RandomForestClassifier("
                f"n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features='{self.max_features}', "
                f"criterion='{self.criterion}')")



# Import fix for majority vote

from collections import Counter   # noqa: E402  (already imported in decision_tree)



# Quick sanity check


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import time

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=50, max_depth=None,
                                 oob_score=True, random_state=42)
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    train_acc = (rf.predict(X_train) == y_train).mean()
    test_acc = (rf.predict(X_test) == y_test).mean()

    print(f"RandomForest | train={train_acc:.4f}  test={test_acc:.4f}  "
          f"oob={rf.oob_score_:.4f}  time={train_time:.2f}s")
    print(f"Feature importances: {rf.get_feature_importance().round(3)}")
