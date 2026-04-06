
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for scripts
warnings.filterwarnings('ignore')

# ── Local implementations ──
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier

# ── sklearn baselines ──
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ── Output directory ──
FIGURES_DIR = '../figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Plot style ──
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
})
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']



# 0.  DATA LOADING & PREPROCESSING


def load_wine_quality(url=None):
    
    csv_url = (url or
               "https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "wine-quality/winequality-red.csv")
    print(f"Loading Wine Quality dataset from:\n  {csv_url}")
    try:
        df = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        # Offline fallback — generate a synthetic stand-in for testing
        print(f"  [WARNING] Could not fetch CSV ({e}). Using synthetic data.")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 1599
        cols = ['fixed acidity','volatile acidity','citric acid',
                'residual sugar','chlorides','free sulfur dioxide',
                'total sulfur dioxide','density','pH','sulphates',
                'alcohol','quality']
        data = rng.random((n, 11))
        quality = rng.integers(3, 9, size=n)
        df = pd.DataFrame(np.column_stack([data, quality]), columns=cols)

    print(f"  Shape: {df.shape}  |  Quality range: {df['quality'].min()}-{df['quality'].max()}")

    feature_names = [c for c in df.columns if c != 'quality']
    X = df[feature_names].values.astype(float)
    y = (df['quality'] >= 6).astype(int).values    # binary: 0 = low, 1 = high

    print(f"  Class distribution: low={( y==0).sum()}  high={(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test, feature_names


# 1.  EXPERIMENT 1 — MODEL COMPARISON


def experiment1_model_comparison(X_train, X_test, y_train, y_test, feature_names):
    
    print("=" * 60)
    print("EXPERIMENT 1: Model Comparison")
    print("=" * 60)

    models = {
        'Custom DT':    DecisionTreeClassifier(max_depth=10, criterion='gini',
                                               random_state=RANDOM_STATE),
        'Custom RF':    RandomForestClassifier(n_estimators=100, max_depth=None,
                                               random_state=RANDOM_STATE),
        'sklearn DT':   SklearnDT(max_depth=10, criterion='gini',
                                  random_state=RANDOM_STATE),
        'sklearn RF':   SklearnRF(n_estimators=100, max_depth=None,
                                  random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predict
        t0 = time.time()
        y_pred_test = model.predict(X_test)
        pred_time = time.time() - t0

        y_pred_train = model.predict(X_train)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test,  y_pred_test)

        results[name] = {
            'train_acc': train_acc,
            'test_acc':  test_acc,
            'train_time': train_time,
            'pred_time':  pred_time,
        }
        print(f"  {name:<12} | train={train_acc:.4f}  test={test_acc:.4f} "
              f"| fit={train_time:.2f}s  pred={pred_time:.3f}s")

    # --- Plot 1a: Test accuracy comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(results.keys())

    # Bar: test accuracy
    ax = axes[0]
    bars = ax.bar(names, [r['test_acc'] for r in results.values()],
                  color=COLORS[:4], edgecolor='white', linewidth=1.2)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.tick_params(axis='x', rotation=15)

    # Bar: train vs test accuracy
    ax = axes[1]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, [r['train_acc'] for r in results.values()],
                w, label='Train', color=COLORS[0], alpha=0.85)
    b2 = ax.bar(x + w/2, [r['test_acc']  for r in results.values()],
                w, label='Test',  color=COLORS[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test Accuracy')
    ax.legend()
    ax.bar_label(b1, fmt='%.3f', padding=2, fontsize=8)
    ax.bar_label(b2, fmt='%.3f', padding=2, fontsize=8)

    # Bar: training time
    ax = axes[2]
    bars = ax.bar(names, [r['train_time'] for r in results.values()],
                  color=COLORS[2:6], edgecolor='white')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison')
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Experiment 1: Model Comparison — Wine Quality Dataset',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  [saved] {path}")
    return results



# 2.  EXPERIMENT 2 — HYPERPARAMETER TUNING

def experiment2_hyperparameter_tuning(X_train, X_test, y_train, y_test):
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Hyperparameter Tuning")
    print("=" * 60)

    # --- 2a. DT max_depth ---
    depths = [1, 2, 3, 5, 10, 15, 20, None]
    depth_labels = [str(d) if d is not None else 'None' for d in depths]
    dt_train, dt_test = [], []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)
        dt_train.append(accuracy_score(y_train, dt.predict(X_train)))
        dt_test.append(accuracy_score(y_test,  dt.predict(X_test)))
    print(f"  DT optimal depth: "
          f"{depth_labels[int(np.argmax(dt_test))]}  "
          f"(test={max(dt_test):.4f})")

    # --- 2b. DT min_samples_split ---
    min_splits = [2, 5, 10, 20, 50]
    mss_train, mss_test = [], []
    for ms in min_splits:
        dt = DecisionTreeClassifier(max_depth=10, min_samples_split=ms,
                                    random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)
        mss_train.append(accuracy_score(y_train, dt.predict(X_train)))
        mss_test.append(accuracy_score(y_test,  dt.predict(X_test)))

    # --- 2c. DT criterion ---
    crit_results = {}
    for crit in ('gini', 'entropy'):
        dt = DecisionTreeClassifier(max_depth=10, criterion=crit,
                                    random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)
        crit_results[crit] = {
            'train': accuracy_score(y_train, dt.predict(X_train)),
            'test':  accuracy_score(y_test,  dt.predict(X_test)),
        }
    print(f"  gini  → test={crit_results['gini']['test']:.4f}")
    print(f"  entropy → test={crit_results['entropy']['test']:.4f}")

    # --- 2d. RF n_estimators ---
    n_trees = [1, 5, 10, 25, 50, 100, 200]
    rf_n_train, rf_n_test = [], []
    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n, max_depth=None,
                                    random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        rf_n_train.append(accuracy_score(y_train, rf.predict(X_train)))
        rf_n_test.append(accuracy_score(y_test,  rf.predict(X_test)))
    print(f"  RF converges around {n_trees[int(np.argmax(rf_n_test))]} trees  "
          f"(test={max(rf_n_test):.4f})")

    # --- 2e. RF max_features ---
    feat_options = [1, 'sqrt', 'log2', 'all']
    rf_feat_train, rf_feat_test = [], []
    for mf in feat_options:
        rf = RandomForestClassifier(n_estimators=50, max_depth=None,
                                    max_features=mf, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        rf_feat_train.append(accuracy_score(y_train, rf.predict(X_train)))
        rf_feat_test.append(accuracy_score(y_test,  rf.predict(X_test)))

    # --- 2f. 2D grid: depth × min_samples_split ---
    depths_grid = [3, 5, 10, 15, 20]
    mss_grid = [2, 5, 10, 20, 50]
    heatmap = np.zeros((len(depths_grid), len(mss_grid)))
    for i, d in enumerate(depths_grid):
        for j, ms in enumerate(mss_grid):
            dt = DecisionTreeClassifier(max_depth=d, min_samples_split=ms,
                                        random_state=RANDOM_STATE)
            dt.fit(X_train, y_train)
            heatmap[i, j] = accuracy_score(y_test, dt.predict(X_test))

    # ---- Plots ------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 2a: depth learning curve
    ax = axes[0, 0]
    ax.plot(depth_labels, dt_train, 'o-', label='Train', color=COLORS[0])
    ax.plot(depth_labels, dt_test,  's--', label='Test', color=COLORS[1])
    ax.set_xlabel('max_depth'); ax.set_ylabel('Accuracy')
    ax.set_title('DT: Accuracy vs max_depth')
    ax.legend(); ax.grid(alpha=0.3)

    # 2b: min_samples_split
    ax = axes[0, 1]
    ax.plot(min_splits, mss_train, 'o-', label='Train', color=COLORS[0])
    ax.plot(min_splits, mss_test,  's--', label='Test', color=COLORS[1])
    ax.set_xlabel('min_samples_split'); ax.set_ylabel('Accuracy')
    ax.set_title('DT: Accuracy vs min_samples_split')
    ax.legend(); ax.grid(alpha=0.3)

    # 2c: criterion bar
    ax = axes[0, 2]
    crits = list(crit_results.keys())
    x = np.arange(len(crits))
    w = 0.35
    ax.bar(x - w/2, [crit_results[c]['train'] for c in crits], w,
           label='Train', color=COLORS[0])
    ax.bar(x + w/2, [crit_results[c]['test']  for c in crits], w,
           label='Test',  color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(crits)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel('Accuracy')
    ax.set_title('DT: Gini vs Entropy')
    ax.legend()

    # 2d: RF n_estimators
    ax = axes[1, 0]
    ax.plot(n_trees, rf_n_train, 'o-', label='Train', color=COLORS[2])
    ax.plot(n_trees, rf_n_test,  's--', label='Test', color=COLORS[3])
    ax.set_xlabel('n_estimators'); ax.set_ylabel('Accuracy')
    ax.set_title('RF: Accuracy vs n_estimators')
    ax.legend(); ax.grid(alpha=0.3)

    # 2e: RF max_features
    ax = axes[1, 1]
    feat_labels = [str(f) for f in feat_options]
    x = np.arange(len(feat_options))
    w = 0.35
    ax.bar(x - w/2, rf_feat_train, w, label='Train', color=COLORS[2])
    ax.bar(x + w/2, rf_feat_test,  w, label='Test',  color=COLORS[3])
    ax.set_xticks(x); ax.set_xticklabels(feat_labels)
    ax.set_ylabel('Accuracy'); ax.set_title('RF: Accuracy vs max_features')
    ax.legend()

    # 2f: grid search heatmap
    ax = axes[1, 2]
    im = ax.imshow(heatmap, cmap='YlGn', aspect='auto',
                   vmin=heatmap.min() - 0.01, vmax=heatmap.max())
    ax.set_xticks(range(len(mss_grid)));    ax.set_xticklabels(mss_grid)
    ax.set_yticks(range(len(depths_grid))); ax.set_yticklabels(depths_grid)
    ax.set_xlabel('min_samples_split'); ax.set_ylabel('max_depth')
    ax.set_title('DT Grid Search: Test Accuracy')
    plt.colorbar(im, ax=ax)
    for i in range(len(depths_grid)):
        for j in range(len(mss_grid)):
            ax.text(j, i, f'{heatmap[i,j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black')

    plt.suptitle('Experiment 2: Hyperparameter Tuning — Wine Quality Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'hyperparameter_tuning.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  [saved] {path}")



# 3.  EXPERIMENT 3 — FEATURE IMPORTANCE


def experiment3_feature_importance(X_train, X_test, y_train, y_test,
                                   feature_names):
   
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Feature Importance")
    print("=" * 60)

    # Fit models
    dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    dt_imp = dt.get_feature_importance()
    rf_imp = rf.get_feature_importance()

    # Sorted indices (descending)
    rf_sorted = np.argsort(rf_imp)[::-1]
    print("  Top 5 features (RF):")
    for rank, idx in enumerate(rf_sorted[:5], 1):
        print(f"    {rank}. {feature_names[idx]:<30} {rf_imp[idx]:.4f}")

    # --- Performance with top-k features ---
    k_values = [1, 3, 5, 8, 11]
    dt_topk, rf_topk = [], []
    for k in k_values:
        top_feats = rf_sorted[:k]
        # DT
        dt_k = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
        dt_k.fit(X_train[:, top_feats], y_train)
        dt_topk.append(accuracy_score(y_test, dt_k.predict(X_test[:, top_feats])))
        # RF
        rf_k = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        rf_k.fit(X_train[:, top_feats], y_train)
        rf_topk.append(accuracy_score(y_test, rf_k.predict(X_test[:, top_feats])))
    print(f"  DT with all features: {dt_topk[-1]:.4f}  | top-5: {dt_topk[2]:.4f}")
    print(f"  RF with all features: {rf_topk[-1]:.4f}  | top-5: {rf_topk[2]:.4f}")

    # ---- Plots -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 3a: Feature importance bar (RF)
    ax = axes[0]
    sorted_names = [feature_names[i] for i in rf_sorted]
    sorted_vals  = rf_imp[rf_sorted]
    dt_vals_sorted = dt_imp[rf_sorted]
    x = np.arange(len(feature_names))
    w = 0.35
    ax.barh(x + w/2, sorted_vals,    w, label='RF',  color=COLORS[3], alpha=0.85)
    ax.barh(x - w/2, dt_vals_sorted, w, label='DT',  color=COLORS[0], alpha=0.85)
    ax.set_yticks(x); ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importances (DT vs RF)')
    ax.legend(); ax.invert_yaxis()

    # 3b: Performance vs k features
    ax = axes[1]
    ax.plot(k_values, dt_topk, 'o-', label='Custom DT', color=COLORS[0])
    ax.plot(k_values, rf_topk, 's--', label='Custom RF', color=COLORS[3])
    ax.set_xlabel('Number of Top Features (k)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Number of Features')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(k_values)

    # 3c: Correlation heatmap
    ax = axes[2]
    all_X = np.vstack([X_train, X_test])
    corr = np.corrcoef(all_X.T)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title('Feature Correlation Heatmap')
    plt.colorbar(im, ax=ax)

    plt.suptitle('Experiment 3: Feature Importance — Wine Quality Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'feature_importance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n  [saved] {path}")


# 4.  ADDITIONAL — BIAS-VARIANCE & LEARNING CURVES


def additional_bias_variance(X_train, X_test, y_train, y_test):
   
    print("\n" + "=" * 60)
    print("ADDITIONAL: Bias-Variance Analysis")
    print("=" * 60)

    depths = list(range(1, 21))
    dt_train, dt_test, rf_test_list = [], [], []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        dt.fit(X_train, y_train)
        dt_train.append(accuracy_score(y_train, dt.predict(X_train)))
        dt_test.append(accuracy_score(y_test,  dt.predict(X_test)))

    # RF at each depth
    for d in depths:
        rf = RandomForestClassifier(n_estimators=50, max_depth=d,
                                    random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        rf_test_list.append(accuracy_score(y_test, rf.predict(X_test)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depths, dt_train, 'o-', label='DT Train',  color=COLORS[0], lw=2)
    ax.plot(depths, dt_test,  's--', label='DT Test',   color=COLORS[1], lw=2)
    ax.plot(depths, rf_test_list, '^-.',
            label='RF Test (50 trees)', color=COLORS[2], lw=2)
    ax.fill_between(depths, dt_train, dt_test, alpha=0.1, color=COLORS[0],
                    label='DT overfit gap')
    ax.set_xlabel('Tree Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Bias-Variance Tradeoff: DT vs RF across Depths')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'bias_variance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")



# 5.  ADDITIONAL — COMPUTATIONAL COMPLEXITY


def additional_complexity(X_train, y_train):
    
    print("\n" + "=" * 60)
    print("ADDITIONAL: Computational Complexity")
    print("=" * 60)

    # a) Dataset size
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    sizes, dt_times, rf_times = [], [], []
    for frac in fractions:
        n = max(50, int(len(X_train) * frac))
        sizes.append(n)
        Xs, ys = X_train[:n], y_train[:n]

        t0 = time.time()
        DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE).fit(Xs, ys)
        dt_times.append(time.time() - t0)

        t0 = time.time()
        RandomForestClassifier(n_estimators=20, max_depth=10,
                               random_state=RANDOM_STATE).fit(Xs, ys)
        rf_times.append(time.time() - t0)

    # b) Depth
    depth_list = [2, 4, 6, 8, 10, 15, 20, None]
    depth_labels = [str(d) if d is not None else 'None' for d in depth_list]
    dt_depth_times = []
    for d in depth_list:
        t0 = time.time()
        DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(
            X_train, y_train)
        dt_depth_times.append(time.time() - t0)

    # c) RF n_estimators
    n_tree_list = [1, 5, 10, 25, 50, 100, 200]
    rf_ntree_times = []
    for n in n_tree_list:
        t0 = time.time()
        RandomForestClassifier(n_estimators=n, max_depth=10,
                               random_state=RANDOM_STATE).fit(X_train, y_train)
        rf_ntree_times.append(time.time() - t0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(sizes, dt_times, 'o-', label='DT', color=COLORS[0])
    ax.plot(sizes, rf_times, 's--', label='RF (20 trees)', color=COLORS[2])
    ax.set_xlabel('Training Set Size'); ax.set_ylabel('Time (s)')
    ax.set_title('Training Time vs Dataset Size')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(depth_labels, dt_depth_times, 'o-', color=COLORS[1])
    ax.set_xlabel('max_depth'); ax.set_ylabel('Time (s)')
    ax.set_title('DT Training Time vs Depth')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(n_tree_list, rf_ntree_times, 's--', color=COLORS[3])
    ax.set_xlabel('n_estimators'); ax.set_ylabel('Time (s)')
    ax.set_title('RF Training Time vs n_estimators')
    ax.grid(alpha=0.3)

    plt.suptitle('Computational Complexity Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'complexity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")



# 6.  ADDITIONAL — DECISION BOUNDARY (PCA 2D)


def additional_decision_boundary(X_train, X_test, y_train, y_test):
    
    print("\n" + "=" * 60)
    print("ADDITIONAL: Decision Boundary (PCA 2D)")
    print("=" * 60)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_tr2 = pca.fit_transform(X_train)
    X_te2 = pca.transform(X_test)

    h = 0.05
    x_min, x_max = X_tr2[:, 0].min() - 0.5, X_tr2[:, 0].max() + 0.5
    y_min, y_max = X_tr2[:, 1].min() - 0.5, X_tr2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_list = [
        ('Decision Tree (depth=4)',
         DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
        ('Random Forest (50 trees)',
         RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
    ]
    for ax, (title, model) in zip(axes, model_list):
        model.fit(X_tr2, y_train)
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap='coolwarm')
        ax.scatter(X_te2[y_test == 0, 0], X_te2[y_test == 0, 1],
                   c=COLORS[1], label='Low quality', edgecolors='k', s=30, alpha=0.7)
        ax.scatter(X_te2[y_test == 1, 0], X_te2[y_test == 1, 1],
                   c=COLORS[0], label='High quality', edgecolors='k', s=30, alpha=0.7)
        acc = accuracy_score(y_test, model.predict(X_te2))
        ax.set_title(f'{title}\nTest Acc = {acc:.3f}')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(fontsize=8)

    plt.suptitle('Decision Boundaries in PCA Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'decision_boundary.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {path}")



# MAIN


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  YSU CS2020 — Assignment 3 Experiments")
    print("=" * 60 + "\n")

    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_wine_quality()

    # Run all experiments
    results = experiment1_model_comparison(X_train, X_test, y_train, y_test,
                                           feature_names)
    experiment2_hyperparameter_tuning(X_train, X_test, y_train, y_test)
    experiment3_feature_importance(X_train, X_test, y_train, y_test, feature_names)
    additional_bias_variance(X_train, X_test, y_train, y_test)
    additional_complexity(X_train, y_train)
    additional_decision_boundary(X_train, X_test, y_train, y_test)

    print("\n✅  All experiments complete. Check the ../figures/ directory.")
