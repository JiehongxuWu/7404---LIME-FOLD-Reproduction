import numpy as np


class MDLPDiscretizer:
    """
    Fayyad-Irani MDLPC (Minimum Description Length Principle for discretization).

    该实现与论文引用的离散化思想一致（Fayyad & Irani, 1993），用于将连续数值特征
    自动离散成区间(bin)，以便后续 LIME-FOLD 的可解释语言使用。

    支持 per-feature min_depth：
      min_depth 可以是整数（所有特征统一）
      也可以是列表（每个特征单独设置），长度需与特征数一致。

    min_samples=10：子集样本数不足时禁止强制切分，避免垃圾切分点。
    """

    def __init__(self, min_depth=2, max_depth=6, min_samples=10):
        self.min_depth = min_depth  # int 或 list[int]
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.cut_points = {}

    def _get_min_depth(self, feature_idx):
        """统一处理 min_depth 为整数或列表两种情况。"""
        if isinstance(self.min_depth, (list, tuple, np.ndarray)):
            return int(self.min_depth[feature_idx])
        return int(self.min_depth)

    def _entropy(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-12))

    def _boundary_points(self, x_sorted, y_sorted):
        """
        对每对相邻不同特征值，
        取两侧所有样本类标签集合的并集，含多个类则为候选切分点。
        """
        unique_vals = np.unique(x_sorted)
        candidates = []
        for i in range(1, len(unique_vals)):
            v_prev, v_curr = unique_vals[i - 1], unique_vals[i]
            merged = np.union1d(y_sorted[x_sorted == v_prev], y_sorted[x_sorted == v_curr])
            if len(merged) > 1:
                idx = int(np.searchsorted(x_sorted, v_curr, side="left"))
                candidates.append(idx)
        return candidates

    def _mdlpc_criterion(self, y, y_l, y_r, gain):
        n = len(y)
        if n <= 1:
            return False
        k = len(np.unique(y))
        k_l = len(np.unique(y_l))
        k_r = len(np.unique(y_r))
        h, h_l, h_r = self._entropy(y), self._entropy(y_l), self._entropy(y_r)
        delta = np.log2(3.0**k - 2.0) - (k * h - k_l * h_l - k_r * h_r)
        threshold = (np.log2(max(n - 1, 1)) + delta) / n
        return gain > threshold

    def _get_cut_points(self, x, y, depth=0, feature_min_depth=2):
        n = len(y)
        if n < 2 or depth >= self.max_depth:
            return []

        order = np.argsort(x, kind="stable")
        x_s, y_s = x[order], y[order]

        candidates = self._boundary_points(x_s, y_s)
        if not candidates:
            return []

        h_s = self._entropy(y_s)
        best_gain, best_idx = -1.0, None
        for idx in candidates:
            y_l, y_r = y_s[:idx], y_s[idx:]
            if len(y_l) == 0 or len(y_r) == 0:
                continue
            gain = h_s - (len(y_l) / n) * self._entropy(y_l) - (len(y_r) / n) * self._entropy(y_r)
            if gain > best_gain:
                best_gain, best_idx = gain, idx

        if best_idx is None or best_gain <= 0:
            return []

        cut_val = (x_s[best_idx - 1] + x_s[best_idx]) / 2.0
        y_l = y_s[:best_idx]
        y_r = y_s[best_idx:]

        # 强制切分条件（可选）：在最初若干层允许切分以生成更细的区间语言
        force_split = (
            depth < feature_min_depth and len(y_l) >= self.min_samples and len(y_r) >= self.min_samples
        )

        if self._mdlpc_criterion(y_s, y_l, y_r, best_gain) or force_split:
            return (
                self._get_cut_points(x_s[:best_idx], y_l, depth + 1, feature_min_depth)
                + [cut_val]
                + self._get_cut_points(x_s[best_idx:], y_r, depth + 1, feature_min_depth)
            )
        return []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.cut_points = {}
        for i in range(X.shape[1]):
            fmd = self._get_min_depth(i)
            cuts = self._get_cut_points(X[:, i], y, depth=0, feature_min_depth=fmd)
            self.cut_points[i] = sorted(set(cuts))
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X_out = np.zeros(X.shape, dtype=int)
        for i, cuts in self.cut_points.items():
            if cuts:
                X_out[:, i] = np.digitize(X[:, i], cuts)
        return X_out

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def print_cut_points(self, feature_names=None):
        for i, cuts in self.cut_points.items():
            name = feature_names[i] if feature_names else f"feature_{i}"
            print(
                f"  {name:15s}: {len(cuts)} cuts → {len(cuts)+1} bins "
                f"| {[round(c, 1) for c in cuts]}"
            )
