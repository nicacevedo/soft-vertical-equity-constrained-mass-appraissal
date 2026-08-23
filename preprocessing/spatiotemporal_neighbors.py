from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


_EARTH_RADIUS_KM = 6371.0088


class SpatioTemporalKernelTargetNeighbors(BaseEstimator, TransformerMixin):
    """
    Kernel-smoothed target-neighbor transformer for comparable-sales features.

    The class extends the original GeoKernelTargetNeighbors idea in three small
    ways:
      1. optional numeric-feature similarity in the neighbor ranking/kernel,
      2. optional global log-price time-trend adjustment of neighbor targets,
      3. optional time-decay multiplier for older comparable sales.

    Output columns
    --------------
    For each row, the transformer returns k weighted neighbor target columns:

        {prefix}_neighbor_1_kernel_adjusted_target
        ...
        {prefix}_neighbor_k_kernel_adjusted_target

    and, if include_aggregate=True:

        {prefix}_local_kernel_adjusted_target_mean

    The aggregate is the row-wise sum of the k weighted adjusted neighbor targets.
    If normalize_weights=True, this is a weighted mean. If normalize_weights=False,
    it is an unnormalized weighted sum.

    With ``include_diagnostics=True``,
    ``{prefix}_eligible_pool_size_after_caps`` reports the number of candidates
    left after hard geographic/time caps, temporal rule, and self exclusion, but
    before composite-distance ranking.  A row needs at least ``k`` such candidates
    to receive comparable features.

    Finite eligibility caps
    -----------------------
    ``max_distance_km`` and ``max_time_distance_days`` are hard eligibility
    restrictions, not kernel bandwidths.  When either cap is supplied, candidate
    retrieval is exact: every fitted sale satisfying the geographic, temporal,
    categorical, and self-exclusion restrictions is considered before the
    composite geographic/time/feature distance is ranked.  In particular,
    ``candidate_multiplier`` and ``max_candidates`` are deliberately ignored in
    this capped path.  The fitted ``candidate_retrieval_metadata_`` attribute
    records which retrieval contract was used.

    Leakage controls
    ----------------
    - The transformer is fitted only on X_train and y_train.
    - transform(X_test) never uses y_test.
    - exclude_self=True removes the same training row when transforming X_train.
    - neighbor_time_rule="past" restricts neighbors to fitted training rows with
      train_date < query_date. This is the most conservative option for temporal
      feature construction.
    - time_trend_fit_mode="causal_prior" fits a target-derived linear trend using
      only fitted rows strictly before each query date, including train transforms.
    """

    def __init__(
        self,
        k: int = 5,
        lat_col: str = "loc_latitude",
        lon_col: str = "loc_longitude",
        date_col: Optional[str] = None,
        # kernel / distance controls
        kernel: str = "gaussian",
        bandwidth: Any = "adaptive",
        bandwidth_scale: float = 1.0,
        min_bandwidth_km: float = 1e-6,
        normalize_weights: bool = True,
        geo_weight: float = 1.0,
        # spatial + feature-distance controls
        max_distance_km: Optional[float] = None,
        use_feature_distance: bool = False,
        numeric_feature_cols: Optional[list[str]] = None,
        feature_alpha: float = 1.0,
        feature_bandwidth: float = 1.0,
        feature_scaler: str = "robust",  # "robust" or "standard"
        feature_missing: str = "median",  # currently only "median"
        candidate_multiplier: int = 10,
        min_candidates: Optional[int] = None,
        max_candidates: Optional[int] = None,
        batch_query_size: Optional[int] = None,
        full_pool_batch_size: int = 32,
        n_jobs: Optional[int] = None,
        # temporal controls
        neighbor_time_rule: str = "none",  # "none", "past", "past_or_same_day"
        prediction_date: Optional[Any] = None,
        use_time_trend: bool = False,
        time_trend: str = "linear",  # "linear" or "monthly_mean"
        time_trend_fit_mode: str = "global",  # "global" or "causal_prior"
        use_time_decay: bool = False,
        time_weight: float = 0.0,
        time_bandwidth_days: Optional[float] = None,
        time_decay_half_life_days: float = 365.25,
        max_time_distance_days: Optional[float] = None,
        # target controls
        target_transform: Optional[str] = None,  # None or "log"
        # output controls
        include_aggregate: bool = True,
        include_diagnostics: bool = False,
        feature_prefix: str = "geo",
        # original categorical filtering controls
        categorical_filter_roots: Optional[Any] = None,
        one_hot_sep: str = "_",
        one_hot_threshold: float = 0.5,
        filter_fallback: str = "global",  # "global" or "raise"
        handle_missing_category: bool = True,
        missing_category_label: str = "__MISSING__",
        allow_binary_filter: bool = True,
        # insufficiency / self controls
        exclude_self: bool = True,
        insufficient_neighbors: str = "nan",  # "nan" or "raise"
    ):
        self.k = k
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.date_col = date_col

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.min_bandwidth_km = min_bandwidth_km
        self.normalize_weights = normalize_weights
        self.geo_weight = geo_weight
        self.max_distance_km = max_distance_km

        self.use_feature_distance = use_feature_distance
        self.numeric_feature_cols = numeric_feature_cols
        self.feature_alpha = feature_alpha
        self.feature_bandwidth = feature_bandwidth
        self.feature_scaler = feature_scaler
        self.feature_missing = feature_missing
        self.candidate_multiplier = candidate_multiplier
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.batch_query_size = batch_query_size
        self.full_pool_batch_size = full_pool_batch_size
        self.n_jobs = n_jobs

        self.neighbor_time_rule = neighbor_time_rule
        self.prediction_date = prediction_date
        self.use_time_trend = use_time_trend
        self.time_trend = time_trend
        self.time_trend_fit_mode = time_trend_fit_mode
        self.use_time_decay = use_time_decay
        self.time_weight = time_weight
        self.time_bandwidth_days = time_bandwidth_days
        self.time_decay_half_life_days = time_decay_half_life_days
        self.max_time_distance_days = max_time_distance_days

        self.target_transform = target_transform
        self.include_aggregate = include_aggregate
        self.include_diagnostics = include_diagnostics
        self.feature_prefix = feature_prefix

        self.categorical_filter_roots = categorical_filter_roots
        self.one_hot_sep = one_hot_sep
        self.one_hot_threshold = one_hot_threshold
        self.filter_fallback = filter_fallback
        self.handle_missing_category = handle_missing_category
        self.missing_category_label = missing_category_label
        self.allow_binary_filter = allow_binary_filter

        self.exclude_self = exclude_self
        self.insufficient_neighbors = insufficient_neighbors

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Any):
        X_train = X.copy()
        y_train = self._prepare_y(y, X_train.index)

        self._validate_init_params()
        self._setup_filter_columns(X_train)
        self._validate_X(X_train, require_filter_cols=True)
        self.candidate_retrieval_metadata_ = self._candidate_retrieval_metadata()
        self.candidate_retrieval_ = str(self.candidate_retrieval_metadata_["mode"])
        self.candidate_multiplier_used_ = bool(
            self.candidate_retrieval_metadata_["candidate_multiplier_used"]
        )

        n_train = int(len(X_train))
        if n_train == 0:
            raise ValueError("X_train is empty.")
        if int(self.k) < 1:
            raise ValueError("k must be >= 1.")
        if bool(self.exclude_self) and int(self.k) >= n_train and self.insufficient_neighbors == "raise":
            raise ValueError("With exclude_self=True, k must be < len(X_train).")
        if not bool(self.exclude_self) and int(self.k) > n_train and self.insufficient_neighbors == "raise":
            raise ValueError("With exclude_self=False, k must be <= len(X_train).")
        if not X_train.index.is_unique:
            raise ValueError(
                "X_train.index must be unique. This is needed to safely exclude "
                "the training row itself when transforming X_train."
            )

        self.train_index_ = X_train.index.copy()
        self.index_to_train_pos_ = {idx: pos for pos, idx in enumerate(self.train_index_)}
        self.y_train_ = y_train.to_numpy(dtype=float)
        self.global_train_pos_ = np.arange(n_train, dtype=int)

        self.train_coords_rad_ = np.radians(
            X_train[[self.lat_col, self.lon_col]].to_numpy(dtype=float)
        )
        self.global_nn_ = NearestNeighbors(
            metric="haversine", algorithm="ball_tree", n_jobs=self.n_jobs,
        )
        self.global_nn_.fit(self.train_coords_rad_)

        self.train_dates_ = self._resolve_dates(X_train, fitting=True)
        self.train_day_values_ = None
        if self.train_dates_ is not None:
            self.train_day_values_ = self._dates_to_day_values(self.train_dates_)

        self._fit_feature_scaler(X_train)
        self._fit_time_trend()
        self._fit_group_neighbor_models(X_train)
        return self

    def _has_finite_eligibility_caps(self) -> bool:
        """Whether candidate retrieval must be exact rather than approximate."""
        return self.max_distance_km is not None or self.max_time_distance_days is not None

    def _exact_temporal_bin_days(self) -> float | None:
        """Bound radius-query memory while preserving the exact time window.

        A bin only controls the temporary tree used for a batch of queries.  Its
        candidate pool is the union of all time-eligible rows for that bin, and
        each individual query is still filtered against its exact timestamp.
        """
        if self.max_time_distance_days is None:
            return None
        return float(min(90.0, max(30.0, float(self.max_time_distance_days) / 4.0)))

    def _candidate_retrieval_metadata(self) -> dict[str, Any]:
        """Describe the fitted retrieval contract for artifact provenance."""
        exact = self._has_finite_eligibility_caps()
        if not exact:
            mode = "geographic_knn_candidate_pool"
        elif self.max_distance_km is not None and self.max_time_distance_days is not None:
            mode = "exact_radius_then_temporal_filter"
        elif self.max_distance_km is not None:
            mode = "exact_radius"
        else:
            mode = "exact_temporal_pool"
        return {
            "mode": mode,
            "exact": bool(exact),
            "max_distance_km": self.max_distance_km,
            "max_time_distance_days": self.max_time_distance_days,
            "candidate_multiplier_used": bool(not exact),
            "max_candidates_used": bool(not exact),
            "temporal_bin_days": self._exact_temporal_bin_days(),
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_is_fitted()
        X_query = X.copy()
        self._validate_X(X_query, require_filter_cols=True)

        n_query = int(len(X_query))
        query_coords_rad = np.radians(
            X_query[[self.lat_col, self.lon_col]].to_numpy(dtype=float)
        )
        query_features_scaled = self._transform_numeric_features(X_query)
        query_dates = self._resolve_dates(X_query, fitting=False)
        query_day_values = None
        if query_dates is not None:
            query_day_values = self._dates_to_day_values(query_dates)

        if self.filter_roots_:
            query_filter_keys = self._make_filter_keys(X_query)
        else:
            query_filter_keys = [None] * n_query

        neighbor_pos = np.full((n_query, int(self.k)), -1, dtype=int)
        distances_km = np.full((n_query, int(self.k)), np.nan, dtype=float)
        feature_distances = np.full((n_query, int(self.k)), np.nan, dtype=float)
        composite_u = np.full((n_query, int(self.k)), np.nan, dtype=float)
        used_filtered_pool = np.zeros(n_query, dtype=int)
        pool_size_used = np.zeros(n_query, dtype=int)
        eligible_pool_size_after_caps = np.zeros(n_query, dtype=int)
        n_valid_neighbors = np.zeros(n_query, dtype=int)

        if self.batch_query_size is not None and not self.filter_roots_:
            (
                distances_km,
                feature_distances,
                composite_u,
                neighbor_pos,
                used_filtered_pool,
                pool_size_used,
                eligible_pool_size_after_caps,
                n_valid_neighbors,
            ) = self._query_neighbors_without_filters_batch(
                query_coords_rad=query_coords_rad,
                query_features_scaled=query_features_scaled,
                query_dates=query_dates,
                query_day_values=query_day_values,
                query_index=X_query.index,
            )
        else:
            for row_idx, original_index in enumerate(X_query.index):
                query_date = None if query_dates is None else query_dates[row_idx]
                query_feat = None if query_features_scaled is None else query_features_scaled[row_idx]

                result = self._query_neighbors_one_row(
                    query_coord_rad=query_coords_rad[row_idx],
                    query_feature_scaled=query_feat,
                    query_index=original_index,
                    query_date=query_date,
                    filter_key=query_filter_keys[row_idx],
                )

                if result is None:
                    if self.insufficient_neighbors == "raise":
                        raise RuntimeError(
                            "Could not find enough valid neighbors for at least one row. "
                            "Reduce k, use coarser categorical filters, relax neighbor_time_rule, "
                            "or set insufficient_neighbors='nan'."
                        )
                    continue

                d_row, f_row, u_row, n_row, used_filter, pool_size, eligible_pool_size = result
                eligible_pool_size_after_caps[row_idx] = int(eligible_pool_size)
                if d_row is None:
                    if self.insufficient_neighbors == "raise":
                        raise RuntimeError(
                            "Could not find enough valid neighbors for at least one row. "
                            "Reduce k, use coarser categorical filters, relax eligibility caps, "
                            "or set insufficient_neighbors='nan'."
                        )
                    continue
                m = min(int(self.k), len(n_row))
                distances_km[row_idx, :m] = d_row[:m]
                feature_distances[row_idx, :m] = f_row[:m]
                composite_u[row_idx, :m] = u_row[:m]
                neighbor_pos[row_idx, :m] = n_row[:m]
                used_filtered_pool[row_idx] = int(used_filter)
                pool_size_used[row_idx] = int(pool_size)
                n_valid_neighbors[row_idx] = int(m)

        weights = self._kernel_values_from_u(composite_u)

        weights = np.where(neighbor_pos >= 0, weights, 0.0)

        if bool(self.normalize_weights):
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = np.divide(
                weights,
                row_sums,
                out=np.zeros_like(weights),
                where=row_sums > 0,
            )

        adjusted_targets = self._adjust_neighbor_targets_to_query_date(neighbor_pos, query_day_values)
        weighted_targets = weights * adjusted_targets
        weighted_targets = np.where(neighbor_pos >= 0, weighted_targets, np.nan)

        out = pd.DataFrame(index=X_query.index)
        for j in range(int(self.k)):
            out[f"{self.feature_prefix}_neighbor_{j + 1}_kernel_adjusted_target"] = weighted_targets[:, j]

        if bool(self.include_aggregate):
            local_mean = np.nansum(weighted_targets, axis=1)
            no_neighbors = n_valid_neighbors == 0
            local_mean[no_neighbors] = np.nan
            out[f"{self.feature_prefix}_local_kernel_adjusted_target_mean"] = local_mean

        if bool(self.include_diagnostics):
            out[f"{self.feature_prefix}_n_valid_neighbors"] = n_valid_neighbors
            out[f"{self.feature_prefix}_used_filtered_pool"] = used_filtered_pool
            out[f"{self.feature_prefix}_neighbor_pool_size"] = pool_size_used
            out[f"{self.feature_prefix}_eligible_pool_size_after_caps"] = eligible_pool_size_after_caps
            out[f"{self.feature_prefix}_weight_sum_before_norm"] = np.where(
                n_valid_neighbors > 0,
                np.nansum(self._kernel_values_from_u(composite_u), axis=1),
                np.nan,
            )
            for j in range(int(self.k)):
                out[f"{self.feature_prefix}_neighbor_{j + 1}_weight"] = weights[:, j]
                out[f"{self.feature_prefix}_neighbor_{j + 1}_distance_km"] = distances_km[:, j]
                out[f"{self.feature_prefix}_neighbor_{j + 1}_feature_distance"] = feature_distances[:, j]
                out[f"{self.feature_prefix}_neighbor_{j + 1}_train_pos"] = np.where(
                    neighbor_pos[:, j] >= 0,
                    neighbor_pos[:, j],
                    np.nan,
                )

        return out

    def fit_transform_train_test(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: Any):
        """Convenience method matching the original class."""
        self.fit(X_train, y_train)
        train_features = self.transform(X_train)
        test_features = self.transform(X_test)
        return train_features, test_features

    # ------------------------------------------------------------------
    # Neighbor querying
    # ------------------------------------------------------------------
    def _query_neighbors_without_filters_batch(
        self,
        *,
        query_coords_rad: np.ndarray,
        query_features_scaled: Optional[np.ndarray],
        query_dates: Optional[pd.DatetimeIndex],
        query_day_values: Optional[np.ndarray],
        query_index: pd.Index,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch the global-pool path without changing its per-row ranking rule.

        The usual path supports categorical pools, whose varying membership requires
        row-wise tree queries.  This path is deliberately narrower: it is used only
        when there is one global pool.  With no hard eligibility caps it retains the
        historical geographic-kNN candidate optimization.  With either hard cap it
        switches to exact bounded radius/temporal-pool retrieval; it never applies
        ``candidate_multiplier`` in that branch.
        """
        if self._has_finite_eligibility_caps():
            return self._query_neighbors_with_exact_caps_batch(
                query_coords_rad=query_coords_rad,
                query_features_scaled=query_features_scaled,
                query_dates=query_dates,
                query_day_values=query_day_values,
                query_index=query_index,
            )

        n_raw = self._candidate_count(pool_size=len(self.global_train_pos_), extra_for_self=1)
        result = self._query_global_pool_in_batches(
            query_coords_rad=query_coords_rad,
            query_features_scaled=query_features_scaled,
            query_day_values=query_day_values,
            query_index=query_index,
            n_neighbors=n_raw,
            batch_size=int(self.batch_query_size),
        )
        fallback_rows = np.flatnonzero(result[-1] < int(self.k))
        if len(fallback_rows) == 0 or n_raw >= len(self.global_train_pos_):
            return result

        fallback = self._query_global_pool_in_batches(
            query_coords_rad=query_coords_rad[fallback_rows],
            query_features_scaled=(
                None if query_features_scaled is None else query_features_scaled[fallback_rows]
            ),
            query_day_values=(
                None if query_day_values is None else query_day_values[fallback_rows]
            ),
            query_index=query_index[fallback_rows],
            n_neighbors=len(self.global_train_pos_),
            batch_size=int(self.full_pool_batch_size),
        )
        for target, replacement in zip(result, fallback):
            target[fallback_rows] = replacement
        return result

    def _query_global_pool_in_batches(
        self,
        *,
        query_coords_rad: np.ndarray,
        query_features_scaled: Optional[np.ndarray],
        query_day_values: Optional[np.ndarray],
        query_index: pd.Index,
        n_neighbors: int,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Query and rank a fixed number of candidates from the global pool.

        ``n_neighbors`` is either the normal candidate count or the complete pool
        for rows whose temporal filter exhausted that first draw.  Keeping both
        paths batched preserves the row-wise result while avoiding one BallTree
        call per fallback row.
        """
        n_query = len(query_coords_rad)
        k = int(self.k)
        neighbor_pos = np.full((n_query, k), -1, dtype=int)
        distances_km = np.full((n_query, k), np.nan, dtype=float)
        feature_distances = np.full((n_query, k), np.nan, dtype=float)
        composite_u = np.full((n_query, k), np.nan, dtype=float)
        used_filtered_pool = np.zeros(n_query, dtype=int)
        pool_size_used = np.zeros(n_query, dtype=int)
        eligible_pool_size_after_caps = np.zeros(n_query, dtype=int)
        n_valid_neighbors = np.zeros(n_query, dtype=int)

        for start in range(0, n_query, batch_size):
            end = min(start + batch_size, n_query)
            distance_rad, local_idx = self.global_nn_.kneighbors(
                query_coords_rad[start:end], n_neighbors=n_neighbors,
            )
            raw_distances = distance_rad.astype(float) * _EARTH_RADIUS_KM
            raw_positions = self.global_train_pos_[local_idx].astype(int)
            valid = np.isfinite(raw_distances)

            if bool(self.exclude_self):
                query_train_pos = np.asarray(
                    [self.index_to_train_pos_.get(index, -1) for index in query_index[start:end]], dtype=int,
                )
                has_train_pos = query_train_pos >= 0
                if has_train_pos.any():
                    same_coord = np.zeros(end - start, dtype=bool)
                    same_coord[has_train_pos] = np.all(
                        np.isclose(
                            query_coords_rad[start:end][has_train_pos],
                            self.train_coords_rad_[query_train_pos[has_train_pos]],
                            rtol=0.0, atol=1e-12,
                        ),
                        axis=1,
                    )
                    valid &= ~(
                        raw_positions == query_train_pos[:, None]
                    ) | ~same_coord[:, None]

            if self.neighbor_time_rule != "none":
                if query_day_values is None or self.train_day_values_ is None:
                    raise ValueError("neighbor_time_rule requires date_col in fit/transform or prediction_date.")
                candidate_days = self.train_day_values_[raw_positions]
                if self.neighbor_time_rule == "past":
                    valid &= candidate_days < query_day_values[start:end, None]
                else:  # ``_validate_init_params`` has already limited the remaining case.
                    valid &= candidate_days <= query_day_values[start:end, None]

            eligible_pool_size_after_caps[start:end] = valid.sum(axis=1)
            enough_candidates = eligible_pool_size_after_caps[start:end] >= k
            valid &= enough_candidates[:, None]

            valid_distances = np.where(valid, raw_distances, np.nan)
            bandwidth = np.full(end - start, float(self.min_bandwidth_km), dtype=float)
            if enough_candidates.any():
                bandwidth[enough_candidates] = np.nanmax(
                    valid_distances[enough_candidates], axis=1,
                )
            bandwidth = np.where(
                np.isfinite(bandwidth),
                np.maximum(bandwidth * float(self.bandwidth_scale), float(self.min_bandwidth_km)),
                float(self.min_bandwidth_km),
            ) if self.bandwidth == "adaptive" else np.full(end - start, float(self.bandwidth))
            u = float(self.geo_weight) * raw_distances / bandwidth[:, None]

            if bool(self.use_feature_distance):
                if query_features_scaled is None:
                    raise RuntimeError("query_features_scaled is required when use_feature_distance=True.")
                feature_distance = np.sqrt(np.mean(
                    (self.train_features_scaled_[raw_positions] - query_features_scaled[start:end, None, :]) ** 2,
                    axis=2,
                ))
                u += float(self.feature_alpha) * feature_distance / float(self.feature_bandwidth)
            else:
                feature_distance = np.zeros_like(raw_distances)

            if bool(self.use_time_decay):
                if query_day_values is None or self.train_day_values_ is None:
                    raise ValueError("use_time_decay=True requires date_col in transform or prediction_date.")
                time_bandwidth = self.time_bandwidth_days
                if time_bandwidth is None:
                    time_bandwidth = self.time_decay_half_life_days
                u += float(self.time_weight) * np.abs(
                    query_day_values[start:end, None] - self.train_day_values_[raw_positions]
                ) / float(time_bandwidth)

            u[~valid] = np.inf
            order = self._stable_top_k_indices(u, k)
            selected_valid = np.take_along_axis(valid, order, axis=1)
            selected_positions = np.take_along_axis(raw_positions, order, axis=1)
            selected_distances = np.take_along_axis(raw_distances, order, axis=1)
            selected_features = np.take_along_axis(feature_distance, order, axis=1)
            selected_u = np.take_along_axis(u, order, axis=1)
            selected_positions[~selected_valid] = -1
            selected_distances[~selected_valid] = np.nan
            selected_features[~selected_valid] = np.nan
            selected_u[~selected_valid] = np.nan
            neighbor_pos[start:end] = selected_positions
            distances_km[start:end] = selected_distances
            feature_distances[start:end] = selected_features
            composite_u[start:end] = selected_u
            n_valid_neighbors[start:end] = selected_valid.sum(axis=1)
            pool_size_used[start + np.flatnonzero(enough_candidates)] = len(self.global_train_pos_)

        return (
            distances_km, feature_distances, composite_u, neighbor_pos,
            used_filtered_pool, pool_size_used, eligible_pool_size_after_caps, n_valid_neighbors,
        )

    def _query_neighbors_with_exact_caps_batch(
        self,
        *,
        query_coords_rad: np.ndarray,
        query_features_scaled: Optional[np.ndarray],
        query_dates: Optional[pd.DatetimeIndex],
        query_day_values: Optional[np.ndarray],
        query_index: pd.Index,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exactly rank every hard-cap-eligible global candidate in bounded batches.

        When a time cap is present, query rows are placed in short temporal bins.
        A temporary BallTree is fitted to the *union* of all rows that could satisfy
        the time cap for that bin, then each row is filtered again against its exact
        timestamp.  This is materially smaller than radius-searching the complete
        historical pool in dense counties while retaining identical eligibility
        semantics to a full-pool search.
        """
        n_query = len(query_coords_rad)
        k = int(self.k)
        neighbor_pos = np.full((n_query, k), -1, dtype=int)
        distances_km = np.full((n_query, k), np.nan, dtype=float)
        feature_distances = np.full((n_query, k), np.nan, dtype=float)
        composite_u = np.full((n_query, k), np.nan, dtype=float)
        used_filtered_pool = np.zeros(n_query, dtype=int)
        pool_size_used = np.zeros(n_query, dtype=int)
        eligible_pool_size_after_caps = np.zeros(n_query, dtype=int)
        n_valid_neighbors = np.zeros(n_query, dtype=int)

        result_arrays = (
            distances_km,
            feature_distances,
            composite_u,
            neighbor_pos,
            pool_size_used,
            eligible_pool_size_after_caps,
            n_valid_neighbors,
        )
        all_rows = np.arange(n_query, dtype=int)
        if self.max_time_distance_days is None:
            self._fill_exact_capped_pool_batches(
                query_rows=all_rows,
                nn=self.global_nn_,
                pool_train_pos=self.global_train_pos_,
                query_coords_rad=query_coords_rad,
                query_features_scaled=query_features_scaled,
                query_dates=query_dates,
                query_index=query_index,
                result_arrays=result_arrays,
            )
        else:
            if query_day_values is None or self.train_day_values_ is None:
                raise ValueError(
                    "max_time_distance_days requires date_col in fit/transform or prediction_date."
                )
            bin_width = float(self._exact_temporal_bin_days())
            bin_ids = np.floor(np.asarray(query_day_values, dtype=float) / bin_width).astype(np.int64)
            for bin_id in np.unique(bin_ids):
                query_rows = np.flatnonzero(bin_ids == bin_id)
                pool_train_pos = self._temporal_union_pool_positions(
                    query_day_values=np.asarray(query_day_values, dtype=float)[query_rows],
                )
                if len(pool_train_pos) == 0:
                    continue
                if len(pool_train_pos) == len(self.global_train_pos_):
                    nn = self.global_nn_
                else:
                    nn = NearestNeighbors(
                        metric="haversine", algorithm="ball_tree", n_jobs=self.n_jobs,
                    )
                    nn.fit(self.train_coords_rad_[pool_train_pos])
                self._fill_exact_capped_pool_batches(
                    query_rows=query_rows,
                    nn=nn,
                    pool_train_pos=pool_train_pos,
                    query_coords_rad=query_coords_rad,
                    query_features_scaled=query_features_scaled,
                    query_dates=query_dates,
                    query_index=query_index,
                    result_arrays=result_arrays,
                )

        return (
            distances_km,
            feature_distances,
            composite_u,
            neighbor_pos,
            used_filtered_pool,
            pool_size_used,
            eligible_pool_size_after_caps,
            n_valid_neighbors,
        )

    def _temporal_union_pool_positions(self, *, query_day_values: np.ndarray) -> np.ndarray:
        """Return a superset containing every time-cap-eligible row for a query bin."""
        if self.max_time_distance_days is None or self.train_day_values_ is None:
            raise RuntimeError("Temporal union pools require fitted dates and max_time_distance_days.")
        query_day_values = np.asarray(query_day_values, dtype=float)
        start = float(np.min(query_day_values))
        end = float(np.max(query_day_values))
        cap = float(self.max_time_distance_days)
        train_days = self.train_day_values_
        if self.neighbor_time_rule == "past":
            keep = (train_days >= start - cap) & (train_days < end)
        elif self.neighbor_time_rule == "past_or_same_day":
            keep = (train_days >= start - cap) & (train_days <= end)
        else:
            keep = (train_days >= start - cap) & (train_days <= end + cap)
        return self.global_train_pos_[keep]

    def _fill_exact_capped_pool_batches(
        self,
        *,
        query_rows: np.ndarray,
        nn: NearestNeighbors,
        pool_train_pos: np.ndarray,
        query_coords_rad: np.ndarray,
        query_features_scaled: Optional[np.ndarray],
        query_dates: Optional[pd.DatetimeIndex],
        query_index: pd.Index,
        result_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Fill result arrays from every capped candidate in one fixed candidate pool."""
        (
            distances_out,
            features_out,
            composite_out,
            positions_out,
            pool_size_out,
            eligible_out,
            n_valid_out,
        ) = result_arrays
        pool_size = int(len(pool_train_pos))
        if pool_size == 0:
            return
        # ``batch_query_size`` was designed for fixed-k arrays (10k in the county
        # benchmark).  Radius results are variable length, so cap the temporary
        # object arrays independently to keep dense urban queries bounded.
        configured = 256 if self.batch_query_size is None else int(self.batch_query_size)
        radius_batch_size = min(max(configured, 1), 256)
        k = int(self.k)
        # Include the floating-point boundary in retrieval, then apply the exact
        # kilometre cap below.  This avoids dropping a sale that is exactly at the
        # declared radius solely because of the radians conversion.
        radius = (
            None
            if self.max_distance_km is None
            else np.nextafter(float(self.max_distance_km) / _EARTH_RADIUS_KM, np.inf)
        )

        for start in range(0, len(query_rows), radius_batch_size):
            rows = query_rows[start:start + radius_batch_size]
            coords = query_coords_rad[rows]
            if radius is None:
                distances_rad, local_indices = nn.kneighbors(coords, n_neighbors=pool_size)
                distance_lists = [distances_rad[number] for number in range(len(rows))]
                index_lists = [local_indices[number] for number in range(len(rows))]
            else:
                distance_lists, index_lists = nn.radius_neighbors(
                    coords,
                    radius=radius,
                    return_distance=True,
                    sort_results=True,
                )

            for local_row, row_idx in enumerate(rows):
                distances_km = np.asarray(distance_lists[local_row], dtype=float) * _EARTH_RADIUS_KM
                candidate_train_pos = pool_train_pos[np.asarray(index_lists[local_row], dtype=int)]
                query_date = None if query_dates is None else query_dates[row_idx]
                self_train_pos = self._get_self_train_pos_if_same_training_row(
                    query_index[row_idx], query_coords_rad[row_idx],
                )
                distances_km, candidate_train_pos = self._filter_candidate_positions(
                    distances_km=distances_km,
                    candidate_train_pos=candidate_train_pos,
                    self_train_pos=self_train_pos,
                    query_date=query_date,
                )
                eligible_out[row_idx] = int(len(candidate_train_pos))
                if len(candidate_train_pos) < k:
                    continue

                # Preserve the historical diagnostic meaning: the size of the
                # source training pool, rather than this implementation's
                # temporary temporal-bin superset.
                pool_size_out[row_idx] = len(self.global_train_pos_)
                query_feature = None if query_features_scaled is None else query_features_scaled[row_idx]
                feature_dist = self._candidate_feature_distance(query_feature, candidate_train_pos)
                time_dist = self._candidate_time_distance(query_date, candidate_train_pos)
                u = self._composite_normalized_distance(distances_km, feature_dist, time_dist)
                order = self._stable_top_k_1d(u, k)
                distances_out[row_idx] = distances_km[order]
                features_out[row_idx] = feature_dist[order]
                composite_out[row_idx] = u[order]
                positions_out[row_idx] = candidate_train_pos[order]
                n_valid_out[row_idx] = k

    @staticmethod
    def _stable_top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
        """Return the same stable top-k ordering without sorting a full pool.

        Full-pool temporal fallbacks can contain hundreds of thousands of rows,
        although only ``k`` are retained.  ``argpartition`` identifies the kth
        value in linear time; resolving ties by original position reproduces the
        previous stable mergesort result exactly.
        """
        if values.shape[1] <= 1_024:
            return np.argsort(values, axis=1, kind="mergesort")[:, :k]

        kth = np.partition(values, kth=k - 1, axis=1)[:, k - 1]
        out = np.empty((values.shape[0], k), dtype=int)
        for row, threshold in enumerate(kth):
            current = values[row]
            below = np.flatnonzero(current < threshold)
            equal = np.flatnonzero(current == threshold)
            chosen = np.concatenate((below, equal[:k - len(below)]))
            out[row] = chosen[np.argsort(current[chosen], kind="mergesort")]
        return out

    @staticmethod
    def _stable_top_k_1d(values: np.ndarray, k: int) -> np.ndarray:
        """Stable exact top-k for one potentially large eligible candidate pool."""
        values = np.asarray(values, dtype=float)
        if values.size <= 1_024:
            return np.argsort(values, kind="mergesort")[:k]
        threshold = np.partition(values, kth=k - 1)[k - 1]
        below = np.flatnonzero(values < threshold)
        equal = np.flatnonzero(values == threshold)
        chosen = np.concatenate((below, equal[:k - len(below)]))
        return chosen[np.argsort(values[chosen], kind="mergesort")]

    def _fit_group_neighbor_models(self, X_train: pd.DataFrame) -> None:
        self.group_models_ = {}
        self.train_filter_keys_ = None

        if not self.filter_roots_:
            return

        self.train_filter_keys_ = self._make_filter_keys(X_train)
        group_to_positions: dict[Any, list[int]] = defaultdict(list)
        for train_pos, key in enumerate(self.train_filter_keys_):
            if key is not None:
                group_to_positions[key].append(train_pos)

        for key, positions in group_to_positions.items():
            positions_arr = np.asarray(positions, dtype=int)
            nn = NearestNeighbors(metric="haversine", algorithm="ball_tree", n_jobs=self.n_jobs)
            nn.fit(self.train_coords_rad_[positions_arr])
            self.group_models_[key] = {"nn": nn, "train_pos": positions_arr}

    def _query_neighbors_one_row(
        self,
        *,
        query_coord_rad: np.ndarray,
        query_feature_scaled: Optional[np.ndarray],
        query_index: Any,
        query_date: Optional[pd.Timestamp],
        filter_key: Any,
    ):
        if self.filter_roots_ and filter_key is not None and filter_key in self.group_models_:
            group = self.group_models_[filter_key]
            result = self._query_from_pool(
                nn=group["nn"],
                pool_train_pos=group["train_pos"],
                query_coord_rad=query_coord_rad,
                query_feature_scaled=query_feature_scaled,
                query_index=query_index,
                query_date=query_date,
            )
            if result is not None and result[0] is not None:
                d_row, f_row, u_row, n_row, eligible_pool_size = result
                return (
                    d_row, f_row, u_row, n_row, True, len(group["train_pos"]), eligible_pool_size,
                )

        if self.filter_roots_ and self.filter_fallback == "raise":
            raise RuntimeError(
                "Could not find enough filtered neighbors for at least one row. "
                "Use a coarser categorical filter, reduce k, or set filter_fallback='global'."
            )

        result = self._query_from_pool(
            nn=self.global_nn_,
            pool_train_pos=self.global_train_pos_,
            query_coord_rad=query_coord_rad,
            query_feature_scaled=query_feature_scaled,
            query_index=query_index,
            query_date=query_date,
        )
        if result is None:
            return None
        d_row, f_row, u_row, n_row, eligible_pool_size = result
        return (
            d_row, f_row, u_row, n_row, False, len(self.global_train_pos_), eligible_pool_size,
        )

    def _query_from_pool(
        self,
        *,
        nn: NearestNeighbors,
        pool_train_pos: np.ndarray,
        query_coord_rad: np.ndarray,
        query_feature_scaled: Optional[np.ndarray],
        query_index: Any,
        query_date: Optional[pd.Timestamp],
    ):
        pool_size = int(len(pool_train_pos))
        if pool_size == 0:
            return None, None, None, None, 0

        self_train_pos = self._get_self_train_pos_if_same_training_row(query_index, query_coord_rad)
        if self._has_finite_eligibility_caps():
            if self.max_distance_km is None:
                result = self._query_raw_candidates(
                    nn=nn,
                    pool_train_pos=pool_train_pos,
                    query_coord_rad=query_coord_rad,
                    n_raw=pool_size,
                )
                if result is None:
                    return None, None, None, None, 0
                distances_km, candidate_train_pos = result
            else:
                distances_rad, local_idx = nn.radius_neighbors(
                    query_coord_rad.reshape(1, -1),
                    radius=np.nextafter(
                        float(self.max_distance_km) / _EARTH_RADIUS_KM,
                        np.inf,
                    ),
                    return_distance=True,
                    sort_results=True,
                )
                distances_km = np.asarray(distances_rad[0], dtype=float) * _EARTH_RADIUS_KM
                candidate_train_pos = pool_train_pos[np.asarray(local_idx[0], dtype=int)]
            distances_km, candidate_train_pos = self._filter_candidate_positions(
                distances_km=distances_km,
                candidate_train_pos=candidate_train_pos,
                self_train_pos=self_train_pos,
                query_date=query_date,
            )
        else:
            self_in_pool = self_train_pos is not None and bool(np.any(pool_train_pos == self_train_pos))
            extra_for_self = int(bool(self.exclude_self) and self_in_pool)
            n_raw = self._candidate_count(pool_size=pool_size, extra_for_self=extra_for_self)
            result = self._query_raw_candidates(
                nn=nn,
                pool_train_pos=pool_train_pos,
                query_coord_rad=query_coord_rad,
                n_raw=n_raw,
            )
            if result is None:
                return None, None, None, None, 0
            distances_km, candidate_train_pos = result
            distances_km, candidate_train_pos = self._filter_candidate_positions(
                distances_km=distances_km,
                candidate_train_pos=candidate_train_pos,
                self_train_pos=self_train_pos,
                query_date=query_date,
            )
            # If the first candidate pull was too small because temporal filtering or
            # self exclusion removed many rows, fall back to querying the full pool.
            if len(candidate_train_pos) < int(self.k) and n_raw < pool_size:
                result = self._query_raw_candidates(
                    nn=nn,
                    pool_train_pos=pool_train_pos,
                    query_coord_rad=query_coord_rad,
                    n_raw=pool_size,
                )
                if result is None:
                    return None, None, None, None, 0
                distances_km, candidate_train_pos = result
                distances_km, candidate_train_pos = self._filter_candidate_positions(
                    distances_km=distances_km,
                    candidate_train_pos=candidate_train_pos,
                    self_train_pos=self_train_pos,
                    query_date=query_date,
                )

        eligible_pool_size = int(len(candidate_train_pos))
        if eligible_pool_size < int(self.k):
            return None, None, None, None, eligible_pool_size

        feature_dist = self._candidate_feature_distance(query_feature_scaled, candidate_train_pos)
        time_dist = self._candidate_time_distance(query_date, candidate_train_pos)
        u = self._composite_normalized_distance(distances_km, feature_dist, time_dist)

        order = self._stable_top_k_1d(u, int(self.k))
        return (
            distances_km[order],
            feature_dist[order],
            u[order],
            candidate_train_pos[order],
            eligible_pool_size,
        )

    def _query_raw_candidates(
        self,
        *,
        nn: NearestNeighbors,
        pool_train_pos: np.ndarray,
        query_coord_rad: np.ndarray,
        n_raw: int,
    ):
        if int(n_raw) <= 0:
            return None
        distances_rad, local_idx = nn.kneighbors(query_coord_rad.reshape(1, -1), n_neighbors=int(n_raw))
        distances_km = distances_rad[0].astype(float) * _EARTH_RADIUS_KM
        candidate_train_pos = pool_train_pos[local_idx[0]].astype(int)
        return distances_km, candidate_train_pos

    def _filter_candidate_positions(
        self,
        *,
        distances_km: np.ndarray,
        candidate_train_pos: np.ndarray,
        self_train_pos: Optional[int],
        query_date: Optional[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray]:
        keep = np.ones(len(candidate_train_pos), dtype=bool)

        if self.max_distance_km is not None:
            # Radius retrieval already enforces this on the exact fast path, but
            # retain the check here for categorical pools and numerical boundaries.
            keep &= np.asarray(distances_km, dtype=float) <= float(self.max_distance_km)

        if bool(self.exclude_self) and self_train_pos is not None:
            keep &= candidate_train_pos != int(self_train_pos)

        if self.neighbor_time_rule != "none":
            if query_date is None or self.train_dates_ is None:
                raise ValueError(
                    "neighbor_time_rule requires date_col in fit/transform or prediction_date."
                )
            candidate_dates = self.train_dates_[candidate_train_pos]
            if self.neighbor_time_rule == "past":
                keep &= candidate_dates < query_date
            elif self.neighbor_time_rule == "past_or_same_day":
                keep &= candidate_dates <= query_date
            else:
                raise ValueError("neighbor_time_rule must be 'none', 'past', or 'past_or_same_day'.")

        if self.max_time_distance_days is not None:
            if query_date is None or self.train_day_values_ is None:
                raise ValueError(
                    "max_time_distance_days requires date_col in fit/transform or prediction_date."
                )
            query_day = float(pd.Timestamp(query_date).value / (24.0 * 3600.0 * 1e9))
            candidate_days = self.train_day_values_[candidate_train_pos]
            keep &= np.abs(query_day - candidate_days) <= float(self.max_time_distance_days)

        return distances_km[keep], candidate_train_pos[keep]

    def _candidate_count(self, *, pool_size: int, extra_for_self: int) -> int:
        base = max(int(self.k) + int(extra_for_self), int(self.k) * int(self.candidate_multiplier))
        if self.min_candidates is not None:
            base = max(base, int(self.min_candidates))
        if self.max_candidates is not None:
            base = min(base, int(self.max_candidates))
        return int(min(max(base, int(self.k) + int(extra_for_self)), int(pool_size)))

    def _get_self_train_pos_if_same_training_row(self, query_index: Any, query_coord_rad: np.ndarray) -> Optional[int]:
        train_pos = self.index_to_train_pos_.get(query_index, None)
        if train_pos is None:
            return None
        same_coordinates = np.allclose(
            query_coord_rad,
            self.train_coords_rad_[int(train_pos)],
            rtol=0.0,
            atol=1e-12,
        )
        return int(train_pos) if bool(same_coordinates) else None

    # ------------------------------------------------------------------
    # Distance, kernel, target adjustment
    # ------------------------------------------------------------------
    def _bandwidth_for_row(self, distances_km: np.ndarray) -> float:
        valid = np.asarray(distances_km, dtype=float)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return float(self.min_bandwidth_km)
        if self.bandwidth == "adaptive":
            h = float(np.max(valid)) * float(self.bandwidth_scale)
        else:
            h = float(self.bandwidth)
        return max(h, float(self.min_bandwidth_km))

    def _composite_normalized_distance(
        self,
        distances_km: np.ndarray,
        feature_dist: np.ndarray,
        time_dist: np.ndarray,
    ) -> np.ndarray:
        h = self._bandwidth_for_row(distances_km)
        u_geo = np.asarray(distances_km, dtype=float) / h
        u = float(self.geo_weight) * u_geo
        if bool(self.use_feature_distance):
            u += (
                float(self.feature_alpha)
                * np.asarray(feature_dist, dtype=float)
                / float(self.feature_bandwidth)
            )
        if bool(self.use_time_decay):
            u += float(self.time_weight) * np.asarray(time_dist, dtype=float)
        return u

    def _kernel_values_from_u(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        finite_u = np.where(np.isfinite(u), u, np.inf)
        if self.kernel == "gaussian":
            out = np.exp(-0.5 * finite_u**2)
        elif self.kernel == "exponential":
            out = np.exp(-finite_u)
        elif self.kernel == "epanechnikov":
            out = np.maximum(0.0, 1.0 - finite_u**2)
        elif self.kernel == "triangular":
            out = np.maximum(0.0, 1.0 - finite_u)
        else:
            raise ValueError(
                "kernel must be one of: 'gaussian', 'exponential', 'epanechnikov', or 'triangular'."
            )
        return np.where(np.isfinite(u), out, 0.0)

    def _adjust_neighbor_targets_to_query_date(
        self,
        neighbor_pos: np.ndarray,
        query_day_values: Optional[np.ndarray],
    ) -> np.ndarray:
        out = np.full(neighbor_pos.shape, np.nan, dtype=float)
        valid = neighbor_pos >= 0
        if not np.any(valid):
            return out

        out[valid] = self.y_train_[neighbor_pos[valid]]
        if not bool(self.use_time_trend):
            return out
        if query_day_values is None or self.train_day_values_ is None:
            raise ValueError("use_time_trend=True requires date_col in transform or prediction_date.")

        if self.time_trend_fit_mode == "causal_prior":
            intercept, slope = self._causal_linear_trend_parameters(query_day_values)
            query_x = query_day_values - float(self.trend_origin_day_)
            query_trend = intercept + slope * query_x
            row_idx, _ = np.where(valid)
            pos = neighbor_pos[valid]
            candidate_x = self.train_day_values_[pos] - float(self.trend_origin_day_)
            out[valid] = out[valid] + query_trend[row_idx] - (
                intercept[row_idx] + slope[row_idx] * candidate_x
            )
            return out

        query_trend = self._predict_time_trend(query_day_values)
        train_trend = self._predict_time_trend(self.train_day_values_)
        row_idx, col_idx = np.where(valid)
        pos = neighbor_pos[valid]
        out[valid] = out[valid] + query_trend[row_idx] - train_trend[pos]
        return out

    def _time_decay_weights(
        self,
        neighbor_pos: np.ndarray,
        query_day_values: Optional[np.ndarray],
    ) -> np.ndarray:
        out = np.ones(neighbor_pos.shape, dtype=float)
        valid = neighbor_pos >= 0
        if not np.any(valid):
            return out
        if query_day_values is None or self.train_day_values_ is None:
            raise ValueError("use_time_decay=True requires date_col in transform or prediction_date.")
        half_life = float(self.time_decay_half_life_days)
        if half_life <= 0.0 or not np.isfinite(half_life):
            raise ValueError("time_decay_half_life_days must be positive and finite.")
        row_idx, _ = np.where(valid)
        pos = neighbor_pos[valid]
        delta_days = np.abs(query_day_values[row_idx] - self.train_day_values_[pos])
        out[valid] = np.power(0.5, delta_days / half_life)
        return out

    # ------------------------------------------------------------------
    # Numeric feature preprocessing
    # ------------------------------------------------------------------
    def _fit_feature_scaler(self, X_train: pd.DataFrame) -> None:
        cols = self.numeric_feature_cols
        if cols is None:
            cols = []
        if isinstance(cols, str):
            cols = [cols]
        self.numeric_feature_cols_ = list(cols)
        self.feature_center_ = None
        self.feature_scale_ = None
        self.train_features_scaled_ = None

        if not bool(self.use_feature_distance):
            return
        if not self.numeric_feature_cols_:
            raise ValueError("use_feature_distance=True requires numeric_feature_cols.")
        missing = [c for c in self.numeric_feature_cols_ if c not in X_train.columns]
        if missing:
            raise ValueError(f"Missing numeric_feature_cols in X_train: {missing}")

        Z = X_train[self.numeric_feature_cols_].apply(pd.to_numeric, errors="coerce")
        if self.feature_missing != "median":
            raise ValueError("feature_missing currently supports only 'median'.")

        if self.feature_scaler == "robust":
            center = Z.median(axis=0).to_numpy(dtype=float)
            q75 = Z.quantile(0.75, axis=0).to_numpy(dtype=float)
            q25 = Z.quantile(0.25, axis=0).to_numpy(dtype=float)
            scale = q75 - q25
        elif self.feature_scaler == "standard":
            center = Z.mean(axis=0).to_numpy(dtype=float)
            scale = Z.std(axis=0, ddof=0).to_numpy(dtype=float)
        else:
            raise ValueError("feature_scaler must be 'robust' or 'standard'.")

        scale = np.asarray(scale, dtype=float)
        scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
        center = np.asarray(center, dtype=float)
        center[~np.isfinite(center)] = 0.0

        self.feature_center_ = center
        self.feature_scale_ = scale
        self.train_features_scaled_ = self._transform_numeric_features(X_train)

    def _transform_numeric_features(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not bool(getattr(self, "use_feature_distance", False)):
            return None
        missing = [c for c in self.numeric_feature_cols_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing numeric_feature_cols in X: {missing}")
        Z = X[self.numeric_feature_cols_].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        Z = np.where(np.isfinite(Z), Z, self.feature_center_)
        return (Z - self.feature_center_) / self.feature_scale_

    def _candidate_feature_distance(
        self,
        query_feature_scaled: Optional[np.ndarray],
        candidate_train_pos: np.ndarray,
    ) -> np.ndarray:
        if not bool(self.use_feature_distance):
            return np.zeros(len(candidate_train_pos), dtype=float)
        if query_feature_scaled is None:
            raise RuntimeError("query_feature_scaled is required when use_feature_distance=True.")
        D = self.train_features_scaled_[candidate_train_pos] - query_feature_scaled.reshape(1, -1)
        # RMS standardized distance keeps the scale stable as the number of features changes.
        return np.sqrt(np.mean(D**2, axis=1))

    def _candidate_time_distance(
        self,
        query_date: Optional[pd.Timestamp],
        candidate_train_pos: np.ndarray,
    ) -> np.ndarray:
        if not bool(self.use_time_decay):
            return np.zeros(len(candidate_train_pos), dtype=float)
        if query_date is None or self.train_dates_ is None:
            raise ValueError("use_time_decay=True requires date_col in fit/transform or prediction_date.")
        bandwidth_days = self.time_bandwidth_days
        if bandwidth_days is None:
            bandwidth_days = self.time_decay_half_life_days
        query_day = float(pd.Timestamp(query_date).value / (24.0 * 3600.0 * 1e9))
        candidate_days = self.train_day_values_[candidate_train_pos]
        return np.abs(query_day - candidate_days) / float(bandwidth_days)

    # ------------------------------------------------------------------
    # Time trend
    # ------------------------------------------------------------------
    def _fit_time_trend(self) -> None:
        self.trend_intercept_ = None
        self.trend_slope_ = None
        self.trend_month_x_ = None
        self.trend_month_y_ = None

        if not bool(self.use_time_trend):
            return
        if self.target_transform != "log":
            raise ValueError("use_time_trend=True requires target_transform='log'.")
        if self.train_dates_ is None or self.train_day_values_ is None:
            raise ValueError("use_time_trend=True requires date_col in X_train.")

        if self.time_trend_fit_mode == "causal_prior":
            # Query-specific OLS fits are evaluated from prefix sufficient
            # statistics.  This makes a training row's trend adjustment depend
            # only on earlier targets, rather than on the complete fit prefix.
            if self.time_trend != "linear":
                raise ValueError("time_trend_fit_mode='causal_prior' requires time_trend='linear'.")
            self.trend_origin_day_ = float(np.min(self.train_day_values_))
            x = self.train_day_values_.astype(float) - self.trend_origin_day_
            order = np.argsort(x, kind="mergesort")
            self.trend_sorted_day_values_ = self.train_day_values_[order]
            x = x[order]
            y = self.y_train_[order].astype(float)
            self.trend_prefix_x_ = np.concatenate(([0.0], np.cumsum(x)))
            self.trend_prefix_y_ = np.concatenate(([0.0], np.cumsum(y)))
            self.trend_prefix_x2_ = np.concatenate(([0.0], np.cumsum(x * x)))
            self.trend_prefix_xy_ = np.concatenate(([0.0], np.cumsum(x * y)))
            return

        if self.time_trend == "linear":
            x = self.train_day_values_.astype(float)
            y = self.y_train_.astype(float)
            x_center = float(np.mean(x))
            x_std = float(np.std(x))
            if not np.isfinite(x_std) or x_std <= 0.0:
                self.trend_intercept_ = float(np.mean(y))
                self.trend_slope_ = 0.0
                self.trend_x_center_ = x_center
                self.trend_x_std_ = 1.0
                return
            xs = (x - x_center) / x_std
            slope, intercept = np.polyfit(xs, y, deg=1)
            self.trend_intercept_ = float(intercept)
            self.trend_slope_ = float(slope)
            self.trend_x_center_ = x_center
            self.trend_x_std_ = x_std
            return

        if self.time_trend == "monthly_mean":
            months = self.train_dates_.to_period("M")
            df = pd.DataFrame({"month": months, "y": self.y_train_})
            g = df.groupby("month", sort=True)["y"].mean()
            if g.empty:
                raise ValueError("Could not fit monthly_mean time trend: no valid dates.")
            self.trend_month_x_ = np.asarray([p.ordinal for p in g.index], dtype=float)
            self.trend_month_y_ = g.to_numpy(dtype=float)
            return

        raise ValueError("time_trend must be 'linear' or 'monthly_mean'.")

    def _causal_linear_trend_parameters(
        self, query_day_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """OLS intercept and slope using only targets dated strictly before each query."""
        if self.time_trend_fit_mode != "causal_prior":
            raise RuntimeError("Causal trend parameters were requested for a global trend fit.")
        query_day_values = np.asarray(query_day_values, dtype=float)
        counts = np.searchsorted(self.trend_sorted_day_values_, query_day_values, side="left")
        n = counts.astype(float)
        sum_x = self.trend_prefix_x_[counts]
        sum_y = self.trend_prefix_y_[counts]
        sum_x2 = self.trend_prefix_x2_[counts]
        sum_xy = self.trend_prefix_xy_[counts]
        mean_x = np.divide(sum_x, n, out=np.zeros_like(sum_x), where=n > 0.0)
        mean_y = np.divide(sum_y, n, out=np.zeros_like(sum_y), where=n > 0.0)
        denominator = sum_x2 - n * mean_x**2
        numerator = sum_xy - n * mean_x * mean_y
        slope = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=(n >= 2.0) & (denominator > np.finfo(float).eps),
        )
        return mean_y - slope * mean_x, slope

    def _predict_time_trend(self, day_values: np.ndarray) -> np.ndarray:
        x = np.asarray(day_values, dtype=float)
        if self.time_trend == "linear":
            xs = (x - float(self.trend_x_center_)) / float(self.trend_x_std_)
            return float(self.trend_intercept_) + float(self.trend_slope_) * xs

        if self.time_trend == "monthly_mean":
            # Convert days to monthly ordinals approximately through pandas.
            dates = pd.to_datetime(x, unit="D", origin="unix")
            month_x = np.asarray([p.ordinal for p in dates.to_period("M")], dtype=float)
            return np.interp(
                month_x,
                self.trend_month_x_,
                self.trend_month_y_,
                left=float(self.trend_month_y_[0]),
                right=float(self.trend_month_y_[-1]),
            )

        raise RuntimeError(f"Unknown fitted time_trend: {self.time_trend}")

    def _resolve_dates(self, X: pd.DataFrame, *, fitting: bool) -> Optional[pd.DatetimeIndex]:
        date_values = None
        if self.date_col is not None and self.date_col in X.columns:
            date_values = pd.to_datetime(X[self.date_col], errors="coerce")
        elif self.prediction_date is not None and not fitting:
            date_values = pd.Series(pd.Timestamp(self.prediction_date), index=X.index)
        elif self.prediction_date is not None and fitting and self.date_col is None:
            # Fitting a trend or time filter from a constant date is not meaningful,
            # but this keeps spatial-only usage with prediction_date harmless.
            date_values = pd.Series(pd.Timestamp(self.prediction_date), index=X.index)

        needs_dates = (
            self.neighbor_time_rule != "none"
            or bool(self.use_time_trend)
            or bool(self.use_time_decay)
            or self.max_time_distance_days is not None
        )
        if date_values is None:
            if needs_dates:
                raise ValueError(
                    "date_col must be provided in X, or prediction_date must be set for transform, "
                    "when using temporal filtering, time trend, time decay, or a time eligibility cap."
                )
            return None

        if pd.isna(date_values).any():
            if self.prediction_date is not None and not fitting:
                date_values = date_values.fillna(pd.Timestamp(self.prediction_date))
            if pd.isna(date_values).any():
                raise ValueError("Date column contains missing or invalid dates.")
        return pd.DatetimeIndex(date_values)

    @staticmethod
    def _dates_to_day_values(dates: pd.DatetimeIndex) -> np.ndarray:
        # ``DatetimeIndex.view('int64')`` uses the index's native resolution.
        # Pandas may preserve a seconds-resolution input, so explicitly request
        # nanoseconds before converting to days.  The caps and time distances
        # must have the same units regardless of the source date dtype.
        nanoseconds = pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]").astype("int64")
        return (nanoseconds / (24.0 * 3600.0 * 1e9)).astype(float)

    # ------------------------------------------------------------------
    # Categorical filters copied from the original class, with minor cleanup
    # ------------------------------------------------------------------
    def _setup_filter_columns(self, X_train: pd.DataFrame) -> None:
        roots = self.categorical_filter_roots
        if roots is None:
            roots = []
        if isinstance(roots, str):
            roots = [roots]
        self.filter_roots_ = list(roots)
        self.filter_columns_by_root_ = {}
        self.filter_mode_by_root_ = {}
        self.filter_semantic_name_by_col_ = {col: self._strip_transformer_prefix(col) for col in X_train.columns}

        for root in self.filter_roots_:
            prefix = f"{root}{self.one_hot_sep}"
            one_hot_cols = [
                col
                for col, semantic_col in self.filter_semantic_name_by_col_.items()
                if semantic_col.startswith(prefix)
            ]
            one_hot_cols = sorted(one_hot_cols, key=lambda c: self.filter_semantic_name_by_col_[c])
            binary_cols = [
                col
                for col, semantic_col in self.filter_semantic_name_by_col_.items()
                if semantic_col == root
            ]
            binary_cols = sorted(binary_cols)

            if one_hot_cols:
                self.filter_columns_by_root_[root] = one_hot_cols
                self.filter_mode_by_root_[root] = "one_hot"
            elif self.allow_binary_filter and binary_cols:
                if len(binary_cols) > 1:
                    raise ValueError(
                        f"Multiple binary-style columns found for root '{root}': {binary_cols}."
                    )
                self.filter_columns_by_root_[root] = [binary_cols[0]]
                self.filter_mode_by_root_[root] = "binary"
            else:
                candidates = [
                    col
                    for col, semantic_col in self.filter_semantic_name_by_col_.items()
                    if root in semantic_col or root in str(col)
                ]
                raise ValueError(
                    f"No valid categorical filter columns found for root '{root}'. "
                    f"Expected one-hot columns starting with '{prefix}' or a binary column exactly '{root}'. "
                    f"Candidate similar columns found: {candidates[:20]}"
                )

    def _make_filter_keys(self, X: pd.DataFrame) -> list[Any]:
        if not self.filter_roots_:
            return [None] * len(X)
        keys = []
        for _, row in X.iterrows():
            key_parts = []
            valid = True
            for root in self.filter_roots_:
                mode = self.filter_mode_by_root_[root]
                cols = self.filter_columns_by_root_[root]
                if mode == "one_hot":
                    values = pd.to_numeric(row[cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    active = np.flatnonzero(values > float(self.one_hot_threshold))
                    if len(active) == 1:
                        active_col = cols[int(active[0])]
                        key_parts.append(self._strip_transformer_prefix(active_col))
                    elif len(active) == 0:
                        if self.handle_missing_category:
                            key_parts.append(f"{root}{self.one_hot_sep}{self.missing_category_label}")
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                elif mode == "binary":
                    col = cols[0]
                    value = self._coerce_binary_value(row[col])
                    if np.isnan(value):
                        if self.handle_missing_category:
                            key_parts.append(f"{root}{self.one_hot_sep}{self.missing_category_label}")
                        else:
                            valid = False
                            break
                    else:
                        key_parts.append(f"{root}{self.one_hot_sep}{int(value > float(self.one_hot_threshold))}")
                else:
                    raise RuntimeError(f"Unknown filter mode for root '{root}': {mode}")
            keys.append(tuple(key_parts) if valid else None)
        return keys

    @staticmethod
    def _strip_transformer_prefix(col: Any) -> str:
        col = str(col)
        return col.split("__")[-1] if "__" in col else col

    @staticmethod
    def _coerce_binary_value(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"1", "true", "t", "yes", "y"}:
                return 1.0
            if s in {"0", "false", "f", "no", "n"}:
                return 0.0
        numeric = pd.to_numeric(value, errors="coerce")
        return np.nan if pd.isna(numeric) else float(numeric)

    # ------------------------------------------------------------------
    # Validation / target prep
    # ------------------------------------------------------------------
    def _prepare_y(self, y: Any, index: pd.Index) -> pd.Series:
        if isinstance(y, pd.Series):
            out = y.reindex(index)
        else:
            out = pd.Series(y, index=index)
        if out.isna().any():
            raise ValueError("y_train contains missing values.")
        out = out.astype(float)
        if self.target_transform == "log":
            if (out <= 0).any():
                raise ValueError("Cannot apply log transform because y_train has nonpositive values.")
            out = np.log(out)
        elif self.target_transform is None:
            pass
        else:
            raise ValueError("target_transform must be None or 'log'.")
        return pd.Series(out, index=index, dtype=float)

    def _validate_init_params(self) -> None:
        if self.filter_fallback not in {"global", "raise"}:
            raise ValueError("filter_fallback must be either 'global' or 'raise'.")
        if self.neighbor_time_rule not in {"none", "past", "past_or_same_day"}:
            raise ValueError("neighbor_time_rule must be 'none', 'past', or 'past_or_same_day'.")
        if self.time_trend_fit_mode not in {"global", "causal_prior"}:
            raise ValueError("time_trend_fit_mode must be 'global' or 'causal_prior'.")
        if self.time_trend_fit_mode == "causal_prior" and self.time_trend != "linear":
            raise ValueError("time_trend_fit_mode='causal_prior' requires time_trend='linear'.")
        if self.insufficient_neighbors not in {"nan", "raise"}:
            raise ValueError("insufficient_neighbors must be 'nan' or 'raise'.")
        if float(self.one_hot_threshold) < 0:
            raise ValueError("one_hot_threshold must be nonnegative.")
        if not isinstance(self.handle_missing_category, bool):
            raise ValueError("handle_missing_category must be True or False.")
        if not isinstance(self.missing_category_label, str) or not self.missing_category_label:
            raise ValueError("missing_category_label must be a non-empty string.")
        if not isinstance(self.allow_binary_filter, bool):
            raise ValueError("allow_binary_filter must be True or False.")
        if int(self.candidate_multiplier) < 1:
            raise ValueError("candidate_multiplier must be >= 1.")
        if self.batch_query_size is not None and int(self.batch_query_size) < 1:
            raise ValueError("batch_query_size must be a positive integer when supplied.")
        if int(self.full_pool_batch_size) < 1:
            raise ValueError("full_pool_batch_size must be a positive integer.")
        if float(self.geo_weight) < 0.0:
            raise ValueError("geo_weight must be nonnegative.")
        if self.max_distance_km is not None:
            max_distance_km = float(self.max_distance_km)
            if not np.isfinite(max_distance_km) or max_distance_km <= 0.0:
                raise ValueError("max_distance_km must be positive and finite when supplied.")
        if float(self.feature_alpha) < 0.0:
            raise ValueError("feature_alpha must be nonnegative.")
        feature_bandwidth = float(self.feature_bandwidth)
        if not np.isfinite(feature_bandwidth) or feature_bandwidth <= 0.0:
            raise ValueError("feature_bandwidth must be positive and finite.")
        if float(self.time_weight) < 0.0:
            raise ValueError("time_weight must be nonnegative.")
        if self.max_time_distance_days is not None:
            max_time_distance_days = float(self.max_time_distance_days)
            if not np.isfinite(max_time_distance_days) or max_time_distance_days <= 0.0:
                raise ValueError("max_time_distance_days must be positive and finite when supplied.")
        if bool(self.use_time_decay):
            bandwidth_days = self.time_bandwidth_days
            if bandwidth_days is None:
                bandwidth_days = self.time_decay_half_life_days
            bandwidth_days = float(bandwidth_days)
            if not np.isfinite(bandwidth_days) or bandwidth_days <= 0.0:
                raise ValueError("time_bandwidth_days must be positive and finite.")

    def _validate_X(self, X: pd.DataFrame, *, require_filter_cols: bool) -> None:
        missing = [c for c in [self.lat_col, self.lon_col] if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if X[[self.lat_col, self.lon_col]].isna().any().any():
            raise ValueError("Latitude/longitude columns contain missing values.")

        if require_filter_cols and hasattr(self, "filter_columns_by_root_"):
            required_filter_cols = [col for cols in self.filter_columns_by_root_.values() for col in cols]
            missing_filter_cols = [c for c in required_filter_cols if c not in X.columns]
            if missing_filter_cols:
                raise ValueError(
                    "X is missing categorical filter columns that were present during fit: "
                    f"{missing_filter_cols[:10]}"
                )

    def _validate_is_fitted(self) -> None:
        if not hasattr(self, "global_nn_"):
            raise RuntimeError("The transformer must be fitted before calling transform().")
