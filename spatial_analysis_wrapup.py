"""
CCAO sales-filtering and model-comparison analysis.

This script loads the CCAO training parquet, applies several sales/property filters,
explores log-sale-price distributions, and compares model performance across the
resulting datasets.

References:
- CCAO residential AVM repository / README for residential modeling classes.
- CCAO class-code definitions PDF for `meta_class` descriptions.
"""

# =============================================================================
# Imports and configuration
# =============================================================================

import json

import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from preprocessing.recipes_pipelined import build_model_pipeline

try:
    # Used for notebook-style table output.
    from IPython.display import display
except ImportError:  # pragma: no cover - fallback for non-notebook execution.
    display = print


try:
    from sklearn.metrics import root_mean_squared_error as _root_mse
except ImportError:  # sklearn < 1.4 compatibility.
    def _root_mse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

DATA_PATH = "./data/CCAO/2025/training_data.parquet"

DATE_COLUMN = "meta_sale_date"
TARGET_COLUMN = "meta_sale_price"

predictor_cols = list(params["model"]["predictor"]["all"])
categorical_cols = list(params["model"]["predictor"]["categorical"])

# Columns needed only for early filtering.
filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]

# Columns needed for exploratory filtering and grouping later in the script.
extra_cols = [
    "meta_class",
    "meta_triad_name",
    "meta_sale_deed_type",
    "sv_review_json",
]

# Neighbor-price feature settings. These values are intentionally simple defaults;
# change them here only if you want a tighter/wider neighborhood definition.
NEIGHBOR_MAX_COUNT = 5
NEIGHBOR_DISTANCE_THRESHOLD_MILES = 0.25
NEIGHBOR_TIME_THRESHOLD_DAYS = 365
NEIGHBOR_MIN_LAG_DAYS = 1

# Prefer latitude/longitude, but allow projected coordinate alternatives if present.
COORD_COLUMN_CANDIDATES = [
    ("loc_latitude", "loc_longitude", "latlon"),
    ("latitude", "longitude", "latlon"),
    ("lat", "lon", "latlon"),
    ("lat", "long", "latlon"),
    ("loc_y_coord", "loc_x_coord", "feet"),
    ("y", "x", "raw"),
]

try:
    available_columns = set(pq.read_schema(DATA_PATH).names)
except Exception:
    # Fallback keeps the script usable in environments where parquet metadata cannot be read.
    available_columns = set(predictor_cols + [TARGET_COLUMN, DATE_COLUMN] + filter_cols + extra_cols)

COORD_COLUMNS = next(
    (
        {"lat_col": lat_col, "lon_col": lon_col, "kind": coord_kind}
        for lat_col, lon_col, coord_kind in COORD_COLUMN_CANDIDATES
        if {lat_col, lon_col}.issubset(available_columns)
    ),
    None,
)
coord_cols = [] if COORD_COLUMNS is None else [COORD_COLUMNS["lat_col"], COORD_COLUMNS["lon_col"]]

# Consolidate required columns before reading the parquet to limit memory use.
cols_to_load = list(
    set(predictor_cols + [TARGET_COLUMN, DATE_COLUMN] + filter_cols + extra_cols + coord_cols)
)


# =============================================================================
# Data loading and base filters
# =============================================================================

# Load only the required columns using PyArrow.
df = pd.read_parquet(DATA_PATH, engine="pyarrow", columns=cols_to_load)

print("AAAA")
print(df.head())

# Remove multicard PINs and rows already flagged as sales-validation outliers.
df = df[
    (df["ind_pin_is_multicard"] != True)
    & (df["sv_is_outlier"] != True)
].copy()

# Ensure sale date is usable for chronological splitting later.
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

# Optional out-of-time split retained for future use.
# df_2024 = df[df[DATE_COLUMN].dt.year == 2024]
# df = df[df[DATE_COLUMN].dt.year <= 2023]

# Keep only modeling, target/date, and analysis columns after the base filters.
final_keep_cols = list(set(predictor_cols + [TARGET_COLUMN, DATE_COLUMN] + extra_cols + coord_cols))
df = df.loc[:, final_keep_cols]
# df_2024 = df_2024.loc[:, final_keep_cols]

# Take a reproducible sample if the dataset is very large.
sample_size = 200000
if len(df) > sample_size:
    df = df.sample(sample_size, random_state=42)
# if len(df_2024) > sample_size:
#     df_2024 = df_2024.sample(sample_size, random_state=42)


# =============================================================================
# Helper functions
# =============================================================================


def safe_parse_json(val):
    """Parse CCAO sales-validation JSON safely, returning an empty dict on failure."""
    if isinstance(val, dict):
        return val
    if pd.isna(val) or val in ["None", None]:
        return {}
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return {}


def _coordinate_matrix(frame, coord_info):
    """Return valid coordinate rows in BallTree-ready units."""
    lat = pd.to_numeric(frame[coord_info["lat_col"]], errors="coerce").to_numpy()
    lon = pd.to_numeric(frame[coord_info["lon_col"]], errors="coerce").to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon)

    if coord_info["kind"] == "latlon":
        valid &= (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        coords = np.deg2rad(np.column_stack([lat[valid], lon[valid]]))
    else:
        coords = np.column_stack([lat[valid], lon[valid]])

    return coords, valid


def _distance_radius(coord_kind):
    """Return the BallTree metric and search radius for the configured coordinates."""
    if coord_kind == "latlon":
        earth_radius_miles = 3958.7613
        return "haversine", NEIGHBOR_DISTANCE_THRESHOLD_MILES / earth_radius_miles
    if coord_kind == "feet":
        return "euclidean", NEIGHBOR_DISTANCE_THRESHOLD_MILES * 5280.0
    return "euclidean", NEIGHBOR_DISTANCE_THRESHOLD_MILES


def add_neighbor_price_features(
    reference_df,
    target_df,
    *,
    coord_info,
    prefix,
    time_window_days=None,
    min_lag_days=NEIGHBOR_MIN_LAG_DAYS,
):
    """
    Add simple, leakage-safe neighbor sale-price proxies.

    Features use only reference sales from the already-split reference set.
    A positive `min_lag_days` prevents same-day/self-sale leakage in training;
    `time_window_days` adds an optional backward-looking temporal threshold.
    """
    if coord_info is None:
        raise ValueError("No usable coordinate columns were found for neighbor features.")

    out = target_df.copy()
    feature_cols = [
        f"{prefix}_log_price_mean",
        f"{prefix}_log_price_median",
        f"{prefix}_nearest_log_price",
        f"{prefix}_neighbor_count",
        f"{prefix}_mean_distance_miles",
    ]

    ref = reference_df.dropna(subset=[TARGET_COLUMN, DATE_COLUMN]).copy()
    ref = ref[ref[TARGET_COLUMN] > 0]
    fallback_log_price = float(np.log(ref[TARGET_COLUMN]).median()) if len(ref) else 0.0

    if ref.empty or target_df.empty:
        out[feature_cols[0]] = fallback_log_price
        out[feature_cols[1]] = fallback_log_price
        out[feature_cols[2]] = fallback_log_price
        out[feature_cols[3]] = 0
        out[feature_cols[4]] = NEIGHBOR_DISTANCE_THRESHOLD_MILES
        return out, feature_cols

    ref_coords, ref_valid = _coordinate_matrix(ref, coord_info)
    target_coords, target_valid = _coordinate_matrix(target_df, coord_info)

    if ref_coords.shape[0] == 0 or target_coords.shape[0] == 0:
        out[feature_cols[0]] = fallback_log_price
        out[feature_cols[1]] = fallback_log_price
        out[feature_cols[2]] = fallback_log_price
        out[feature_cols[3]] = 0
        out[feature_cols[4]] = NEIGHBOR_DISTANCE_THRESHOLD_MILES
        return out, feature_cols

    ref_valid_df = ref.loc[ref_valid].copy()
    ref_dates = pd.to_datetime(ref_valid_df[DATE_COLUMN]).to_numpy(dtype="datetime64[ns]")
    ref_log_prices = np.log(ref_valid_df[TARGET_COLUMN].to_numpy(dtype=float))

    metric, radius = _distance_radius(coord_info["kind"])
    tree = BallTree(ref_coords, metric=metric)
    neighbor_idx, neighbor_dist = tree.query_radius(
        target_coords,
        r=radius,
        return_distance=True,
        sort_results=True,
    )

    mean_vals = np.full(len(target_df), fallback_log_price, dtype=float)
    median_vals = np.full(len(target_df), fallback_log_price, dtype=float)
    nearest_vals = np.full(len(target_df), fallback_log_price, dtype=float)
    count_vals = np.zeros(len(target_df), dtype=int)
    distance_vals = np.full(len(target_df), NEIGHBOR_DISTANCE_THRESHOLD_MILES, dtype=float)

    target_positions = np.flatnonzero(target_valid)
    target_dates = pd.to_datetime(target_df.loc[target_valid, DATE_COLUMN]).to_numpy(dtype="datetime64[ns]")
    max_time_delta = None if time_window_days is None else np.timedelta64(int(time_window_days), "D")

    for local_pos, row_pos in enumerate(target_positions):
        idx = neighbor_idx[local_pos]
        dist = neighbor_dist[local_pos]
        if idx.size == 0:
            continue

        time_delta = target_dates[local_pos] - ref_dates[idx]
        keep = time_delta >= np.timedelta64(int(min_lag_days), "D")
        if max_time_delta is not None:
            keep &= time_delta <= max_time_delta

        if not np.any(keep):
            continue

        idx = idx[keep][:NEIGHBOR_MAX_COUNT]
        dist = dist[keep][:NEIGHBOR_MAX_COUNT]
        prices = ref_log_prices[idx]

        if coord_info["kind"] == "latlon":
            dist_miles = dist * 3958.7613
        elif coord_info["kind"] == "feet":
            dist_miles = dist / 5280.0
        else:
            dist_miles = dist

        mean_vals[row_pos] = float(np.mean(prices))
        median_vals[row_pos] = float(np.median(prices))
        nearest_vals[row_pos] = float(prices[0])
        count_vals[row_pos] = int(prices.size)
        distance_vals[row_pos] = float(np.mean(dist_miles))

    out[feature_cols[0]] = mean_vals
    out[feature_cols[1]] = median_vals
    out[feature_cols[2]] = nearest_vals
    out[feature_cols[3]] = count_vals
    out[feature_cols[4]] = distance_vals

    return out, feature_cols


def add_benchmark_neighbor_features(train_data, test_data, baseline_test_data, mode):
    """Create leakage-safe neighbor benchmark features inside each split."""
    if mode == "spatial":
        prefix = "nb_spatial"
        time_window_days = None
    elif mode == "spacetime":
        prefix = "nb_spacetime"
        time_window_days = NEIGHBOR_TIME_THRESHOLD_DAYS
    else:
        return train_data, test_data, baseline_test_data, []

    # Training rows are featurized using only earlier training sales. Use the
    # train/test date gap as the minimum lag when it is larger than one day, so
    # in-sample neighbor freshness is closer to what test rows receive.
    test_start = pd.to_datetime(test_data[DATE_COLUMN]).min()
    train_end = pd.to_datetime(train_data[DATE_COLUMN]).max()
    train_min_lag_days = NEIGHBOR_MIN_LAG_DAYS
    if pd.notna(test_start) and pd.notna(train_end):
        split_gap_days = int(np.ceil((test_start - train_end) / pd.Timedelta(days=1)))
        train_min_lag_days = max(NEIGHBOR_MIN_LAG_DAYS, split_gap_days)

    train_data, feature_cols = add_neighbor_price_features(
        train_data,
        train_data,
        coord_info=COORD_COLUMNS,
        prefix=prefix,
        time_window_days=time_window_days,
        min_lag_days=train_min_lag_days,
    )
    test_data, _ = add_neighbor_price_features(
        train_data,
        test_data,
        coord_info=COORD_COLUMNS,
        prefix=prefix,
        time_window_days=time_window_days,
        min_lag_days=NEIGHBOR_MIN_LAG_DAYS,
    )
    baseline_test_data, _ = add_neighbor_price_features(
        train_data,
        baseline_test_data,
        coord_info=COORD_COLUMNS,
        prefix=prefix,
        time_window_days=time_window_days,
        min_lag_days=NEIGHBOR_MIN_LAG_DAYS,
    )

    return train_data, test_data, baseline_test_data, feature_cols


# =============================================================================
# Evaluation metrics
# =============================================================================

# Accuracy and assessment-ratio metrics adapted from the provided
# compute_taxation_metrics helper. Inputs here are sale prices on the price scale.


def oos_r2_score(y_test, y_pred, y_train):
    """Out-of-sample R2 using the training-set mean as the baseline."""
    y_train = np.asarray(y_train, dtype=float).flatten()
    y_test = np.asarray(y_test, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    mse_model = np.mean((y_test - y_pred) ** 2)
    mse_baseline = np.mean((y_test - np.mean(y_train)) ** 2)

    if mse_baseline == 0:
        return 0.0
    return 1 - (mse_model / mse_baseline)


def _ensure_arrays(a, b=None):
    """Convert one or two array-like inputs to float numpy arrays."""
    a = np.asarray(a, dtype=float)
    if b is not None:
        b = np.asarray(b, dtype=float)
        return a, b
    return a


def _handle_na(arrays, na_rm=False):
    """Apply common NA filtering across one or more arrays."""
    if isinstance(arrays, tuple):
        combined = np.column_stack(arrays)
        mask = ~np.isnan(combined).any(axis=1)

        if na_rm:
            return [arr[mask] for arr in arrays]
        if not np.all(mask):
            return [None] * len(arrays)
        return arrays

    mask = ~np.isnan(arrays)
    if na_rm:
        return arrays[mask]
    if not np.all(mask):
        return None
    return arrays


def cod(ratio, na_rm=False):
    """Coefficient of Dispersion: average absolute percent deviation from median ratio."""
    ratio = _ensure_arrays(ratio)
    ratio = _handle_na(ratio, na_rm)

    if ratio is None or len(ratio) == 0:
        return np.nan

    med_ratio = np.median(ratio)
    if med_ratio == 0:
        return np.nan

    return (np.mean(np.abs(ratio - med_ratio)) / med_ratio) * 100


def cov_iaao(assessed, sale_price, na_rm=False):
    """IAAO coefficient of variation for assessment ratios."""
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    cleaned = _handle_na((assessed, sale_price), na_rm)

    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    m = (
        np.isfinite(assessed)
        & np.isfinite(sale_price)
        & (assessed > 0)
        & (sale_price > 0)
    )
    assessed, sale_price = assessed[m], sale_price[m]

    if assessed.size < 2:
        return np.nan

    ratio = assessed / sale_price
    ratio = ratio[np.isfinite(ratio)]

    if ratio.size < 2:
        return np.nan

    mean_ratio = np.mean(ratio)
    if mean_ratio == 0:
        return np.nan

    return np.std(ratio, ddof=1) / mean_ratio


def vei(assessed, sale_price, na_rm=False):
    """Vertical Equity Indicator point estimate using IAAO-style percentile groups."""
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    cleaned = _handle_na((assessed, sale_price), na_rm)

    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    m = (
        np.isfinite(assessed)
        & np.isfinite(sale_price)
        & (assessed > 0)
        & (sale_price > 0)
    )
    assessed, sale_price = assessed[m], sale_price[m]

    n = len(assessed)
    if n < 20:
        return np.nan

    if n <= 50:
        k = 2
    elif n <= 500:
        k = 4
    else:
        k = 10

    ratio = assessed / sale_price
    ratio = ratio[np.isfinite(ratio)]

    if ratio.size == 0:
        return np.nan

    med = np.median(ratio)
    if not np.isfinite(med) or med == 0:
        return np.nan

    proxy = 0.5 * sale_price + 0.5 * (assessed / med)
    order = np.argsort(proxy, kind="mergesort")
    chunks = np.array_split(np.arange(n), k)

    first_idx = order[chunks[0]]
    last_idx = order[chunks[-1]]

    if first_idx.size < 10 or last_idx.size < 10:
        return np.nan

    m_first = np.median(assessed[first_idx] / sale_price[first_idx])
    m_last = np.median(assessed[last_idx] / sale_price[last_idx])

    if not (np.isfinite(m_first) and np.isfinite(m_last)):
        return np.nan

    return 100.0 * (m_last - m_first) / med


def prd(assessed, sale_price, na_rm=False):
    """Price-Related Differential."""
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    cleaned = _handle_na((assessed, sale_price), na_rm)

    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    if len(assessed) == 0:
        return np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = assessed / sale_price

    valid_ratios = np.isfinite(ratio)
    ratio = ratio[valid_ratios]
    assessed = assessed[valid_ratios]
    sale_price = sale_price[valid_ratios]

    mean_ratio = np.mean(ratio)
    weighted_mean_ratio = np.sum(assessed) / np.sum(sale_price)

    if weighted_mean_ratio == 0:
        return np.nan

    return mean_ratio / weighted_mean_ratio


def prb(assessed, sale_price, na_rm=False):
    """Coefficient of Price-Related Bias."""
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    cleaned = _handle_na((assessed, sale_price), na_rm)

    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    if len(assessed) < 2:
        return np.nan

    ratio = assessed / sale_price
    med_ratio = np.median(ratio)

    if med_ratio == 0:
        return np.nan

    lhs = (ratio - med_ratio) / med_ratio
    inner_term = ((assessed / med_ratio) + sale_price) * 0.5
    valid_idx = inner_term > 0

    if not np.any(valid_idx):
        return np.nan

    lhs = lhs[valid_idx]
    rhs = np.log2(inner_term[valid_idx])

    try:
        slope, _ = np.polyfit(rhs, lhs, 1)
        return slope
    except Exception:
        return np.nan


def _calc_gini(assessed, sale_price):
    """Gini helper for the Modified Kakwani Index."""
    gini_df = pd.DataFrame({"av": assessed, "sp": sale_price})
    gini_df = gini_df.sort_values(by=["sp", "av"], ascending=[True, False])

    assessed_sorted = gini_df["av"].values
    sale_sorted = gini_df["sp"].values
    n = len(assessed_sorted)
    seq = np.arange(1, n + 1)

    g_assessed = (2 * np.sum(assessed_sorted * seq) / np.sum(assessed_sorted)) - (n + 1)
    g_sale = (2 * np.sum(sale_sorted * seq) / np.sum(sale_sorted)) - (n + 1)

    return g_assessed / n, g_sale / n


def mki(assessed, sale_price, na_rm=False):
    """Modified Kakwani Index."""
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    cleaned = _handle_na((assessed, sale_price), na_rm)

    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    if len(assessed) == 0:
        return np.nan

    g_av, g_sp = _calc_gini(assessed, sale_price)

    if g_sp == 0:
        return np.nan
    return g_av / g_sp


def compute_taxation_metrics(y_real, y_pred, scale="price", y_train=None):
    """Compute accuracy and tax-assessment ratio metrics."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if scale == "log":
        y_real_log, y_pred_log = y_real, y_pred
        y_real, y_pred = np.exp(y_real), np.exp(y_pred)
        y_train_baseline = np.exp(y_train) if y_train is not None else y_real
    else:
        y_real_log, y_pred_log = np.log(y_real), np.log(y_pred)
        y_train_baseline = y_train if y_train is not None else y_real

    metrics = {}

    # 1. Accuracy metrics
    metrics["R2"] = r2_score(y_real, y_pred)
    metrics["OOS R2"] = oos_r2_score(y_real, y_pred, y_train_baseline)
    metrics["R2 (log)"] = r2_score(y_real_log, y_pred_log)
    metrics["RMSE"] = _root_mse(y_real, y_pred)
    metrics["MAE"] = mean_absolute_error(y_real, y_pred)
    metrics["MAPE"] = mean_absolute_percentage_error(y_real, y_pred)
    metrics["MdAPE"] = 100 * median_absolute_error(y_real / y_pred, y_pred / y_pred)

    # 2. Ratio metrics
    ratios = y_pred / y_real
    metrics["Corr(r,price)"] = np.corrcoef(ratios, y_real)[0, 1]
    metrics["Corr(r,logprice)"] = np.corrcoef(ratios, y_real_log)[0, 1]
    metrics["Slope(r~logy)"] = np.polyfit(y_real_log, ratios, 1)[0]
    metrics["Std ratio"] = float(np.std(ratios))
    metrics["Median ratio"] = np.median(ratios)
    metrics["Mean ratio"] = np.mean(ratios)
    metrics["W. Mean ratio"] = np.sum(y_pred) / np.sum(y_real)

    # 3. Assessment-uniformity and vertical-equity metrics
    metrics["COD"] = cod(ratios, na_rm=True)
    metrics["COV_IAAO"] = cov_iaao(y_pred, y_real, na_rm=True)
    metrics["VEI"] = vei(y_pred, y_real, na_rm=True)
    metrics["PRD"] = prd(y_pred, y_real, na_rm=True)
    metrics["PRB"] = prb(y_pred, y_real, na_rm=True)
    metrics["MKI"] = mki(y_pred, y_real, na_rm=True)

    return metrics


def calculate_evaluation_metrics(y_real, y_pred, y_train=None):
    """Compatibility wrapper for the model loop below."""
    baseline = y_real if y_train is None else y_train
    return compute_taxation_metrics(y_real, y_pred, scale="price", y_train=baseline)


# =============================================================================
# Arms-length sale exploration and filter
# =============================================================================

# Extract the arms-length flag from the sales-validation review JSON.
df["is_arms_length"] = df["sv_review_json"].apply(
    lambda x: safe_parse_json(x).get("is_arms_length")
)

# Label True/False values for plotting; missing values are tracked separately.
df["arms_length_group"] = df["is_arms_length"].map(
    {
        True: "Arms-Length",
        False: "Non-Arms-Length",
    }
).fillna("Missing Data")

# Use log sale price for distribution comparisons.
df["log_price"] = np.log(df["meta_sale_price"])

plt.figure(figsize=(10, 6))

# Plot only the True/False groups for a cleaner comparison.
for group_name in ["Arms-Length", "Non-Arms-Length"]:
    group = df[df["arms_length_group"] == group_name]

    if not group.empty:
        label_text = (
            f"{group_name} "
            f"(μ={group['log_price'].mean():.2f}, "
            f"σ={group['log_price'].std():.2f})"
        )
        plt.hist(group["log_price"], bins=100, alpha=0.5, density=True, label=label_text)

plt.xlabel("Log Meta Sale Price")
plt.ylabel("Density")
plt.title("Normalized Distribution: Arms-Length vs. Non-Arms-Length Sales")
plt.legend()
plt.show()

statistics_table_arms = df.groupby("arms_length_group")["log_price"].describe().reset_index()
display(statistics_table_arms)

# Keep rows where the sale is not explicitly marked non-arm's-length.
df_new = df.loc[df["is_arms_length"] != False].copy()

print(f"Clean dataframe shape: {df_new.shape}")
display(df_new.head(10))


# =============================================================================
# Deed-type exploration and filter
# =============================================================================

# Recalculate on the filtered dataframe to avoid relying on parent columns.
df_new["log_price"] = np.log(df_new["meta_sale_price"])

plt.figure(figsize=(10, 6))

for deed_type, group in df_new.groupby("meta_sale_deed_type"):
    label_text = (
        f"{deed_type} "
        f"(μ={group['log_price'].mean():.2f}, "
        f"σ={group['log_price'].std():.2f})"
    )
    plt.hist(group["log_price"], bins=100, alpha=0.5, density=True, label=label_text)

plt.xlabel("Log Meta Sale Price")
plt.ylabel("Density")
plt.title("Normalized Distribution of Log Sale Price by Deed Type")
plt.legend()
plt.show()

statistics_table = df_new.groupby("meta_sale_deed_type")["log_price"].describe().reset_index()
display(statistics_table)

# Keep only deed types 01 and 02.
df_new_2 = df_new[df_new.meta_sale_deed_type.isin(["01", "02"])].copy()
np.log(df_new_2.meta_sale_price).hist(
    bins=100,
    density=True,
    alpha=0.9,
    label="01 and 02",
    color="red",
)
plt.show()

# df_new_2.drop(columns=["meta_sale_deed_type"], inplace=True)
df_new_2.head(10)


# =============================================================================
# Past-sale indicator exploration
# =============================================================================

# Vectorized indicator creation: 1 if the property has a prior sale, else 0.
df_new_2["past_sale_indicator"] = (
    df_new_2["meta_sale_count_past_n_years"] > 0
).astype(int)

# Pre-calculate log price for this filtered dataframe.
df_new_2["log_price"] = np.log(df_new_2["meta_sale_price"])

plt.figure(figsize=(10, 6))

for indicator, group in df_new_2.groupby("past_sale_indicator"):
    label_name = "Has Past Sales (1)" if indicator == 1 else "No Past Sales (0)"
    label_text = (
        f"{label_name} "
        f"(μ={group['log_price'].mean():.2f}, "
        f"σ={group['log_price'].std():.2f})"
    )
    plt.hist(group["log_price"], bins=100, alpha=0.5, density=True, label=label_text)

plt.xlabel("Log Meta Sale Price")
plt.ylabel("Density")
plt.title("Normalized Distribution of Log Sale Price by Past Sale Indicator")
plt.legend()
plt.show()

statistics_table_2 = (
    df_new_2.groupby("past_sale_indicator")["log_price"].describe().reset_index()
)
display(statistics_table_2)


# =============================================================================
# Property-class grouping and single-family filter
# =============================================================================

# CCAO residential property-class grouping used for meta_class analysis.
meta_class_group = {
    "202": "Single-family",
    "203": "Single-family",
    "204": "Single-family",
    "205": "Single-family",
    "206": "Single-family",
    "207": "Single-family",
    "208": "Single-family",
    "209": "Single-family",
    "210": "Single-family / townhome-rowhouse",
    "234": "Single-family / split-level",
    "278": "Single-family",
    "295": "Single-family / townhome-rowhouse",
    "211": "Multi-family",
    "212": "Multi-family / mixed-use residential-commercial",
    "218": "Bed & breakfast; treated like single-family for modeling, usually hand-valued",
}

# Map class codes to descriptions, forcing strings for reliable matching.
mapped_classes = df_new_2["meta_class"].astype(str).map(meta_class_group)

# Collapse property types into a binary single-family / non-single-family grouping.
df_new_2["property_group"] = np.where(
    mapped_classes.fillna("").str.lower().str.contains("single-family"),
    "Single-Family",
    "Not Single-Family",
)

if "log_price" not in df_new_2.columns:
    df_new_2["log_price"] = np.log(df_new_2["meta_sale_price"])

plt.figure(figsize=(10, 6))

for group_name, group in df_new_2.groupby("property_group"):
    label_text = (
        f"{group_name} "
        f"(μ={group['log_price'].mean():.2f}, "
        f"σ={group['log_price'].std():.2f})"
    )
    plt.hist(group["log_price"], bins=100, alpha=0.5, density=True, label=label_text)

plt.xlabel("Log Meta Sale Price")
plt.ylabel("Density")
plt.title("Normalized Distribution of Log Sale Price: Single-Family vs Others")
plt.legend()
plt.show()

statistics_table_group = df_new_2.groupby("property_group")["log_price"].describe().reset_index()
statistics_table_group = statistics_table_group.sort_values(
    by="count",
    ascending=False,
).reset_index(drop=True)
display(statistics_table_group)

# Keep only single-family properties for the next dataset.
df_new_3 = df_new_2.loc[df_new_2["property_group"] == "Single-Family"].copy()

display(df_new_3["property_group"].value_counts())
df_new_3.drop(columns=["property_group"], inplace=True)

np.log(df_new_3.meta_sale_price).hist(
    bins=100,
    density=True,
    alpha=0.9,
    label="01 and 02",
    color="red",
)
plt.show()


# =============================================================================
# Triad exploration for single-family properties
# =============================================================================

if "log_price" not in df_new_3.columns:
    df_new_3["log_price"] = np.log(df_new_3["meta_sale_price"])

plt.figure(figsize=(10, 6))

for triad, group in df_new_3.groupby("meta_triad_name"):
    label_text = (
        f"{triad} "
        f"(μ={group['log_price'].mean():.2f}, "
        f"σ={group['log_price'].std():.2f})"
    )
    plt.hist(group["log_price"], bins=100, alpha=0.5, density=True, label=label_text)

plt.xlabel("Log Meta Sale Price")
plt.ylabel("Density")
plt.title("Normalized Distribution of Log Sale Price by Triad (Single-Family)")
plt.legend()
plt.show()

statistics_table_triad = df_new_3.groupby("meta_triad_name")["log_price"].describe().reset_index()
statistics_table_triad = statistics_table_triad.sort_values(
    by="count",
    ascending=False,
).reset_index(drop=True)
display(statistics_table_triad)


# =============================================================================
# Final dataset cleanup before model comparison
# =============================================================================

# Add the columns created during exploration to the drop list
exploration_cols = [
    "is_arms_length", 
    "arms_length_group", 
    "log_price", 
    "past_sale_indicator", 
    "property_group"
]

# Remove analysis-only columns before modeling where present.
for dataset in [df, df_new, df_new_2, df_new_3]:
    dataset.drop(columns=extra_cols + exploration_cols, inplace=True, errors="ignore")

# Model the original and progressively filtered datasets. Neighbor-price variants
# are added only on top of Models M1 / Dataset 1 and M3 / Dataset 3.
dataset_specs = [
    {"model_id": "M1", "name": "Dataset 1", "data": df, "neighbor_mode": None},
]

if COORD_COLUMNS is not None:
    dataset_specs.extend(
        [
            {
                "model_id": "M1-S",
                "name": "Dataset 1 + spatial neighbors",
                "data": df,
                "neighbor_mode": "spatial",
            },
            {
                "model_id": "M1-ST",
                "name": "Dataset 1 + spatial-time neighbors",
                "data": df,
                "neighbor_mode": "spacetime",
            },
        ]
    )
else:
    print("Skipping neighbor-price benchmarks: no supported coordinate columns found.")

dataset_specs.extend(
    [
        {"model_id": "M2", "name": "Dataset 2", "data": df_new, "neighbor_mode": None},
        {"model_id": "M3", "name": "Dataset 3", "data": df_new_2, "neighbor_mode": None},
    ]
)

if COORD_COLUMNS is not None:
    dataset_specs.extend(
        [
            {
                "model_id": "M3-S",
                "name": "Dataset 3 + spatial neighbors",
                "data": df_new_2,
                "neighbor_mode": "spatial",
            },
            {
                "model_id": "M3-ST",
                "name": "Dataset 3 + spatial-time neighbors",
                "data": df_new_2,
                "neighbor_mode": "spacetime",
            },
        ]
    )

dataset_specs.append(
    {"model_id": "M4", "name": "Dataset 4", "data": df_new_3, "neighbor_mode": None}
)

# =============================================================================
# Chronological train/test evaluation across datasets
# =============================================================================

# build_model_pipeline is imported from preprocessing.recipes_pipelined above;
# calculate_evaluation_metrics is defined in this script.
# =============================================================================
# Chronological train/test evaluation across datasets
# =============================================================================

in_sample_results = []
oos_results = []
oos_results_2 = []

# --- 1. DEFINE THE UNFILTERED BASELINE TEST SET ---
# Sort and split Dataset 1 exactly as it will be in the loop.
df_baseline = dataset_specs[0]["data"].copy().sort_values(by="meta_sale_date")
baseline_split_idx = int(0.8 * len(df_baseline))
baseline_test_data_raw = df_baseline.iloc[baseline_split_idx:].copy()


for spec in dataset_specs:
    model_id = spec["model_id"]
    dataset_name = spec["name"]
    neighbor_mode = spec["neighbor_mode"]
    print(f"--- Processing {model_id}: {dataset_name} | neighbor_mode={neighbor_mode or 'none'} ---")

    # Sort chronologically before the 80/20 train/test split.
    dataset = spec["data"].copy().sort_values(by="meta_sale_date")

    split_idx = int(0.8 * len(dataset))
    train_data = dataset.iloc[:split_idx].copy()
    test_data = dataset.iloc[split_idx:].copy()
    baseline_test_data = baseline_test_data_raw.copy()

    # Add neighbor-price features after splitting so test targets are not used as features.
    train_data, test_data, baseline_test_data, neighbor_feature_cols = add_benchmark_neighbor_features(
        train_data,
        test_data,
        baseline_test_data,
        neighbor_mode,
    )

    dataset_predictor_cols = [c for c in predictor_cols if c in train_data.columns]
    dataset_predictor_cols += [c for c in neighbor_feature_cols if c not in dataset_predictor_cols]
    dataset_categorical_cols = [c for c in categorical_cols if c in train_data.columns]

    # Build a fresh pipeline for each dataset to avoid shared fitted state.
    linear_pipeline = build_model_pipeline(
        pred_vars=dataset_predictor_cols,
        cat_vars=dataset_categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    # Fit only on training data; transform test data without refitting.
    train_transformed = linear_pipeline.fit_transform(
        train_data,
        train_data["meta_sale_price"],
    )
    test_transformed = linear_pipeline.transform(test_data)
    
    # --- 2. TRANSFORM BASELINE TEST DATA ---
    # Apply the current loop's pipeline to the unfiltered baseline test data.
    baseline_test_transformed = linear_pipeline.transform(baseline_test_data)

    # Separate features and target after pipeline transformation.
    id_vars = params["model"]["predictor"]["id"]
    if isinstance(id_vars, str):
        id_vars = [id_vars]

    cols_to_drop = ["meta_sale_price", "meta_sale_date"] + id_vars

    X_train = train_transformed.drop(columns=cols_to_drop, errors="ignore")
    y_train = (
        train_transformed["meta_sale_price"]
        if "meta_sale_price" in train_transformed
        else train_data["meta_sale_price"]
    )

    X_test = test_transformed.drop(columns=cols_to_drop, errors="ignore")
    y_test = (
        test_transformed["meta_sale_price"]
        if "meta_sale_price" in test_transformed
        else test_data["meta_sale_price"]
    )

    # --- 3. EXTRACT FEATURES FROM TRANSFORMED BASELINE ---
    X_test_2 = baseline_test_transformed.drop(columns=cols_to_drop, errors="ignore")
    y_test_2 = (
        baseline_test_transformed["meta_sale_price"]
        if "meta_sale_price" in baseline_test_transformed
        else baseline_test_data["meta_sale_price"]
    )

    # Fit linear regression and evaluate in-sample and out-of-sample predictions.
    model = LinearRegression()
    model.fit(X_train, np.log(y_train))

    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)
    y_test_pred_log_2 = model.predict(X_test_2)

    y_train_pred = np.exp(y_train_pred_log)
    y_test_pred = np.exp(y_test_pred_log)
    y_test_pred_2 = np.exp(y_test_pred_log_2)

    train_metrics = calculate_evaluation_metrics(y_train, y_train_pred, y_train=y_train)
    test_metrics = calculate_evaluation_metrics(y_test, y_test_pred, y_train=y_train)
    test_metrics_2 = calculate_evaluation_metrics(y_test_2, y_test_pred_2, y_train=y_train)

    for metrics in (train_metrics, test_metrics, test_metrics_2):
        metrics["Model ID"] = model_id
        metrics["Dataset"] = dataset_name
        metrics["Neighbor Mode"] = neighbor_mode or "none"

    in_sample_results.append(train_metrics)
    oos_results.append(test_metrics)
    oos_results_2.append(test_metrics_2)

# Aggregate and display final model-comparison tables.
in_sample_df = pd.DataFrame(in_sample_results).set_index("Model ID")
oos_df = pd.DataFrame(oos_results).set_index("Model ID")
oos_df_2 = pd.DataFrame(oos_results_2).set_index("Model ID")

# Identifier columns plus metrics to show in the final tables.
metrics_to_show = [
    "Dataset",
    "Neighbor Mode",
    "R2",
    "RMSE",
    "MAE",
    "MAPE",
    "MdAPE",
    "COD",
    "COV_IAAO",
    "VEI",
    "PRD",
    "PRB",
]
in_sample_df = in_sample_df[metrics_to_show]
oos_df = oos_df[metrics_to_show]
oos_df_2 = oos_df_2[metrics_to_show]

print("\n" + "=" * 50)
print("IN-SAMPLE (TRAIN) METRICS")
print("=" * 50)
display(in_sample_df)

print("\n" + "=" * 50)
print("OUT-OF-SAMPLE (TEST) METRICS [Current Dataset]")
print("=" * 50)
display(oos_df)

print("\n" + "=" * 50)
print("OUT-OF-SAMPLE (TEST) METRICS [Baseline / Dataset 1]")
print("=" * 50)
display(oos_df_2)


