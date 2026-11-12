import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .utility import drop_time, drop_constant, drop_near_constant

# RobustScorer versions (see FINDINGS.md). "1" is BARO as published and stays the default,
# so the paper's results reproduce. "2" fixes two scoring effects found on RCAEval RE2-OB:
#   - a series constant before the inject time was dropped (DataFrame path: the two halves
#     were preprocessed apart, then intersected), however much it moved after it;
#   - a zero IQR before the inject time gave a scale of 1, so the score was in raw units
#     (a bursty disk-I/O series in bytes outranked everything).
SCORER_VERSION = "1"
SCORER_VERSIONS = ("1", "2")


def _check_version(version):
    if version not in SCORER_VERSIONS:
        raise ValueError(f"unknown scorer version {version!r}; choose one of {', '.join(SCORER_VERSIONS)}")


def robust_score(normal, anomal, window, flat_fraction=0.01):
    """How far anomal rises above normal, in robust units (scorer version 2).

    BARO's score, max((anomal - median(normal)) / IQR(normal)), with a fallback for an IQR
    of zero: the standard deviation of normal, and for a series flat before the inject time
    flat_fraction of its range over the window, so a step away from a flat baseline scores
    about 1 / flat_fraction (100) whatever its units.
    """
    normal, anomal, window = (np.asarray(x, dtype=float) for x in (normal, anomal, window))
    normal, anomal, window = (x[~np.isnan(x)] for x in (normal, anomal, window))
    if not len(normal) or not len(anomal):
        return 0.0
    # a scale this small is rounding error (the IQR of a constant comes out as ~1e-18)
    tiny = 1e-9 * max(float(np.max(np.abs(window))), np.finfo(float).tiny)
    median = np.median(normal)
    q1, q3 = np.percentile(normal, [25, 75])
    scale = q3 - q1
    if scale <= tiny:
        scale = np.std(normal)
    if scale <= tiny:
        scale = flat_fraction * (np.max(window) - np.min(window))
    if scale <= tiny:
        return 0.0
    return float((np.max(anomal) - median) / scale)


def select_useful_cols(data):
    """Select useful columns from the dataset based on certain criteria.
    
    Parameters:
    - data : pandas.DataFrame
        The dataset to select columns from.
        
    Returns:
    - selected_cols : list
        A list of selected column names.
    """
    selected_cols = []
    for c in data.columns:
        # keep time
        if "time" in c:
            selected_cols.append(c)

        # cpu
        if c.endswith("_cpu") and data[c].std() > 1:
            selected_cols.append(c)

        # mem
        if c.endswith("_mem") and data[c].std() > 1:
            selected_cols.append(c)

        # latency
        # if ("lat50" in c or "latency" in c) and (data[c] * 1000).std() > 10:
        if "lat50" in c and (data[c] * 1000).std() > 10:
            selected_cols.append(c)
    return selected_cols



def drop_extra(df: pd.DataFrame):
    """Drop extra columns from the DataFrame.
    
    Parameters:
    - df : pandas.DataFrame
        The DataFrame to remove extra columns from.
        
    Returns:
    - df : pandas.DataFrame
        The DataFrame after removing extra columns.
    """
    if "time.1" in df:
        df = df.drop(columns=["time.1"])

    # remove cols has "frontend-external" in name
    # remove cols start with "main_" or "PassthroughCluster_", etc.
    for col in df.columns:
        if (
            "frontend-external" in col
            or col.startswith("main_")
            or col.startswith("PassthroughCluster_")
            or col.startswith("redis_")
            or col.startswith("rabbitmq")
            or col.startswith("queue")
            or col.startswith("session")
            or col.startswith("istio-proxy")
        ):
            df = df.drop(columns=[col])

    return df




def convert_mem_mb(df: pd.DataFrame):
    """Convert memory values in the DataFrame to MBs.
    
    Parameters:
    - df : pandas.DataFrame
        The DataFrame containing memory values.
        
    Returns:
    - df : pandas.DataFrame
        The DataFrame with memory values converted to MBs.
    """
    # Convert memory to MBs
    def update_mem(x):
        if not x.name.endswith("_mem"):
            return x
        x /= 1e6
        # x = x.astype(int)
        return x

    return df.apply(update_mem)


def preprocess(data, dataset=None, dk_select_useful=False):
    """Preprocess the dataset.
    
    Parameters:
    - data : pandas.DataFrame
        The dataset to preprocess.
    - dataset : str, optional
        The dataset name. Default is None.
    - dk_select_useful : bool, optional
        Whether to select useful columns. Default is False.
        
    Returns:
    - data : pandas.DataFrame
        The preprocessed dataset.
    """
    data = drop_constant(drop_time(data))
    data = convert_mem_mb(data)

    if dk_select_useful is True:
        data = drop_extra(data)
        data = drop_near_constant(data)
        data = data[select_useful_cols(data)]
    return data





def nsigma(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    """Perform nsigma analysis on the dataset.
    
    Parameters:
    - data : pandas.DataFrame
        The dataset to perform nsigma analysis on.
    - inject_time : int, optional
        The time of injection of anomalies. Default is None.
    - dataset : str, optional
        The dataset name. Default is None.
    - num_loop : int, optional
        Number of loops. Default is None.
    - sli : int, optional
        SLI (Service Level Indicator). Default is None.
    - anomalies : list, optional
        List of anomalies. Default is None.
    - kwargs : dict
        Additional keyword arguments.
        
    Returns:
    - dict
        A dictionary containing node names and ranks.
    """
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = StandardScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


def robust_scorer_dict(metrics: dict, inject_time: int, version: str = SCORER_VERSION) -> list:
    """Fast RobustScorer for RCAEval JSON metrics format.

    Directly processes metrics dict without DataFrame conversion.
    Uses IQR-based scoring (same algorithm as sklearn RobustScaler).

    Parameters:
    - metrics : dict
        Dictionary mapping metric names to time-series data.
        Format: {"metric_name": [[timestamp, value], ...], ...}
    - inject_time : int
        Unix timestamp of fault injection.

    - version : str
        Scorer version. "1": |max(post) - median| / IQR, with IQR 0 replaced by 1 (raw units).
        "2": robust_score (rises only, as in the DataFrame path; no raw-unit fallback);
        series constant over the whole window are skipped.

    Returns:
    - list
        List of (metric_name, score) tuples, sorted descending by score.
    """
    _check_version(version)
    ranked_list = []

    for metric_name, metric_data in metrics.items():
        # Split by inject_time
        pre_data = [float(v) for ts, v in metric_data if ts < inject_time]
        post_data = [float(v) for ts, v in metric_data if ts >= inject_time]

        # Skip metrics with insufficient pre-fault data or no post-fault data
        if len(pre_data) < 4 or not post_data:
            continue

        if version == "2":
            window = np.array(pre_data + post_data)
            if np.isnan(window).all() or np.nanmin(window) == np.nanmax(window):
                continue
            ranked_list.append((metric_name, robust_score(pre_data, post_data, window)))
            continue

        # Calculate median and IQR from pre-fault data
        pre_data = sorted(pre_data)
        n = len(pre_data)
        median = pre_data[n // 2]
        q75 = pre_data[int(n * 0.75)]
        q25 = pre_data[int(n * 0.25)]
        iqr = q75 - q25

        # Handle zero IQR (constant values)
        if iqr == 0:
            iqr = 1

        # Score = max deviation in post-fault period normalized by IQR
        max_val = max(post_data)
        score = abs(max_val - median) / iqr
        ranked_list.append((metric_name, score))

    ranked_list.sort(key=lambda x: x[1], reverse=True)
    return ranked_list


def robust_scorer(
    data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None,
    version=SCORER_VERSION, **kwargs
):
    """Perform root cause analysis using RobustScorer.

    Supports both input formats:
    - pd.DataFrame (FSE CSV format) - existing behavior with sklearn RobustScaler
    - dict (RCAEval JSON format) - fast path via robust_scorer_dict()

    Parameters:
    - data : pandas.DataFrame or dict
        The data to perform RobustScorer on.
        If dict: metrics in format {"metric_name": [[ts, val], ...], ...}
        If DataFrame: requires "time" column for splitting
    - inject_time : int, optional
        The time of fault injection. Required for dict input.
    - dataset : str, optional
        The dataset name. Default is None.
    - num_loop : int, optional
        Number of loops. Default is None. Just for future API compatible
    - sli : int, optional
        SLI (Service Level Indicator). Default is None. Just for future API compatible
    - anomalies : list, optional
        List of anomalies. Default is None. Only used for DataFrame input.
    - version : str, optional
        Scorer version, "1" (default, as published) or "2" (see SCORER_VERSION).
    - kwargs : dict
        Additional keyword arguments.

    Returns:
    - dict
        A dictionary containing:
        - node_names: list of metric names
        - ranks: ranked list of root causes (metric names only)
        - scores: list of (metric_name, score) tuples (dict input, and DataFrame input with version "2")
    """
    _check_version(version)
    # Auto-detect input format
    if isinstance(data, dict):
        # Fast path for JSON metrics (RCAEval format)
        if inject_time is None:
            raise ValueError("inject_time is required for dict input")

        ranked = robust_scorer_dict(data, inject_time, version=version)
        return {
            "node_names": list(data.keys()),
            "ranks": [m for m, _ in ranked],
            "scores": ranked,
        }

    # Existing DataFrame path (FSE format)
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    if version == "2":
        # preprocess the whole window once, so a series is dropped only if it is constant
        # throughout, not when it is constant before the inject time
        n_normal = len(normal_df)
        values = preprocess(
            data=pd.concat([normal_df, anomal_df]), dataset=dataset,
            dk_select_useful=kwargs.get("dk_select_useful", False),
        )
        scored = []
        for col in values.columns:
            v = values[col].to_numpy(dtype=float)
            scored.append((col, robust_score(v[:n_normal], v[n_normal:], v)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            "node_names": values.columns.to_list(),
            "ranks": [c for c, _ in scored],
            "scores": scored,
        }

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }


