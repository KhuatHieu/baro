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
# "3" (experimental, FINDINGS.md F2) scores as "2" and ranks in tiers (rank_scored): a
#     sustained step up from a quiet baseline first, then series whose normal period has spread
#     by z-score, then quiet series without a step (whose z-score is not meaningful).
SCORER_VERSION = "1"
SCORER_VERSIONS = ("1", "2", "3")
STEP_RANKS = ("tier", "none")
QUIET_RANKS = ("last", "score")


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


def sustained_step(normal, anomal, min_persistence=0.9, min_rise=1.0, baseline="quiet"):
    """A step up from a flat baseline that holds after the inject time (scorer version 3).

    Returns {"persistence", "rise"} when normal has a baseline (baseline "quiet": no
    interquartile spread, baseline = its median, so a few blips are allowed; "flat": strictly
    constant), the series differs from that baseline in at least min_persistence of the
    anomalous samples, and its anomalous median is
    at least min_rise above the baseline (relative, so 1.0 = at least doubled; any rise from a
    zero baseline counts, as rise = inf). Otherwise None. On RE2-OB the defaults flag the root
    cause's disk I/O and socket count in every disk and socket fault (smallest rise there: 1.44)
    and 7 other series in 90 cases, 6 of them on the root-cause service (FINDINGS.md F2).
    """
    normal, anomal = (np.asarray(x, dtype=float) for x in (normal, anomal))
    normal, anomal = normal[~np.isnan(normal)], anomal[~np.isnan(anomal)]
    if not len(normal) or not len(anomal):
        return None
    if baseline == "flat":
        if np.ptp(normal) > 0:
            return None
        base = normal[0]
    else:  # "quiet": no interquartile spread; a few blips before the inject time are allowed
        if not is_quiet(normal):
            return None
        base = float(np.median(normal))
    tol = 1e-9 * max(abs(base), float(np.max(np.abs(anomal))), np.finfo(float).tiny)
    persistence = float(np.mean(np.abs(anomal - base) > tol))
    median = float(np.median(anomal))
    if median - base <= tol:
        return None
    rise = float("inf") if abs(base) <= tol else (median - base) / abs(base)
    if persistence < min_persistence or rise < min_rise:
        return None
    return {"persistence": round(persistence, 3), "rise": rise}


def is_quiet(normal):
    """True when normal has (almost) no interquartile spread, so a z-score against it is not
    meaningful: robust_score then falls back to the standard deviation or the window range."""
    normal = np.asarray(normal, dtype=float)
    normal = normal[~np.isnan(normal)]
    if not len(normal):
        return True
    q1, q3 = np.percentile(normal, [25, 75])
    return q3 - q1 <= 1e-9 * max(float(np.max(np.abs(normal))), np.finfo(float).tiny)


def rank_scored(scored, steps, step_rank="tier", quiet=(), quiet_rank="last"):
    """Order (name, score) pairs (scorer version 3), in tiers:
    1. with step_rank "tier", series with a sustained step (largest rise first);
    2. series whose normal period has spread, by score (BARO's z-score);
    3. with quiet_rank "last", quiet series (is_quiet) without a step, by their fallback score.
    step_rank "none" and quiet_rank "score" turn tiers 1 and 3 off."""
    if step_rank not in STEP_RANKS:
        raise ValueError(f"unknown step_rank {step_rank!r}; choose one of {', '.join(STEP_RANKS)}")
    if quiet_rank not in QUIET_RANKS:
        raise ValueError(f"unknown quiet_rank {quiet_rank!r}; choose one of {', '.join(QUIET_RANKS)}")
    quiet = set(quiet)

    def key(item):
        name, score = item
        if step_rank == "tier" and name in steps:
            return (2, steps[name]["rise"], score)
        if quiet_rank == "last" and name in quiet:
            return (0, 0, score)
        return (1, 0, score)

    return sorted(scored, key=key, reverse=True)


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


def robust_scorer_dict(metrics: dict, inject_time: int, version: str = SCORER_VERSION,
                       step_rank: str = "tier", steps: dict = None, quiet_rank: str = "last") -> list:
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
        "3": as "2", ordered by rank_scored(step_rank, quiet_rank); flagged series go into
        `steps` if given.

    Returns:
    - list
        List of (metric_name, score) tuples, sorted descending by score.
    """
    _check_version(version)
    ranked_list = []
    found = {} if steps is None else steps
    quiet = set()

    for metric_name, metric_data in metrics.items():
        # Split by inject_time
        pre_data = [float(v) for ts, v in metric_data if ts < inject_time]
        post_data = [float(v) for ts, v in metric_data if ts >= inject_time]

        # Skip metrics with insufficient pre-fault data or no post-fault data
        if len(pre_data) < 4 or not post_data:
            continue

        if version in ("2", "3"):
            window = np.array(pre_data + post_data)
            if np.isnan(window).all() or np.nanmin(window) == np.nanmax(window):
                continue
            ranked_list.append((metric_name, robust_score(pre_data, post_data, window)))
            if version == "3":
                step = sustained_step(pre_data, post_data)
                if step:
                    found[metric_name] = step
                if is_quiet(pre_data):
                    quiet.add(metric_name)
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

    if version == "3":
        return rank_scored(ranked_list, found, step_rank, quiet, quiet_rank)
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
        Scorer version, "1" (default, as published), "2" or "3" (see SCORER_VERSION).
        Version "3" also takes step_rank ("tier" or "none"), quiet_rank ("last" or "score"),
        min_persistence and min_rise (see sustained_step), and returns "steps".
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

        steps = {}
        ranked = robust_scorer_dict(data, inject_time, version=version,
                                    step_rank=kwargs.get("step_rank", "tier"), steps=steps,
                                    quiet_rank=kwargs.get("quiet_rank", "last"))
        result = {
            "node_names": list(data.keys()),
            "ranks": [m for m, _ in ranked],
            "scores": ranked,
        }
        if version == "3":
            result["steps"] = steps
        return result

    # Existing DataFrame path (FSE format)
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        # anomal_df is the rest
        anomal_df = data.tail(len(data) - anomalies[0])

    if version in ("2", "3"):
        # preprocess the whole window once, so a series is dropped only if it is constant
        # throughout, not when it is constant before the inject time
        n_normal = len(normal_df)
        values = preprocess(
            data=pd.concat([normal_df, anomal_df]), dataset=dataset,
            dk_select_useful=kwargs.get("dk_select_useful", False),
        )
        scored, steps, quiet = [], {}, set()
        for col in values.columns:
            v = values[col].to_numpy(dtype=float)
            scored.append((col, robust_score(v[:n_normal], v[n_normal:], v)))
            if version == "3":
                step = sustained_step(v[:n_normal], v[n_normal:], **{
                    k: kwargs[k] for k in ("min_persistence", "min_rise", "baseline") if k in kwargs})
                if step:
                    steps[col] = step
                if is_quiet(v[:n_normal]):
                    quiet.add(col)
        if version == "3":
            scored = rank_scored(scored, steps, kwargs.get("step_rank", "tier"), quiet,
                                 kwargs.get("quiet_rank", "last"))
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
        result = {
            "node_names": values.columns.to_list(),
            "ranks": [c for c, _ in scored],
            "scores": scored,
        }
        if version == "3":
            result["steps"] = steps
        return result

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


