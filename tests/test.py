"""Tests."""
import os
import shutil
from os import path

import numpy as np
import pandas as pd
import pytest
import tempfile

from baro.anomaly_detection import nsigma, bocpd
from baro.root_cause_analysis import SCORER_VERSION, robust_score, robust_scorer, robust_scorer_dict
from baro.utility import (
    visualize_metrics,
    download_data,
    drop_constant,
    download_online_boutique_dataset,
    download_sock_shop_dataset,
    download_train_ticket_dataset,
)

def test_download_dataset():
    """Test download dataset."""
    local_path = tempfile.NamedTemporaryFile().name
    download_online_boutique_dataset(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)
    
    local_path = tempfile.NamedTemporaryFile().name
    download_sock_shop_dataset(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)    
    
    local_path = tempfile.NamedTemporaryFile().name
    download_train_ticket_dataset(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)


def test_nsigma_basic():
    """Test nsigma basic."""
    time_col = np.arange(0, 1000, 1)
    normal_latency = np.random.normal(3, 1, 500)
    normal_latency = np.clip(normal_latency, 1, 5)    
    
    abnormal_latency = np.random.normal(50, 1, 500)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # make df from time_col and latency
    df = pd.DataFrame({'time': time_col, 'latency': latency})
    anomalies = nsigma(df, startsfrom=300)
    assert abs(anomalies[0] - 500) < 10, anomalies

     
def test_bocpd_basic():
    """Test bocpd basic."""
    time_col = np.arange(0, 200, 1)
    normal_latency = np.random.normal(3, 1, 100)
    normal_latency = np.clip(normal_latency, 1, 5)    
    
    abnormal_latency = np.random.normal(50, 1, 100)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # make df from time_col and latency
    df = pd.DataFrame({'time': time_col, 'latency': latency})
    
    anomalies = bocpd(df)
    assert abs(anomalies[0] - 100) < 10, anomalies


def test_bocpd_no_latency_error_cols():
    """Test bocpd when no latency or error columns are present."""
    np.random.seed(0)
    time_col = np.arange(0, 200, 1)
    normal_metric = np.random.normal(3, 1, 100)
    normal_metric = np.clip(normal_metric, 1, 5)    
    
    abnormal_metric = np.random.normal(50, 1, 100)
    metric = np.concatenate((normal_metric, abnormal_metric))
    
    # make df with no latency or error columns
    df = pd.DataFrame({'time': time_col, 'cpu_usage': metric})
    
    # This should trigger the warning and still work
    anomalies = bocpd(df)
    assert len(anomalies) > 0, "Should still detect anomalies even without latency/error columns"
    assert abs(anomalies[0] - 100) < 10, anomalies


def test_bocpd_only_time_col():
    """Test bocpd when only time column is present."""
    time_col = np.arange(0, 100, 1)
    df = pd.DataFrame({'time': time_col})
    
    # This should trigger warning and return empty list
    anomalies = bocpd(df)
    assert anomalies == [], "Should return empty list when no non-time columns"

def test_baro():
    """Test BARO end-to-end"""
    local_path = tempfile.NamedTemporaryFile().name
    download_data(local_path=local_path)
    df = pd.read_csv(local_path)
    df = df[60:660].reset_index(drop=True)

    # select latency and error rate
    time_col = pd.Series(range(df.shape[0])) # df["time"]
    selected_cols = [c for c in df.columns if "latency-50" in c or "error" in c]
    selected_df = drop_constant(df[selected_cols])
    selected_df.insert(0, "time", time_col)
    
    # anomaly detection
    anomalies = bocpd(selected_df)
    
    # root cause analysis
    ranks = robust_scorer(df, anomalies=anomalies)["ranks"]
    
    # check if cartservice is in the top 5
    service_ranks = [r.split("_")[0] for r in ranks]
    assert "cartservice" in service_ranks[:5]


def test_bocpd_missing_data_normal_period():
    """Test bocpd when metrics data is missing during normal period."""
    np.random.seed(42)
    time_col = np.arange(0, 200, 1)
    
    # Create normal period with some missing data
    normal_latency = np.random.normal(3, 1, 100)
    normal_latency = np.clip(normal_latency, 1, 5)
    
    # Introduce missing data in normal period (first 50 timesteps)
    normal_latency[:30] = np.nan
    
    # Create anomalous period (intact data)
    abnormal_latency = np.random.normal(50, 1, 100)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # Make dataframe
    df = pd.DataFrame({'time': time_col, 'service_latency': latency})
    
    # Should still detect anomalies despite missing normal period data
    anomalies = bocpd(df)
    
    # Should detect anomaly around timestep 100, allowing for some variance due to missing data
    assert len(anomalies) > 0, "Should detect anomalies even with missing normal period data"
    assert abs(anomalies[0] - 100) < 50, f"Expected anomaly around 100, got {anomalies[0]}"


def test_bocpd_completely_missing_normal_period():
    """Test bocpd when all metrics data is missing during normal period."""
    np.random.seed(42)
    time_col = np.arange(0, 200, 1)
    
    # Create completely missing normal period
    normal_latency = np.full(100, np.nan)
    
    # Create anomalous period (intact data)
    abnormal_latency = np.random.normal(50, 1, 100)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # Make dataframe
    df = pd.DataFrame({'time': time_col, 'service_latency': latency})
    
    # Should still work, possibly with warnings
    anomalies = bocpd(df)
    
    # Should still detect some change point when data becomes available
    assert len(anomalies) > 0, "Should detect anomalies even with completely missing normal period"


def test_robust_scorer_missing_normal_period():
    """Test robust_scorer when normal period data is missing."""
    np.random.seed(42)
    time_col = np.arange(0, 200, 1)
    
    # Create normal period with missing data
    normal_cpu = np.random.normal(30, 5, 100)
    normal_cpu[:50] = np.nan  # Missing first half of normal period
    
    normal_mem = np.random.normal(500, 50, 100)
    normal_mem[25:75] = np.nan  # Missing middle portion
    
    # Create anomalous period
    abnormal_cpu = np.random.normal(80, 10, 100)
    abnormal_mem = np.random.normal(1000, 100, 100)
    
    cpu = np.concatenate((normal_cpu, abnormal_cpu))
    mem = np.concatenate((normal_mem, abnormal_mem))
    
    # Make dataframe
    df = pd.DataFrame({
        'time': time_col,
        'service_cpu': cpu,
        'service_mem': mem
    })
    
    # Test with anomalies detected at timestep 100
    anomalies = [100]
    
    # Should still work with missing normal period data
    result = robust_scorer(df, anomalies=anomalies)
    ranks = result["ranks"]
    
    # Should return some ranking despite missing normal period data
    assert len(ranks) > 0, "Should return rankings even with missing normal period data"
    assert len(result["node_names"]) > 0, "Should return node names"


def test_robust_scorer_insufficient_normal_period():
    """Test robust_scorer when normal period has insufficient data."""
    np.random.seed(42)
    time_col = np.arange(0, 50, 1)  # Very short dataset
    
    # Create very short normal period (only 20 timesteps)
    normal_cpu = np.random.normal(30, 5, 20)
    normal_mem = np.random.normal(500, 50, 20)
    
    # Create short anomalous period
    abnormal_cpu = np.random.normal(80, 10, 30)
    abnormal_mem = np.random.normal(1000, 100, 30)
    
    cpu = np.concatenate((normal_cpu, abnormal_cpu))
    mem = np.concatenate((normal_mem, abnormal_mem))
    
    # Make dataframe
    df = pd.DataFrame({
        'time': time_col,
        'service_cpu': cpu,
        'service_mem': mem
    })
    
    # Test with anomalies detected at timestep 20
    anomalies = [20]
    
    # Should still work with very short normal period
    result = robust_scorer(df, anomalies=anomalies)
    ranks = result["ranks"]
    
    # Should return some ranking despite insufficient normal period data
    assert len(ranks) > 0, "Should return rankings even with insufficient normal period data"
    

def test_bocpd_zero_variance_normal_period():
    """Test bocpd when normal period has zero variance (constant values)."""
    np.random.seed(42)
    time_col = np.arange(0, 200, 1)
    
    # Create constant normal period
    normal_latency = np.full(100, 3.0)
    
    # Create anomalous period
    abnormal_latency = np.random.normal(50, 1, 100)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # Make dataframe
    df = pd.DataFrame({'time': time_col, 'service_latency': latency})
    
    # Should detect anomalies when moving from constant to variable
    anomalies = bocpd(df)
    
    # Should detect anomaly around timestep 100
    assert len(anomalies) > 0, "Should detect anomalies when moving from constant to variable data"
    assert abs(anomalies[0] - 100) < 20, f"Expected anomaly around 100, got {anomalies[0]}"


def test_bocpd_all_nan_data():
    """Test bocpd when all data is NaN."""
    time_col = np.arange(0, 100, 1)
    data = pd.DataFrame({'time': time_col, 'service_latency': np.full(100, np.nan)})
    
    # Should handle all NaN data gracefully
    anomalies = bocpd(data)
    
    # Should return some result (even if it's not meaningful)
    assert isinstance(anomalies, list), "Should return a list even with all NaN data"


def test_robust_scorer_all_nan_normal_period():
    """Test robust_scorer when normal period is all NaN."""
    np.random.seed(42)
    time_col = np.arange(0, 50, 1)
    
    # Create all NaN normal period
    normal_data = np.full(25, np.nan)
    
    # Create valid anomalous period
    abnormal_data = np.random.normal(80, 10, 25)
    
    data = np.concatenate((normal_data, abnormal_data))
    
    # Make dataframe
    df = pd.DataFrame({
        'time': time_col,
        'service_cpu': data
    })
    
    # Test with anomalies detected at timestep 25
    anomalies = [25]
    
    # Should handle all NaN normal period gracefully
    result = robust_scorer(df, anomalies=anomalies)
    
    # Should return some result (ranks might be based on limited data)
    assert isinstance(result, dict), "Should return a dict even with all NaN normal period"
    assert "ranks" in result, "Should have ranks key"
    assert "node_names" in result, "Should have node_names key"


def test_robust_scorer_mixed_availability():
    """Test robust_scorer with mixed data availability across metrics."""
    np.random.seed(42)
    time_col = np.arange(0, 100, 1)
    
    # Create mixed availability scenario
    # Service A: missing in normal period, available in anomalous period
    service_a_normal = np.full(50, np.nan)
    service_a_anomal = np.random.normal(80, 10, 50)
    service_a = np.concatenate((service_a_normal, service_a_anomal))
    
    # Service B: available in normal period, missing in anomalous period
    service_b_normal = np.random.normal(30, 5, 50)
    service_b_anomal = np.full(50, np.nan)
    service_b = np.concatenate((service_b_normal, service_b_anomal))
    
    # Service C: available in both periods
    service_c_normal = np.random.normal(500, 50, 50)
    service_c_anomal = np.random.normal(1000, 100, 50)
    service_c = np.concatenate((service_c_normal, service_c_anomal))
    
    # Make dataframe
    df = pd.DataFrame({
        'time': time_col,
        'serviceA_cpu': service_a,
        'serviceB_mem': service_b,
        'serviceC_latency': service_c
    })
    
    # Test with anomalies detected at timestep 50
    anomalies = [50]
    
    # Should handle mixed availability gracefully
    result = robust_scorer(df, anomalies=anomalies)
    ranks = result["ranks"]
    
    # Should return rankings for available metrics
    assert len(ranks) > 0, "Should return rankings for available metrics"
    
    # Service C should be rankable since it has data in both periods
    service_c_in_ranks = any('serviceC' in rank for rank in ranks)
    assert service_c_in_ranks, "ServiceC should be in rankings as it has data in both periods"


def test_bocpd_sporadic_missing_data():
    """Test bocpd with sporadic missing data throughout the time series."""
    np.random.seed(42)
    time_col = np.arange(0, 200, 1)
    
    # Create base data
    normal_latency = np.random.normal(3, 1, 100)
    abnormal_latency = np.random.normal(50, 1, 100)
    latency = np.concatenate((normal_latency, abnormal_latency))
    
    # Add sporadic missing data (every 10th point)
    sporadic_missing = np.arange(0, 200, 10)
    latency[sporadic_missing] = np.nan
    
    # Make dataframe
    df = pd.DataFrame({'time': time_col, 'service_latency': latency})
    
    # Should still detect anomalies despite sporadic missing data
    anomalies = bocpd(df)
    
    # Should detect anomaly around timestep 100
    assert len(anomalies) > 0, "Should detect anomalies despite sporadic missing data"
    # Allow for more variance due to missing data
    assert abs(anomalies[0] - 100) < 50, f"Expected anomaly around 100, got {anomalies[0]}"


def test_bocpd_constant_data_edge_case():
    """Test bocpd with constant data (edge case that could cause division by zero)."""
    time_col = np.arange(0, 100, 1)
    data = pd.DataFrame({'time': time_col, 'service_latency': np.full(100, 5.0)})
    
    # Should handle constant data gracefully (no anomalies detected)
    anomalies = bocpd(data)
    
    # Should return empty list or handle gracefully
    assert isinstance(anomalies, list), "Should return a list even with constant data"


def test_bocpd_single_data_point():
    """Test bocpd with single data point (edge case)."""
    data = pd.DataFrame({'time': [0], 'service_latency': [5.0]})
    
    # Should handle single data point gracefully
    anomalies = bocpd(data)
    
    # Should return empty list or handle gracefully
    assert isinstance(anomalies, list), "Should return a list even with single data point"


def test_bocpd_empty_dataframe():
    """Test bocpd with empty dataframe."""
    data = pd.DataFrame({'time': [], 'service_latency': []})
    
    # Should handle empty dataframe gracefully
    anomalies = bocpd(data)
    
    # Should return empty list
    assert anomalies == [], "Should return empty list for empty dataframe"


def test_end_to_end_missing_normal_period():
    """Test end-to-end BARO workflow with missing normal period data."""
    np.random.seed(42)
    
    # Simulate a realistic scenario where monitoring was down during normal period
    time_col = np.arange(0, 300, 1)
    
    # Create realistic microservice metrics
    # Normal period: missing data for first 100 timesteps due to monitoring failure
    normal_cpu = np.full(150, np.nan)
    normal_cpu[50:] = np.random.normal(30, 5, 100)  # Monitoring comes online at timestep 50
    
    normal_mem = np.full(150, np.nan)
    normal_mem[75:] = np.random.normal(500, 50, 75)  # Memory monitoring comes online later
    
    normal_latency = np.random.normal(50, 10, 150)
    normal_latency[:100] = np.nan  # Latency monitoring also fails initially
    
    # Anomalous period: incident occurs, all monitoring is working
    abnormal_cpu = np.random.normal(80, 10, 150)
    abnormal_mem = np.random.normal(1200, 100, 150)
    abnormal_latency = np.random.normal(200, 20, 150)
    
    # Combine periods
    cpu_data = np.concatenate((normal_cpu, abnormal_cpu))
    mem_data = np.concatenate((normal_mem, abnormal_mem))
    latency_data = np.concatenate((normal_latency, abnormal_latency))
    
    # Create dataframe
    df = pd.DataFrame({
        'time': time_col,
        'cartservice_cpu': cpu_data,
        'cartservice_mem': mem_data,
        'cartservice_latency': latency_data,
        'paymentservice_cpu': np.random.normal(25, 3, 300),  # This service has full data
        'paymentservice_latency': np.random.normal(30, 5, 300)
    })
    
    # Test anomaly detection despite missing normal period data
    anomalies = bocpd(df)
    
    # Should detect anomalies when incident occurs (around timestep 150)
    assert len(anomalies) > 0, "Should detect anomalies despite missing normal period data"
    
    # Test root cause analysis
    result = robust_scorer(df, anomalies=anomalies)
    ranks = result["ranks"]
    
    # Should return meaningful rankings despite missing normal period data
    assert len(ranks) > 0, "Should return rankings despite missing normal period data"
    
    # Services with data in both periods should be rankable
    cartservice_metrics = [r for r in ranks if 'cartservice' in r]
    paymentservice_metrics = [r for r in ranks if 'paymentservice' in r]
    
    # Should have some rankings for both services
    assert len(cartservice_metrics) > 0 or len(paymentservice_metrics) > 0, \
        "Should have rankings for services with available data"
    
    print(f"Detected anomalies at: {anomalies}")
    print(f"Top 3 root causes: {ranks[:3]}")
    print(f"Available metrics: {result['node_names']}")
    
    # Verify the system provides meaningful output for operational use
    assert len(result['node_names']) > 0, "Should identify available metrics for analysis"
    

# --- RobustScorer version 2 (FINDINGS.md, F1) ---

def _flat_before_df(n=100, inject=50):
    """cart_cpu rises a little (noisy baseline); cart_diskio is 0 before inject, gigabytes after;
    redis_diskio is bursty (IQR 0) and spikes in bytes."""
    rng = np.random.default_rng(0)
    t = np.arange(n)
    after = t >= inject
    redis = np.zeros(n)
    redis[::17] = 1e9
    return pd.DataFrame({
        "time": t,
        "cart_cpu": 1 + 0.1 * rng.standard_normal(n) + 3 * after,
        "cart_diskio": np.where(after, 5e9, 0.0),
        "redis_diskio": redis,
        "front_latency": 10 + rng.standard_normal(n),
    }), inject


def test_robust_score_does_not_score_in_raw_units():
    assert robust_score([0.0] * 50, [9e9] * 50, [0.0] * 50 + [9e9] * 50) == pytest.approx(100)
    bursty = [0.0] * 45 + [1e9] * 5  # IQR 0: version 1 would score 2e9 here
    assert robust_score(bursty, [2e9] * 50, bursty + [2e9] * 50) == pytest.approx(2e9 / np.std(bursty))
    assert robust_score([1, 2, 3, 4, 5], [9], [1, 2, 3, 4, 5, 9]) == pytest.approx((9 - 3) / 2)  # BARO's z-score
    assert robust_score([4] * 5, [4] * 5, [4] * 10) == 0.0
    assert robust_score([np.nan] * 5, [1.0], [np.nan] * 5 + [1.0]) == 0.0


def test_scorer_version_1_stays_the_default():
    assert SCORER_VERSION == "1"
    df, inject = _flat_before_df()
    assert robust_scorer(df, inject_time=inject)["ranks"] == robust_scorer(df, inject_time=inject, version="1")["ranks"]
    with pytest.raises(ValueError):
        robust_scorer(df, inject_time=inject, version="9")


def test_scorer_version_2_keeps_series_flat_before_the_inject_time():
    df, inject = _flat_before_df()
    v1 = robust_scorer(df, inject_time=inject)["ranks"]
    assert "cart_diskio" not in v1  # constant in the normal half, so dropped
    assert v1[0] == "redis_diskio"  # scored in bytes
    v2 = robust_scorer(df, inject_time=inject, version="2")
    assert set(v2["ranks"][:2]) == {"cart_diskio", "cart_cpu"}
    assert all(score < 1e3 for _, score in v2["scores"])


def test_scorer_dict_version_2():
    df, inject = _flat_before_df()
    metrics = {c: [[int(t), float(v)] for t, v in zip(df["time"], df[c])] for c in df.columns if c != "time"}
    metrics["const"] = [[t, 3.0] for t in range(100)]
    v1 = [m for m, _ in robust_scorer_dict(metrics, inject)]
    assert v1[0] in ("cart_diskio", "redis_diskio")  # raw units: bytes win
    v2 = robust_scorer_dict(metrics, inject, version="2")
    assert {m for m, _ in v2[:2]} == {"cart_diskio", "cart_cpu"}
    assert "const" not in [m for m, _ in v2]
    assert robust_scorer(metrics, inject_time=inject, version="2")["ranks"] == [m for m, _ in v2]
