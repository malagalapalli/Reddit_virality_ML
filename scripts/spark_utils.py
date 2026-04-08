"""
spark_utils.py - Utility functions for Spark session management and configuration.

Provides centralized Spark session creation with optimized settings for
processing the ~18GB Reddit TLDR-17 dataset on a single multi-core machine.
"""

import yaml
import os
from pyspark.sql import SparkSession


def get_project_root():
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_name="spark_config.yaml"):
    """Load a YAML configuration file from the config/ directory."""
    config_path = os.path.join(get_project_root(), "config", config_name)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_spark_session(app_name=None, memory="8g", cores="*"):
    """
    Create and return a configured SparkSession.
    
    Parameters
    ----------
    app_name : str, optional
        Application name. Defaults to config value.
    memory : str
        Driver memory allocation (e.g., '8g').
    cores : str
        Number of cores ('*' = all available).
    
    Returns
    -------
    SparkSession
        Configured Spark session.
    """
    config = load_config("spark_config.yaml")
    spark_cfg = config["spark"]
    
    if app_name is None:
        app_name = spark_cfg["app_name"]
    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", spark_cfg["executor_memory"])
        .config("spark.driver.maxResultSize", spark_cfg["max_result_size"])
        .config("spark.sql.shuffle.partitions", spark_cfg["shuffle_partitions"])
        .config("spark.default.parallelism", spark_cfg["default_parallelism"])
        .config("spark.serializer", spark_cfg["serializer"])
        .config("spark.sql.adaptive.enabled", str(spark_cfg["adaptive_enabled"]).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled", 
                str(spark_cfg["coalesce_partitions"]).lower())
        .config("spark.memory.fraction", spark_cfg["memory_fraction"])
        .config("spark.memory.storageFraction", spark_cfg["memory_storage_fraction"])
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    
    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Spark session created: {app_name}")
    print(f"  Spark version: {spark.version}")
    print(f"  Driver memory: {memory}")
    print(f"  Parallelism: local[{cores}]")
    print(f"  Spark UI: http://localhost:4040")
    
    return spark


def get_data_path(key):
    """
    Get a data path from the configuration.
    
    Parameters
    ----------
    key : str
        Data path key (e.g., 'raw_csv', 'parquet_dir').
    
    Returns
    -------
    str
        Absolute path to the data location.
    """
    config = load_config("spark_config.yaml")
    relative_path = config["data"][key]
    return os.path.join(get_project_root(), relative_path)


def stop_spark(spark):
    """Safely stop a Spark session."""
    if spark is not None:
        spark.stop()
        print("Spark session stopped.")
