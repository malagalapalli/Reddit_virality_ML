"""
Tests for custom PySpark transformers and feature engineering pipeline.
Run with: python -m pytest tests/ -v
"""
import os
import sys
import pytest

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


class TestProjectStructure:
    """Verify project directory structure matches coursework requirements."""

    def test_required_directories_exist(self):
        """All required directories should exist."""
        required_dirs = ['config', 'data', 'notebooks', 'scripts', 'tableau', 'tests']
        for d in required_dirs:
            path = os.path.join(PROJECT_ROOT, d)
            assert os.path.isdir(path), f"Missing required directory: {d}"

    def test_config_files_exist(self):
        """Config files should exist."""
        configs = ['config/spark_config.yaml', 'config/project_config.yaml']
        for c in configs:
            path = os.path.join(PROJECT_ROOT, c)
            assert os.path.isfile(path), f"Missing config file: {c}"

    def test_script_files_exist(self):
        """Utility scripts should exist."""
        scripts = [
            'scripts/__init__.py',
            'scripts/spark_utils.py',
            'scripts/custom_transformer.py',
            'scripts/feature_engineering.py',
        ]
        for s in scripts:
            path = os.path.join(PROJECT_ROOT, s)
            assert os.path.isfile(path), f"Missing script: {s}"

    def test_notebooks_exist(self):
        """All required notebooks should exist."""
        notebook_dir = os.path.join(PROJECT_ROOT, 'notebooks')
        notebooks = os.listdir(notebook_dir)
        ipynb_files = [f for f in notebooks if f.endswith('.ipynb')]
        assert len(ipynb_files) >= 8, (
            f"Expected at least 8 notebooks, found {len(ipynb_files)}: {ipynb_files}"
        )

    def test_raw_data_directory_exists(self):
        """Raw data directory should exist."""
        path = os.path.join(PROJECT_ROOT, 'data', 'raw')
        assert os.path.isdir(path), "Missing data/raw/ directory"


class TestConfigLoading:
    """Tests for configuration loading utilities."""

    def test_load_spark_config(self):
        """Spark config should load and contain expected keys."""
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'spark_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'spark' in config, "Config missing 'spark' section"
        assert 'data_paths' in config, "Config missing 'data_paths' section"
        assert 'processing' in config, "Config missing 'processing' section"

    def test_load_project_config(self):
        """Project config should load and contain expected keys."""
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'project_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'project' in config, "Config missing 'project' section"
        assert 'features' in config, "Config missing 'features' section"
        assert 'models' in config, "Config missing 'models' section"

    def test_spark_config_values(self):
        """Spark config should have reasonable values."""
        import yaml
        config_path = os.path.join(PROJECT_ROOT, 'config', 'spark_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        spark = config['spark']
        assert 'driver_memory' in spark
        assert 'shuffle_partitions' in spark
        assert int(spark['shuffle_partitions']) > 0

        processing = config['processing']
        assert 0 < float(processing['sample_fraction']) <= 1.0
        assert 0 < float(processing['test_size']) < 1.0
        assert int(processing['tfidf_features']) > 0


class TestSparkUtils:
    """Tests for spark_utils.py functions."""

    def test_get_project_root(self):
        """get_project_root should return a valid directory."""
        from scripts.spark_utils import get_project_root
        root = get_project_root()
        assert os.path.isdir(root), f"Project root not a directory: {root}"

    def test_load_config(self):
        """load_config should return a dictionary."""
        from scripts.spark_utils import load_config
        config = load_config()
        assert isinstance(config, dict)
        assert 'spark' in config

    def test_get_data_path(self):
        """get_data_path should return string paths."""
        from scripts.spark_utils import get_data_path
        path = get_data_path('raw_csv')
        assert isinstance(path, str)
        assert 'reddit_posts.csv' in path


class TestCustomTransformer:
    """Tests for custom PySpark transformer imports and instantiation."""

    def test_text_feature_extractor_import(self):
        """TextFeatureExtractor should be importable."""
        from scripts.custom_transformer import TextFeatureExtractor
        assert TextFeatureExtractor is not None

    def test_summary_feature_extractor_import(self):
        """SummaryFeatureExtractor should be importable."""
        from scripts.custom_transformer import SummaryFeatureExtractor
        assert SummaryFeatureExtractor is not None

    def test_text_feature_extractor_init(self):
        """TextFeatureExtractor should accept inputCol and outputCol params."""
        from scripts.custom_transformer import TextFeatureExtractor
        extractor = TextFeatureExtractor(inputCol="body", outputCol="text_features")
        assert extractor.getInputCol() == "body"
        assert extractor.getOutputCol() == "text_features"

    def test_summary_feature_extractor_init(self):
        """SummaryFeatureExtractor should accept inputCol and outputCol params."""
        from scripts.custom_transformer import SummaryFeatureExtractor
        extractor = SummaryFeatureExtractor(inputCol="summary", outputCol="summary_features")
        assert extractor.getInputCol() == "summary"
        assert extractor.getOutputCol() == "summary_features"


class TestFeatureEngineering:
    """Tests for feature engineering pipeline builder."""

    def test_build_preprocessing_pipeline_import(self):
        """build_preprocessing_pipeline should be importable."""
        from scripts.feature_engineering import build_preprocessing_pipeline
        assert callable(build_preprocessing_pipeline)

    def test_build_simple_pipeline_import(self):
        """build_simple_pipeline should be importable."""
        from scripts.feature_engineering import build_simple_pipeline
        assert callable(build_simple_pipeline)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
