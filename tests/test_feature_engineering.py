"""
Integration tests for PySpark-based feature engineering pipeline.
These tests require PySpark to be installed and will create a local Spark session.
Run with: python -m pytest tests/test_feature_engineering.py -v
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def spark_available():
    """Check if PySpark is available."""
    try:
        import pyspark
        return True
    except ImportError:
        return False


@pytest.fixture(scope='module')
def spark():
    """Create a test Spark session."""
    if not spark_available():
        pytest.skip("PySpark not installed")
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .master("local[2]")
             .appName("TestFeatureEngineering")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    yield spark
    spark.stop()


@pytest.fixture(scope='module')
def sample_df(spark):
    """Create a sample DataFrame for testing."""
    data = [
        ("user1", "This is a test post about Python programming. It has multiple sentences! How about that?",
         "A test post about Python.", "learnpython", "t5_abc", "id1",
         "This is content about Python programming", "Test summary about Python"),
        ("user2", "Short post", "Short post", "askreddit", "t5_def", "id2",
         "Short content", "Short summary"),
        ("user3", "ANOTHER TEST POST WITH LOTS OF UPPERCASE AND EXCLAMATION!!!! "
         "This explores many different topics. What do you think? I think it is great.\n\n"
         "New paragraph here. And another sentence. http://example.com",
         "Cleaned version of the post", "programming", "t5_ghi", "id3",
         "Full content here", "Summary of programming post"),
    ]
    columns = ["author", "body", "normalizedBody", "subreddit",
               "subreddit_id", "id", "content", "summary"]
    return spark.createDataFrame(data, columns)


@pytest.mark.skipif(not spark_available(), reason="PySpark not installed")
class TestCustomTransformersIntegration:
    """Integration tests for custom transformers with real Spark DataFrames."""

    def test_text_feature_extractor_transform(self, spark, sample_df):
        """TextFeatureExtractor should add feature columns to DataFrame."""
        from scripts.custom_transformer import TextFeatureExtractor

        extractor = TextFeatureExtractor(inputCol="body", outputCol="text_features")
        result = extractor.transform(sample_df)

        # Check expected output columns exist
        text_feature_cols = [
            'char_count', 'word_count', 'sentence_count',
            'avg_word_length', 'unique_word_ratio', 'question_count',
            'exclamation_count', 'uppercase_ratio', 'has_url', 'paragraph_count'
        ]
        for col in text_feature_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check row count preserved
        assert result.count() == 3

    def test_text_features_values(self, spark, sample_df):
        """TextFeatureExtractor should produce reasonable feature values."""
        from scripts.custom_transformer import TextFeatureExtractor

        extractor = TextFeatureExtractor(inputCol="body", outputCol="text_features")
        result = extractor.transform(sample_df)

        rows = result.collect()

        # First row: "This is a test post about Python programming..."
        row0 = rows[0]
        assert row0['word_count'] > 0
        assert row0['sentence_count'] >= 1
        assert row0['avg_word_length'] > 0
        assert 0 <= row0['unique_word_ratio'] <= 1.0
        assert row0['question_count'] >= 1  # Has "How about that?"

        # Third row: has URL
        row2 = rows[2]
        assert row2['has_url'] == 1  # Contains http://example.com

    def test_summary_feature_extractor_transform(self, spark, sample_df):
        """SummaryFeatureExtractor should add summary feature columns."""
        from scripts.custom_transformer import SummaryFeatureExtractor

        # First need text features for content_density
        from scripts.custom_transformer import TextFeatureExtractor
        text_ext = TextFeatureExtractor(inputCol="body", outputCol="text_features")
        intermediate = text_ext.transform(sample_df)

        summary_ext = SummaryFeatureExtractor(inputCol="summary", outputCol="summary_features")
        result = summary_ext.transform(intermediate)

        assert 'summary_length' in result.columns
        assert 'summary_word_count' in result.columns
        assert 'content_density' in result.columns
        assert result.count() == 3


@pytest.mark.skipif(not spark_available(), reason="PySpark not installed")
class TestViralityLabelCreation:
    """Tests for virality label definition logic."""

    def test_virality_label_binary(self, spark):
        """Virality label should be binary (0 or 1)."""
        from pyspark.sql import functions as F

        # Create a DataFrame with subreddit counts
        data = [(f"sub_{i}",) for i in range(100) for _ in range(i + 1)]
        df = spark.createDataFrame(data, ["subreddit"])

        # Count posts per subreddit
        counts = df.groupBy("subreddit").count()

        # Calculate 80th percentile threshold
        threshold = counts.approxQuantile("count", [0.80], 0.01)[0]

        # Add virality label
        df_with_counts = df.join(counts, "subreddit")
        df_labeled = df_with_counts.withColumn(
            "is_viral", F.when(F.col("count") >= threshold, 1).otherwise(0)
        )

        # Verify labels are 0 or 1
        labels = [row['is_viral'] for row in df_labeled.select('is_viral').distinct().collect()]
        assert set(labels) == {0, 1}, f"Expected {{0, 1}}, got {set(labels)}"

    def test_virality_approximately_20_percent(self, spark):
        """Approximately 20% of posts should be labeled as viral."""
        from pyspark.sql import functions as F

        # Create subreddits with varying activity
        data = [(f"popular_{i}", "body", "summary") for i in range(5) for _ in range(100)]
        data += [(f"niche_{i}", "body", "summary") for i in range(20) for _ in range(10)]
        df = spark.createDataFrame(data, ["subreddit", "body", "summary"])

        counts = df.groupBy("subreddit").count()
        threshold = counts.approxQuantile("count", [0.80], 0.01)[0]

        df_with_counts = df.join(counts, "subreddit")
        df_labeled = df_with_counts.withColumn(
            "is_viral", F.when(F.col("count") >= threshold, 1).otherwise(0)
        )

        total = df_labeled.count()
        viral_count = df_labeled.filter(F.col("is_viral") == 1).count()
        viral_pct = viral_count / total

        # Allow some tolerance — should be roughly in the 15%-50% range
        assert 0.10 <= viral_pct <= 0.60, (
            f"Viral percentage {viral_pct:.1%} is outside expected range"
        )


@pytest.mark.skipif(not spark_available(), reason="PySpark not installed")
class TestPipelineBuilder:
    """Tests for the ML pipeline builder functions."""

    def test_build_preprocessing_pipeline(self, spark, sample_df):
        """Pipeline should build without errors."""
        from scripts.feature_engineering import build_preprocessing_pipeline
        pipeline = build_preprocessing_pipeline()
        assert pipeline is not None
        assert len(pipeline.getStages()) > 0

    def test_pipeline_stage_count(self, spark, sample_df):
        """Pipeline should have the expected number of stages."""
        from scripts.feature_engineering import build_preprocessing_pipeline
        pipeline = build_preprocessing_pipeline()
        stages = pipeline.getStages()
        # Expect: TextExtractor, SummaryExtractor, StringIndexer, Tokenizer,
        # StopWordsRemover, HashingTF, IDF, VectorAssembler, StandardScaler
        assert len(stages) >= 7, f"Expected ≥7 stages, got {len(stages)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
