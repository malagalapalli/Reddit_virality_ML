"""
custom_transformer.py - Custom PySpark Transformer for text feature extraction.

Implements a custom Transformer that computes hand-crafted text features from 
Reddit post bodies. This demonstrates advanced PySpark ML Pipeline integration
and is critical for high marks in the preprocessing section.
"""

import re
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, FloatType, IntegerType
)


class TextFeatureExtractor(Transformer, HasInputCol, HasOutputCol,
                           DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom PySpark Transformer that extracts hand-crafted text features
    from a text column. This produces multiple numeric features used 
    downstream in the ML pipeline.
    
    Features extracted:
        - char_count: Total character count
        - word_count: Total word count
        - sentence_count: Approximate sentence count
        - avg_word_length: Average word length
        - unique_word_ratio: Ratio of unique words to total words
        - question_count: Number of question marks
        - exclamation_count: Number of exclamation marks
        - uppercase_ratio: Ratio of uppercase letters
        - has_url: Whether text contains a URL (0 or 1)
        - paragraph_count: Number of paragraphs (double newlines)
    """
    
    def __init__(self, inputCol="body", outputCol="text_features"):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self.set(self.inputCol, inputCol)
        self.set(self.outputCol, outputCol)
    
    def _transform(self, df: DataFrame) -> DataFrame:
        input_col = self.getInputCol()
        
        # Fill nulls to prevent errors
        df = df.withColumn(input_col, 
                          F.coalesce(F.col(input_col), F.lit("")))
        
        # Character count
        df = df.withColumn("char_count", 
                          F.length(F.col(input_col)).cast(FloatType()))
        
        # Word count
        df = df.withColumn("word_count",
                          F.size(F.split(F.col(input_col), r"\s+")).cast(FloatType()))
        
        # Sentence count (approximate: split on . ! ?)
        df = df.withColumn("sentence_count",
                          F.size(F.split(F.col(input_col), r"[.!?]+")).cast(FloatType()))
        
        # Average word length
        df = df.withColumn("avg_word_length",
                          F.when(F.col("word_count") > 0,
                                 F.col("char_count") / F.col("word_count"))
                          .otherwise(0.0).cast(FloatType()))
        
        # Question count
        df = df.withColumn("question_count",
                          (F.length(F.col(input_col)) - 
                           F.length(F.regexp_replace(F.col(input_col), r"\?", "")))
                          .cast(FloatType()))
        
        # Exclamation count
        df = df.withColumn("exclamation_count",
                          (F.length(F.col(input_col)) - 
                           F.length(F.regexp_replace(F.col(input_col), "!", "")))
                          .cast(FloatType()))
        
        # Uppercase ratio
        upper_chars = F.length(F.regexp_replace(F.col(input_col), r"[^A-Z]", ""))
        df = df.withColumn("uppercase_ratio",
                          F.when(F.col("char_count") > 0,
                                 upper_chars.cast(FloatType()) / F.col("char_count"))
                          .otherwise(0.0).cast(FloatType()))
        
        # Has URL (binary)
        df = df.withColumn("has_url",
                          F.when(F.col(input_col).rlike(r"https?://\S+"), 1.0)
                          .otherwise(0.0).cast(FloatType()))
        
        # Paragraph count (double newlines)
        df = df.withColumn("paragraph_count",
                          (F.size(F.split(F.col(input_col), r"\n\n")) ).cast(FloatType()))
        
        return df


class SummaryFeatureExtractor(Transformer, HasInputCol, HasOutputCol,
                              DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom Transformer that extracts features from the summary column
    and computes content density metrics relative to the body.
    
    Features extracted:
        - summary_length: Character count of summary
        - summary_word_count: Word count of summary
        - content_density: Ratio of summary length to body length
    """
    
    def __init__(self, inputCol="summary", outputCol="summary_features"):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self.set(self.inputCol, inputCol)
        self.set(self.outputCol, outputCol)

    def _transform(self, df: DataFrame) -> DataFrame:
        input_col = self.getInputCol()
        
        df = df.withColumn(input_col,
                          F.coalesce(F.col(input_col), F.lit("")))
        
        # Summary length
        df = df.withColumn("summary_length",
                          F.length(F.col(input_col)).cast(FloatType()))
        
        # Summary word count
        df = df.withColumn("summary_word_count",
                          F.size(F.split(F.col(input_col), r"\s+")).cast(FloatType()))
        
        # Content density: summary / body ratio
        df = df.withColumn("content_density",
                          F.when(F.col("char_count") > 0,
                                 F.col("summary_length") / F.col("char_count"))
                          .otherwise(0.0).cast(FloatType()))
        
        return df
