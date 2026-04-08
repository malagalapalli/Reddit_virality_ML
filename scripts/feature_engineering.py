"""
feature_engineering.py - Feature engineering pipeline utilities.

Provides functions for building complete PySpark ML preprocessing pipelines
including text feature extraction, TF-IDF vectorization, and feature assembly.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF,
    StringIndexer, VectorAssembler, StandardScaler,
    SQLTransformer
)
from scripts.custom_transformer import TextFeatureExtractor, SummaryFeatureExtractor


# Numeric features produced by custom transformers
NUMERIC_FEATURES = [
    "char_count", "word_count", "sentence_count", "avg_word_length",
    "unique_word_ratio", "question_count", "exclamation_count",
    "uppercase_ratio", "has_url", "paragraph_count",
    "summary_length", "summary_word_count", "content_density"
]


def build_preprocessing_pipeline(
    text_col="normalizedBody",
    summary_col="summary",
    subreddit_col="subreddit",
    label_col="is_viral",
    max_tfidf_features=5000
):
    """
    Build a complete PySpark ML preprocessing pipeline.
    
    Pipeline stages:
    1. Custom TextFeatureExtractor (body features)
    2. Custom SummaryFeatureExtractor (summary features)
    3. Unique word ratio via SQL Transformer
    4. Subreddit StringIndexer
    5. Tokenizer → StopWordsRemover → HashingTF → IDF (TF-IDF)
    6. VectorAssembler (combine all features)
    7. StandardScaler
    
    Parameters
    ----------
    text_col : str
        Column name for main text body.
    summary_col : str
        Column name for summary text.
    subreddit_col : str
        Column for subreddit name.
    label_col : str
        Target column name.
    max_tfidf_features : int
        Number of TF-IDF hash features.
    
    Returns
    -------
    Pipeline
        Complete preprocessing pipeline.
    """
    
    # Stage 1: Custom text feature extraction
    text_extractor = TextFeatureExtractor(inputCol=text_col, outputCol="text_feats")
    
    # Stage 2: Custom summary feature extraction  
    summary_extractor = SummaryFeatureExtractor(inputCol=summary_col, outputCol="summ_feats")
    
    # Stage 3: Compute unique word ratio via SQL
    unique_word_sql = SQLTransformer(
        statement="""
        SELECT *, 
            CASE WHEN word_count > 0 
                 THEN CAST(size(array_distinct(split(normalizedBody, ' '))) AS FLOAT) / word_count 
                 ELSE 0.0 
            END AS unique_word_ratio
        FROM __THIS__
        """
    )
    
    # Stage 4: Encode subreddit as numeric index
    subreddit_indexer = StringIndexer(
        inputCol=subreddit_col, 
        outputCol="subreddit_index",
        handleInvalid="keep"
    )
    
    # Stage 5: TF-IDF pipeline (Tokenizer → StopWords → HashTF → IDF)
    tokenizer = Tokenizer(inputCol=text_col, outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(
        inputCol="filtered_words", 
        outputCol="raw_tfidf", 
        numFeatures=max_tfidf_features
    )
    idf = IDF(inputCol="raw_tfidf", outputCol="tfidf_features", minDocFreq=5)
    
    # Stage 6: Assemble all features into a single vector
    assembler_inputs = NUMERIC_FEATURES + ["subreddit_index", "tfidf_features"]
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="raw_features",
        handleInvalid="skip"
    )
    
    # Stage 7: Scale features
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False  # Sparse vectors don't support withMean=True
    )
    
    # Assemble pipeline
    pipeline = Pipeline(stages=[
        text_extractor,
        summary_extractor,
        unique_word_sql,
        subreddit_indexer,
        tokenizer,
        stopwords_remover,
        hashing_tf,
        idf,
        assembler,
        scaler
    ])
    
    return pipeline


def build_simple_pipeline(text_col="normalizedBody", max_tfidf_features=5000):
    """
    Build a simpler pipeline without custom transformers (for testing).
    Uses only standard PySpark ML stages.
    """
    tokenizer = Tokenizer(inputCol=text_col, outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_tfidf",
        numFeatures=max_tfidf_features
    )
    idf = IDF(inputCol="raw_tfidf", outputCol="features", minDocFreq=5)
    
    return Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf])
