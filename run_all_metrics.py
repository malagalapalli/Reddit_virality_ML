"""
=============================================================================
FAST ALL-IN-ONE: Complete ML Pipeline with All Metrics
Runs on a sample for speed, produces all required output metrics
=============================================================================
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

# Fix Hadoop/Spark on Windows - MUST be before pyspark import
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['SPARK_LOCAL_DIRS'] = r'C:\temp\spark'
os.makedirs(r'C:\temp\spark', exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from collections import OrderedDict

print("=" * 70)
print("PREDICTING SOCIAL MEDIA CONTENT VIRALITY")
print("Distributed Machine Learning on Reddit Big Data")
print("=" * 70)

# ============================================================================
# STEP 1: Load Data Sample with PySpark
# ============================================================================
print("\n[STEP 1] Starting PySpark and loading data...")
t_total_start = time.time()

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

NUM_CORES = os.cpu_count() or 4
print(f"  CPU cores available: {NUM_CORES}")

# Use short paths to avoid Windows space-in-path issues
SPARK_TEMP = r"C:\temp\spark"
SPARK_WAREHOUSE = r"C:\temp\spark\warehouse"
os.makedirs(SPARK_TEMP, exist_ok=True)
os.makedirs(SPARK_WAREHOUSE, exist_ok=True)

spark = (SparkSession.builder
    .master(f"local[{NUM_CORES}]")
    .appName("RedditViralityPipeline")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", str(NUM_CORES * 2))
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.local.dir", SPARK_TEMP)
    .config("spark.sql.warehouse.dir", SPARK_WAREHOUSE)
    .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={SPARK_TEMP}")
    .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print(f"  Spark session created: local[{NUM_CORES}]")

csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "reddit_posts.csv")
schema = StructType([
    StructField("author", StringType(), True),
    StructField("body", StringType(), True),
    StructField("normalizedBody", StringType(), True),
    StructField("subreddit", StringType(), True),
    StructField("subreddit_id", StringType(), True),
    StructField("id", StringType(), True),
    StructField("content", StringType(), True),
    StructField("summary", StringType(), True),
])

# Load 10% sample for speed
t0 = time.time()
raw_df = spark.read.csv(csv_path, header=True, schema=schema)
sample_df = raw_df.sample(fraction=0.05, seed=42).cache()
sample_count = sample_df.count()
t_load = time.time() - t0
print(f"  Loaded {sample_count:,} rows (5% sample) in {t_load:.1f}s")

# ============================================================================
# STEP 2: Create Virality Label
# ============================================================================
print("\n[STEP 2] Creating virality labels...")
sub_counts = sample_df.groupBy("subreddit").count()
threshold = sub_counts.approxQuantile("count", [0.80], 0.01)[0]
print(f"  Virality threshold (80th percentile): {threshold:.0f} posts per subreddit")

df = sample_df.join(sub_counts.withColumnRenamed("count", "sub_count"), "subreddit")
df = df.withColumn("is_viral", F.when(F.col("sub_count") >= threshold, 1).otherwise(0))
df = df.filter(F.col("body").isNotNull() & (F.length(F.col("body")) > 10))
df = df.filter(F.col("summary").isNotNull() & (F.length(F.col("summary")) > 5))

viral_count = df.filter(F.col("is_viral") == 1).count()
non_viral_count = df.filter(F.col("is_viral") == 0).count()
total = viral_count + non_viral_count
print(f"  Viral: {viral_count:,} ({100*viral_count/total:.1f}%)")
print(f"  Non-viral: {non_viral_count:,} ({100*non_viral_count/total:.1f}%)")

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================
print("\n[STEP 3] Feature engineering...")
t0 = time.time()

# Text features via UDFs
from pyspark.sql.functions import udf, col, length, size, split, regexp_replace
from pyspark.ml.feature import (StringIndexer, Tokenizer, StopWordsRemover,
                                 HashingTF, IDF, VectorAssembler)
from pyspark.ml.linalg import Vectors, VectorUDT

df = df.withColumn("char_count", length(col("body")).cast("double"))
df = df.withColumn("word_count", size(split(col("body"), "\\s+")).cast("double"))
df = df.withColumn("sentence_count",
    (size(split(col("body"), "[.!?]+")) - 1).cast("double"))
df = df.withColumn("sentence_count",
    F.when(col("sentence_count") < 1, F.lit(1.0)).otherwise(col("sentence_count")))
df = df.withColumn("avg_word_length",
    (col("char_count") / F.when(col("word_count") > 0, col("word_count")).otherwise(1)).cast("double"))

# Unique word ratio
@udf("double")
def unique_ratio(text):
    if not text: return 0.0
    words = text.lower().split()
    return len(set(words)) / max(len(words), 1)

df = df.withColumn("unique_word_ratio", unique_ratio(col("body")))

# Other features
df = df.withColumn("question_count",
    (size(split(col("body"), "\\?")) - 1).cast("double"))
df = df.withColumn("exclamation_count",
    (size(split(col("body"), "!")) - 1).cast("double"))

@udf("double")
def upper_ratio(text):
    if not text or len(text) == 0: return 0.0
    return sum(1 for c in text if c.isupper()) / len(text)

df = df.withColumn("uppercase_ratio", upper_ratio(col("body")))
df = df.withColumn("has_url",
    F.when(col("body").contains("http"), 1.0).otherwise(0.0))
df = df.withColumn("paragraph_count",
    (size(split(col("body"), "\n\n")) ).cast("double"))
df = df.withColumn("summary_length", length(col("summary")).cast("double"))
df = df.withColumn("summary_word_count",
    size(split(col("summary"), "\\s+")).cast("double"))
df = df.withColumn("content_density",
    (col("summary_length") / F.when(col("char_count") > 0, col("char_count")).otherwise(1)).cast("double"))

# Fill nulls
numeric_cols = ["char_count", "word_count", "sentence_count", "avg_word_length",
                "unique_word_ratio", "question_count", "exclamation_count",
                "uppercase_ratio", "has_url", "paragraph_count",
                "summary_length", "summary_word_count", "content_density"]
for c in numeric_cols:
    df = df.withColumn(c, F.when(col(c).isNull(), 0.0).otherwise(col(c)))

# Subreddit indexing
indexer = StringIndexer(inputCol="subreddit", outputCol="subreddit_index",
                        handleInvalid="keep")
df = indexer.fit(df).transform(df)

# TF-IDF on body text
tokenizer = Tokenizer(inputCol="body", outputCol="words")
df = tokenizer.transform(df)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = remover.transform(df)
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features",
                       numFeatures=2000)
df = hashingTF.transform(df)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df)
df = idf_model.transform(df)

# Assemble all features
assembler = VectorAssembler(
    inputCols=numeric_cols + ["subreddit_index", "tfidf_features"],
    outputCol="features",
    handleInvalid="skip"
)
df = assembler.transform(df)
df = df.select("features", col("is_viral").cast("double").alias("label"))
df = df.filter(col("features").isNotNull() & col("label").isNotNull()).cache()

feature_count = df.first()["features"].size
t_feat = time.time() - t0
print(f"  Features: {feature_count} dimensions ({len(numeric_cols)} numeric + 1 categorical + 2000 TF-IDF)")
print(f"  Rows after cleaning: {df.count():,}")
print(f"  Feature engineering time: {t_feat:.1f}s")

# Train/Test Split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
train_count = train_df.count()
test_count = test_df.count()
print(f"  Train: {train_count:,} | Test: {test_count:,}")

# ============================================================================
# STEP 4: PySpark MLlib Models
# ============================================================================
print("\n[STEP 4] Training PySpark MLlib models...")
print(f"  Using {NUM_CORES} CPU cores")

from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
                                        DecisionTreeClassifier, GBTClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

bin_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                          metricName="areaUnderROC")
mc_eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                 metricName="accuracy")
mc_eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                metricName="f1")
mc_eval_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="weightedPrecision")
mc_eval_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                 metricName="weightedRecall")

spark_models = OrderedDict({
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label",
                                               maxIter=50, regParam=0.01),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label",
                                             numTrees=50, maxDepth=10, seed=42),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                             maxDepth=15, seed=42),
    "GBT": GBTClassifier(featuresCol="features", labelCol="label",
                          maxIter=50, maxDepth=8, stepSize=0.1, seed=42),
})

spark_results = []

for name, model in spark_models.items():
    print(f"\n  Training {name}...")
    t0 = time.time()
    fitted = model.fit(train_df)
    t_train = time.time() - t0

    preds = fitted.transform(test_df)

    accuracy = mc_eval_acc.evaluate(preds)
    precision = mc_eval_prec.evaluate(preds)
    recall = mc_eval_rec.evaluate(preds)
    f1 = mc_eval_f1.evaluate(preds)
    auc_roc = bin_eval.evaluate(preds)

    # Confusion matrix
    pred_and_labels = preds.select("prediction", "label").rdd.map(
        lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(pred_and_labels)
    cm = metrics.confusionMatrix().toArray()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    result = {
        "Model": name, "Framework": "PySpark",
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "F1-Score": f1, "AUC-ROC": auc_roc,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Training Time (s)": round(t_train, 2),
        "Cores Used": NUM_CORES,
    }
    spark_results.append(result)
    print(f"    Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC-ROC: {auc_roc:.4f} | Time: {t_train:.1f}s")
    print(f"    Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# ============================================================================
# STEP 5: Hyperparameter Tuning (CrossValidator)
# ============================================================================
print("\n[STEP 5] Hyperparameter tuning with CrossValidator (3-fold)...")
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Tune top 3 models
tuning_results = []

# LR tuning
print("  Tuning Logistic Regression...")
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_grid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.001, 0.01, 0.1])
    .addGrid(lr.elasticNetParam, [0.0, 0.5])
    .addGrid(lr.maxIter, [50, 100])
    .build())
lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_grid,
                        evaluator=bin_eval, numFolds=3, seed=42)
t0 = time.time()
lr_cv_model = lr_cv.fit(train_df)
t_lr_tune = time.time() - t0
best_lr = lr_cv_model.bestModel
tuning_results.append({
    "Model": "Logistic Regression",
    "Best regParam": round(best_lr._java_obj.getRegParam(), 4),
    "Best elasticNetParam": round(best_lr._java_obj.getElasticNetParam(), 4),
    "Best maxIter": best_lr._java_obj.getMaxIter(),
    "Best CV AUC-ROC": round(max(lr_cv_model.avgMetrics), 4),
    "Tuning Time (s)": round(t_lr_tune, 2)
})
print(f"    Best: regParam={best_lr._java_obj.getRegParam():.4f}, "
      f"elasticNet={best_lr._java_obj.getElasticNetParam():.1f}, "
      f"maxIter={best_lr._java_obj.getMaxIter()} | "
      f"AUC={max(lr_cv_model.avgMetrics):.4f} | Time: {t_lr_tune:.1f}s")

# RF tuning
print("  Tuning Random Forest...")
rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
rf_grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [30, 50])
    .addGrid(rf.maxDepth, [8, 12])
    .build())
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_grid,
                        evaluator=bin_eval, numFolds=3, seed=42)
t0 = time.time()
rf_cv_model = rf_cv.fit(train_df)
t_rf_tune = time.time() - t0
best_rf = rf_cv_model.bestModel
tuning_results.append({
    "Model": "Random Forest",
    "Best numTrees": best_rf._java_obj.getNumTrees(),
    "Best maxDepth": best_rf._java_obj.getMaxDepth(),
    "Best CV AUC-ROC": round(max(rf_cv_model.avgMetrics), 4),
    "Tuning Time (s)": round(t_rf_tune, 2)
})
print(f"    Best: numTrees={best_rf._java_obj.getNumTrees()}, "
      f"maxDepth={best_rf._java_obj.getMaxDepth()} | "
      f"AUC={max(rf_cv_model.avgMetrics):.4f} | Time: {t_rf_tune:.1f}s")

# GBT tuning
print("  Tuning GBT...")
gbt = GBTClassifier(featuresCol="features", labelCol="label", seed=42)
gbt_grid = (ParamGridBuilder()
    .addGrid(gbt.maxIter, [30, 50])
    .addGrid(gbt.maxDepth, [5, 8])
    .addGrid(gbt.stepSize, [0.05, 0.1])
    .build())
gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_grid,
                         evaluator=bin_eval, numFolds=3, seed=42)
t0 = time.time()
gbt_cv_model = gbt_cv.fit(train_df)
t_gbt_tune = time.time() - t0
best_gbt = gbt_cv_model.bestModel
tuning_results.append({
    "Model": "GBT",
    "Best maxIter": best_gbt._java_obj.getMaxIter(),
    "Best maxDepth": best_gbt._java_obj.getMaxDepth(),
    "Best stepSize": round(best_gbt._java_obj.getStepSize(), 4),
    "Best CV AUC-ROC": round(max(gbt_cv_model.avgMetrics), 4),
    "Tuning Time (s)": round(t_gbt_tune, 2)
})
print(f"    Best: maxIter={best_gbt._java_obj.getMaxIter()}, "
      f"maxDepth={best_gbt._java_obj.getMaxDepth()}, "
      f"stepSize={best_gbt._java_obj.getStepSize():.2f} | "
      f"AUC={max(gbt_cv_model.avgMetrics):.4f} | Time: {t_gbt_tune:.1f}s")

# ============================================================================
# STEP 6: Evaluate Tuned Models
# ============================================================================
print("\n[STEP 6] Evaluating tuned models on test set...")
tuned_models = {
    "Logistic Regression (Tuned)": lr_cv_model.bestModel,
    "Random Forest (Tuned)": rf_cv_model.bestModel,
    "GBT (Tuned)": gbt_cv_model.bestModel,
}

tuned_results = []
for name, fitted in tuned_models.items():
    preds = fitted.transform(test_df)
    accuracy = mc_eval_acc.evaluate(preds)
    precision = mc_eval_prec.evaluate(preds)
    recall = mc_eval_rec.evaluate(preds)
    f1 = mc_eval_f1.evaluate(preds)
    auc_roc = bin_eval.evaluate(preds)

    pred_and_labels = preds.select("prediction", "label").rdd.map(
        lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(pred_and_labels)
    cm = metrics.confusionMatrix().toArray()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    tuned_results.append({
        "Model": name, "Framework": "PySpark (Tuned)",
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "F1-Score": f1, "AUC-ROC": auc_roc,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    })
    print(f"  {name}: Acc={accuracy:.4f} | F1={f1:.4f} | AUC={auc_roc:.4f}")
    print(f"    CM: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# ============================================================================
# STEP 7: Scikit-learn Comparison
# ============================================================================
print("\n[STEP 7] Scikit-learn comparison (single machine)...")
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.ensemble import GradientBoostingClassifier as SKGBT
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# Convert to pandas (use sample for sklearn)
pdf = df.sample(fraction=min(100000 / df.count(), 1.0), seed=42).toPandas()

from scipy.sparse import csr_matrix
X_list = []
for row in pdf["features"]:
    X_list.append(row.toArray())
X = np.array(X_list)
y = pdf["label"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Sklearn dataset: {len(X_train)} train, {len(X_test)} test")

sk_models = OrderedDict({
    "Logistic Regression": SKLogReg(max_iter=100, C=10, random_state=42, n_jobs=-1),
    "Random Forest": SKRandomForest(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "Decision Tree": SKDecisionTree(max_depth=15, random_state=42),
    "GBT": SKGBT(n_estimators=50, max_depth=8, learning_rate=0.1, random_state=42),
})

sklearn_results = []
for name, model in sk_models.items():
    print(f"  Training {name} (sklearn)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_val = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn_sk, fp_sk, fn_sk, tp_sk = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    sklearn_results.append({
        "Model": name, "Framework": "Scikit-learn",
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1_val, "AUC-ROC": auc,
        "TP": int(tp_sk), "TN": int(tn_sk), "FP": int(fp_sk), "FN": int(fn_sk),
        "Training Time (s)": round(t_train, 2),
        "Cores Used": 1,
    })
    print(f"    Acc={acc:.4f} | F1={f1_val:.4f} | AUC={auc:.4f} | Time: {t_train:.1f}s")

# ============================================================================
# FINAL OUTPUT: All Results
# ============================================================================
t_total = time.time() - t_total_start

print("\n" + "=" * 90)
print("COMPLETE RESULTS SUMMARY")
print("=" * 90)

print("\n" + "-" * 90)
print("TABLE 1: PySpark MLlib Model Performance (Base Models)")
print("-" * 90)
header = f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Time(s)':>8} {'Cores':>5}"
print(header)
print("-" * 90)
for r in spark_results:
    print(f"{r['Model']:<25} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f} {r['Recall']:>7.4f} "
          f"{r['F1-Score']:>7.4f} {r['AUC-ROC']:>7.4f} {r['TP']:>6d} {r['TN']:>6d} "
          f"{r['FP']:>6d} {r['FN']:>6d} {r['Training Time (s)']:>8.2f} {r['Cores Used']:>5d}")

print("\n" + "-" * 90)
print("TABLE 2: Hyperparameter Tuning Results (3-Fold CrossValidator)")
print("-" * 90)
for t in tuning_results:
    print(f"\n  {t['Model']}:")
    for k, v in t.items():
        if k != "Model":
            print(f"    {k}: {v}")

print("\n" + "-" * 90)
print("TABLE 3: Tuned Model Performance")
print("-" * 90)
header = f"{'Model':<35} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}"
print(header)
print("-" * 90)
for r in tuned_results:
    print(f"{r['Model']:<35} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f} {r['Recall']:>7.4f} "
          f"{r['F1-Score']:>7.4f} {r['AUC-ROC']:>7.4f} {r['TP']:>6d} {r['TN']:>6d} "
          f"{r['FP']:>6d} {r['FN']:>6d}")

print("\n" + "-" * 90)
print("TABLE 4: Scikit-learn Comparison (Single Machine)")
print("-" * 90)
header = f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'Time(s)':>8} {'Cores':>5}"
print(header)
print("-" * 90)
for r in sklearn_results:
    print(f"{r['Model']:<25} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f} {r['Recall']:>7.4f} "
          f"{r['F1-Score']:>7.4f} {r['AUC-ROC']:>7.4f} {r['TP']:>6d} {r['TN']:>6d} "
          f"{r['FP']:>6d} {r['FN']:>6d} {r['Training Time (s)']:>8.2f} {r['Cores Used']:>5d}")

print("\n" + "-" * 90)
print("TABLE 5: PySpark vs Scikit-learn Comparison")
print("-" * 90)
header = f"{'Model':<25} {'Spark AUC':>10} {'SK AUC':>10} {'Spark Time':>12} {'SK Time':>12} {'Spark Cores':>12}"
print(header)
print("-" * 90)
for sp, sk in zip(spark_results, sklearn_results):
    print(f"{sp['Model']:<25} {sp['AUC-ROC']:>10.4f} {sk['AUC-ROC']:>10.4f} "
          f"{sp['Training Time (s)']:>11.2f}s {sk['Training Time (s)']:>11.2f}s {sp['Cores Used']:>12d}")

print(f"\n{'=' * 90}")
print(f"TOTAL PIPELINE TIME: {t_total:.1f}s ({t_total/60:.1f} minutes)")
print(f"Dataset: {sample_count:,} rows | Features: {feature_count}")
print(f"Cores: {NUM_CORES} | Framework: PySpark {spark.version}")
print(f"{'=' * 90}")

# ============================================================================
# Save results to CSV for Tableau
# ============================================================================
tableau_dir = os.path.join(PROJECT_ROOT, "tableau", "data")
os.makedirs(tableau_dir, exist_ok=True)

all_results = spark_results + sklearn_results
pd.DataFrame(all_results).to_csv(os.path.join(tableau_dir, "all_model_results.csv"), index=False)
pd.DataFrame(tuned_results).to_csv(os.path.join(tableau_dir, "tuned_model_results.csv"), index=False)
pd.DataFrame(tuning_results).to_csv(os.path.join(tableau_dir, "tuning_hyperparameters.csv"), index=False)
print("\nResults saved to tableau/data/")

spark.stop()
print("\nDone! Spark session stopped.")
