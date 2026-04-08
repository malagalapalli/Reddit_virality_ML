import os, sys, traceback
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['SPARK_LOCAL_DIRS'] = r'C:\temp\spark'
os.makedirs(r'C:\temp\spark', exist_ok=True)

try:
    from pyspark.sql import SparkSession
    print("PySpark imported OK")
    spark = (SparkSession.builder
        .master("local[2]")
        .appName("Test")
        .config("spark.driver.memory", "1g")
        .config("spark.local.dir", r"C:\temp\spark")
        .config("spark.sql.warehouse.dir", r"C:\temp\spark\warehouse")
        .getOrCreate())
    print("Spark session created OK")
    df = spark.createDataFrame([(1, "hello"), (2, "world")], ["id", "text"])
    print(f"Test DataFrame: {df.count()} rows")
    spark.stop()
    print("SUCCESS - PySpark works!")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
