import pyspark.sql.functions as F

from src.etl_pipeline.spark_config import get_spark_session


def silver_layer(spark, source_path, target_path):
    """
    Raw data processing (removes missing values, creates time features)
    Time features are later used for partitioning
    TODO: docs what?
    See docs/? for processing decisions
    """
    spark = get_spark_session()
    df = spark.read.parquet(source_path)

    df_clean = df.fillna({"CustomerID": "missing", "Description": "unknown"})
    df_clean = df_clean.drop_duplicates()
    df_clean = (
        df_clean.withColumn("Year", F.year("InvoiceDate"))
        .withColumn("Month", F.month("InvoiceDate"))
        .withColumn("Hour", F.hour("InvoiceDate"))
        .withColumn("Weekday", F.dayofweek("InvoiceDate"))
        .withColumn("Month_Year", F.date_trunc("month", "InvoiceDate"))
    )
    df_clean = df_clean.filter(F.col("UnitPrice") > 0)

    # should be append in prod
    df_clean.write.mode("overwrite").parquet(target_path)

    print(f"Silver: wrote {df_clean.count()} records into {target_path}")
