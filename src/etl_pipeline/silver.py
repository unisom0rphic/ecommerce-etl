import pyspark.sql
from pyspark.sql.functions import col, month, year


def silver_layer(df: "pyspark.sql.DataFrame") -> "pyspark.sql.DataFrame":
    """
    Bronze layer processing (removes missing values, creates time features)
    Time features are later used for partitioning
    TODO: docs what?
    See docs/? for processing decisions
    """
    df_clean = df.fillna({"CustomerID": "missing", "Description": "unknown"})
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.withColumn("Year", year(col("InvoiceData")))
    df_clean = df_clean.withColumn("Month", month(col("InvoiceData")))

    return df_clean
