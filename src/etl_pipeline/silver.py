from pyspark.sql.functions import col, month, year


def silver_layer(df: "pyspark.sql.DataFrame"):
    df_clean = df.fillna({"CustomerID": "missing", "Description": "missing"})
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.withColumn("year", year(col("InvoiceData")))
    df_clean = df_clean.withColumn("month", month(col("InvoiceData")))

    return df_clean
