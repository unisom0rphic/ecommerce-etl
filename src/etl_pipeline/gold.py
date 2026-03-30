# feature engineering,
import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window, types


def gold_layer(spark: pyspark.sql.SparkSession, source_path, target_path):
    # Determine numeric columns and handle outliers
    df = spark.read.parquet(source_path)
    numeric_cols = [
        f.name for f in df.schema.fields if isinstance(f.dataType, types.NumericType)
    ]
    df_enriched = cap_outliers(df, numeric_cols)

    df_enriched = (
        df_enriched.withColumn("Revenue", F.col("Quantity") * F.col("UnitPrice"))
        .withColumn("IsQ4", (F.month("InvoiceDate") >= 10).cast("int"))
        .withColumn(
            # simplified
            "IsHolidayPeriod",
            (F.month("InvoiceDate") == 12).cast("int"),
        )
    )

    # cyclic time features
    df_enriched = (
        df_enriched.withColumn("Month_sin", F.sin(2 * F.pi() * F.col("Month") / 12))
        .withColumn("Month_cos", F.cos(2 * F.pi() * F.col("Month") / 12))
        .withColumn("Weekday_sin", F.sin(2 * F.pi() * F.col("Weekday") / 7))
        .withColumn("Weekday_cos", F.cos(2 * F.pi() * F.col("Weekday") / 7))
    )

    # convert to unix for easier windowing logic
    df_enriched = df_enriched.withColumn("ts_unix", F.unix_timestamp("InvoiceDate"))

    # helper columns
    w_customer = Window.partitionBy("CustomerID").orderBy("InvoiceDate")
    w_stock = Window.partitionBy("StockCode").orderBy("InvoiceDate")

    DAY_SEC = 86400
    RANGE_7D = 7 * DAY_SEC
    RANGE_30D = 30 * DAY_SEC

    df_enriched = df_enriched.withColumn(
        "Days_Since_Last_Purchase",
        F.datediff(F.col("InvoiceDate"), F.lag("InvoiceDate", 1).over(w_customer)),
    )

    df_enriched = df_enriched.withColumn(
        "Cumulative_Purchase_Count",
        F.count("InvoiceNo").over(
            w_customer.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    df_enriched = df_enriched.withColumn(
        "Cumulative_Spend",
        F.sum("Revenue").over(
            w_customer.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    df_enriched = df_enriched.withColumn(
        "Is_First_Purchase", (F.col("Cumulative_Purchase_Count") == 1).cast("int")
    )

    df_enriched = df_enriched.withColumn(
        "Client_Rolling_Rev_30D",
        F.sum("Revenue").over(w_customer.rowsBetween(-RANGE_30D, 0)),
    )

    df_enriched = df_enriched.withColumn(
        "Stock_Rolling_Count_7D",
        F.count("InvoiceNo").over(w_stock.rowsBetween(-RANGE_7D, 0)),
    )

    df_enriched = df_enriched.withColumn(
        "Lag_1_Quantity", F.lag("Quantity", 1).over(w_customer)
    )

    df_enriched.drop("ts_unix")

    # TODO: target column

    return df_enriched


def cap_outliers(df: DataFrame, columns: list[str]):
    """
    Removes outliers via IQR
    """
    if not df.count():
        print("WARNING: DataFrame is empty")
        return df

    for col_name in columns:
        if not isinstance(df.schema[col_name].dataType, types.NumericType):
            raise ValueError(
                f"Capping outliers failed: column {col_name} isn't numeric"
            )

        Q1 = df.agg(F.percentile_approx(col_name, 0.25)).collect()[0][0]
        Q3 = df.agg(F.percentile_approx(col_name, 0.75)).collect()[0][0]

        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) < lower_bound, lower_bound)
            .when(F.col(col_name) > upper_bound, upper_bound)
            .otherwise(F.col(col_name)),
        )

    return df
