import pyspark.sql
import pyspark.sql.functions as F
from pyspark.sql import types
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


def remove_outliers(df: pyspark.sql.DataFrame, columns: list[str]):
    """
    Removes outliers via IQR
    """
    if not df.count():
        print("WARNING: DataFrame is empty")
        return df

    for col_name in columns:
        if not isinstance(df.schema[col_name].dataType, types.NumericType):
            raise ValueError(
                f"Removing outliers failed: column {col_name} isn't numeric"
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
