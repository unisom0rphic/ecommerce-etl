# feature engineering,
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, types


def gold_layer(df: DataFrame):
    numeric_cols = [
        f.name for f in df.schema.fields if isinstance(f.dataType, types.NumericType)
    ]
    df_enriched = remove_outliers(df, numeric_cols)

    # TODO: feature engineering
    df_enriched = df_enriched.withColumn(
        "Revenue", F.col("Quantity") * F.col("UnitPrice")
    )

    # from pyspark.sql.functions import when
    # df = df.withColumn("category",
    #     when(col("age") < 18, "Minor")
    #     .when(col("age") < 65, "Adult")
    #     .otherwise("Senior")
    # )

    return df_enriched


def remove_outliers(df: DataFrame, columns: list[str]):
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
