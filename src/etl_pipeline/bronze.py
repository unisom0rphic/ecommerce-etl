"""
- Ты просто робот, только имитация жизни.
Робот сможет сочинить симфонию? Робот сможет превратить
кусок холста в шедевр искусств?
- А ты?

Имитация выгрузки файлов из внешних источников
(на самом деле мы просто читаем .csv в parquet)
"""


def bronze_layer(spark, source_path, target_path):
    """
    Reads data from .csv file (we'll use Kafka later)
    """
    # read data (from Kafka)
    df_raw = spark.read.option("header", "true").csv(source_path)

    # should be append in prod
    df_raw.write.mode("overwrite").parquet(target_path)

    total_rows = df_raw.count()
    print(f"Bronze: wrote {total_rows} records into {target_path}")
    return total_rows
