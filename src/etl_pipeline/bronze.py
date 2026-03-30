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
    Reads data from .csv file (we'll make Kafka later)
    """
    # read data (from Kafka)
    df_raw = spark.read.option("header", "true").csv(source_path)
    table_name = "data"

    # should be append in prod
    df_raw.write.mode("overwrite").parquet(target_path)

    print(f"Bronze: wrote {df_raw.count()} records into {table_name}")
