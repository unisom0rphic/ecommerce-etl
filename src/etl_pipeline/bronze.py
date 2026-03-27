"""
- Ты просто робот, только имитация жизни.
Робот сможет сочинить симфонию? Робот сможет превратить
кусок холста в шедевр искусств?
- А ты?

Имитация выгрузки файлов из внешних источников
(на самом деле мы просто читаем .csv в delta)
"""

from spark_config import get_spark_session


def bronze_layer():
    spark = get_spark_session()

    path = spark.conf.get("spark.sql.warehouse.dir")

    df_raw = spark.read.option("header", "true").csv(f"{path}/../data/data.csv")
    table_name = "data"  # file_path.split('/')[-1].split('.')[0]

    df_raw.write.format("parquet").mode("append").save(f"{path}/bronze/{table_name}")

    print(f"Bronze: wrote {df_raw.count()} records into {table_name}")


if __name__ == "__main__":
    bronze_layer()
