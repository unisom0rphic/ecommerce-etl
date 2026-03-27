# строим session, добавляем пути к данным
from pyspark.sql import SparkSession


# fmt: off
# maybe use spark_config.yaml or something
def get_spark_session():
    return SparkSession.builder \
        .master('local') \
        .appName('E-Commerce ETL') \
        .config('spark.sql.warehouse.dir', 'spark-warehouse') \
        .getOrCreate()
# fmt: on

spark = get_spark_session()
print(f"Name: {spark.conf.get('spark.app.name')}")
print(f"Working directory: {spark.conf.get(('spark.sql.warehouse.dir'))}")
