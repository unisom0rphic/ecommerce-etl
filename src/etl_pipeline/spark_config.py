# строим session, добавляем пути к данным
import os

from pyspark.sql import SparkSession


# fmt: off
# maybe use spark_config.yaml or something
def get_spark_session():
    # fix for windows
    # TODO: move to env
    os.environ["HADOOP_HOME"] = r"C:\Users\Wednesday\winutils\hadoop-3.0.0"
    os.environ["PATH"] = f"{os.environ['HADOOP_HOME']}\\bin;{os.environ['PATH']}"

    return SparkSession.builder \
        .master('local') \
        .appName('E-Commerce ETL') \
        .config('spark.sql.warehouse.dir', 'spark-warehouse') \
        .getOrCreate()
# fmt: on

spark = get_spark_session()
print(f"Name: {spark.conf.get('spark.app.name')}")
print(f"Working directory: {spark.conf.get(('spark.sql.warehouse.dir'))}")
