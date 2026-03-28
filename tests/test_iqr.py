import os
import shutil
import sys
import tempfile
import unittest
import uuid

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# TODO: move to utils
from src.etl_pipeline.gold import remove_outliers

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_DAEMON"] = "false"


# TODO: review tests
class TestRemoveOutliers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()

        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("LocalTestSession")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.python.worker.timeout", "120")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _create_df(self, data, schema):
        file_path = os.path.join(self.tmp_dir, f"temp_{uuid.uuid4().hex}.csv")

        with open(file_path, "w", encoding="utf-8") as f:
            headers = [field.name for field in schema.fields]
            f.write(",".join(headers) + "\n")

            for row in data:
                row_str = ",".join([str(v) if v is not None else "" for v in row])
                f.write(row_str + "\n")

        return self.spark.read.option("header", "true").schema(schema).csv(file_path)

    def test_agg_collect(self):
        """Проверяем, что agg + collect работает"""
        df = self.spark.createDataFrame([(1,), (2,), (3,)], ["v"])
        result = df.agg(F.percentile_approx("v", 0.5)).collect()
        print(f"Agg result: {result}")
        self.assertEqual(len(result), 1)

    def test_count(self):
        """Проверяем, что count работает"""
        df = self.spark.createDataFrame([(1,), (2,)], ["v"])
        self.assertEqual(df.count(), 2)

    def test_remove_outliers_happy_path(self):
        """Тест: выбросы успешно ограничиваются границами IQR"""
        data = [(1,), (2,), (3,), (4,), (5,), (100,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = remove_outliers(df, ["value"])
        result_data = sorted([row.value for row in result_df.collect()])

        self.assertNotEqual(result_data[-1], 100)
        self.assertEqual(result_data[0], 1)
        self.assertEqual(result_data[1], 2)

    def test_non_numeric_column_raises_error(self):
        """Тест: функция должна выбрасывать ошибку для нечисловых колонок"""
        data = [("A",), ("B",), ("C",)]
        schema = StructType([StructField("text_col", StringType(), True)])
        df = self._create_df(data, schema)

        with self.assertRaises(ValueError):
            remove_outliers(df, ["text_col"])

    def test_no_outliers(self):
        """Тест: если выбросов нет, данные не меняются"""
        data = [(10,), (11,), (12,), (13,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = remove_outliers(df, ["value"])

        original_values = sorted([row.value for row in df.collect()])
        result_values = sorted([row.value for row in result_df.collect()])

        self.assertEqual(original_values, result_values)

    def test_empty_dataframe(self):
        """Тест: обработка пустого DataFrame"""
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df([], schema)

        result_df = remove_outliers(df, ["value"])
        self.assertEqual(result_df.count(), 0)

    def test_float_values(self):
        """Тест: работа с float значениями"""
        data = [(1.5,), (2.5,), (3.5,), (4.5,), (100.0,)]
        schema = StructType([StructField("value", FloatType(), True)])
        df = self._create_df(data, schema)

        result_df = remove_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        self.assertNotIn(100.0, result_data)

    def test_negative_outliers(self):
        """Тест: обработка отрицательных выбросов"""
        data = [(-100,), (1,), (2,), (3,), (4,), (5,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = remove_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        self.assertNotIn(-100, result_data)

    def test_null_values(self):
        """Тест: обработка NULL значений"""
        data = [(1,), (2,), (None,), (4,), (5,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = remove_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        self.assertNotIn(None, result_data)


if __name__ == "__main__":
    unittest.main()
