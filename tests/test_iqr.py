import os
import sys
import unittest

from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# TODO: move to utils
from src.etl_pipeline.gold import cap_outliers

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_DAEMON"] = "false"


class TestRemoveOutliers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[1]").appName("IQRTest").getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def _create_df(self, data, schema):
        return self.spark.createDataFrame(data, schema)

    def test_remove_outliers_happy_path(self):
        """Тест: выбросы успешно ограничиваются границами IQR"""
        data = [(1,), (2,), (3,), (4,), (5,), (100,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])
        result_data = sorted([row.value for row in result_df.collect()])

        self.assertNotEqual(result_data[-1], 100)
        self.assertEqual(result_data[0], 1)
        self.assertEqual(result_data[1], 2)

    def test_non_numeric_column_raises_error(self):
        """Raises ValueError on non-numeric columns"""
        data = [("A",), ("B",), ("C",)]
        schema = StructType([StructField("text_col", StringType(), True)])
        df = self._create_df(data, schema)

        with self.assertRaises(ValueError):
            cap_outliers(df, ["text_col"])

    def test_no_outliers(self):
        """No outliers -> no changes"""
        data = [(10,), (11,), (12,), (13,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])

        original_values = sorted([row.value for row in df.collect()])
        result_values = sorted([row.value for row in result_df.collect()])

        self.assertEqual(original_values, result_values)

    def test_empty_dataframe(self):
        """Empty DataFrame -> returns empty DataFrame"""
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df([], schema)

        result_df = cap_outliers(df, ["value"])
        self.assertEqual(result_df.count(), 0)

    def test_float_values(self):
        """Test floats"""
        data = [(1.5,), (2.5,), (3.5,), (4.5,), (100.0,)]
        schema = StructType([StructField("value", FloatType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        self.assertNotIn(100.0, result_data)

    def test_negative_outliers(self):
        """Test negative outliers"""
        data = [(-100,), (1,), (2,), (3,), (4,), (5,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        self.assertNotIn(-100, result_data)


if __name__ == "__main__":
    unittest.main()
