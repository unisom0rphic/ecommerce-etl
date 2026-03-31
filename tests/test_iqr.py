import os
import sys

import pytest
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# TODO: move to utils
from src.etl_pipeline.gold import cap_outliers
from src.etl_pipeline.spark_config import get_spark_session

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_DAEMON"] = "false"


@pytest.fixture(scope="session")
def spark_session():
    spark = get_spark_session()
    yield spark
    spark.stop()


class TestRemoveOutliers:
    @pytest.fixture(autouse=True)
    def _setup_spark(self, spark_session):
        self.spark = spark_session

    def _create_df(self, data, schema):
        return self.spark.createDataFrame(data, schema)

    def test_remove_outliers_happy_path(self):
        """IQR bounds the outliers"""
        data = [(1,), (2,), (3,), (4,), (5,), (100,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])
        result_data = sorted([row.value for row in result_df.collect()])

        assert result_data[-1] != 100
        assert result_data[0] == 1
        assert result_data[1] == 2

    def test_non_numeric_column_raises_error(self):
        """Raises ValueError on non-numeric columns"""
        data = [("A",), ("B",), ("C",)]
        schema = StructType([StructField("text_col", StringType(), True)])
        df = self._create_df(data, schema)

        with pytest.raises(ValueError):
            cap_outliers(df, ["text_col"])

    def test_no_outliers(self):
        """No outliers -> no changes"""
        data = [(10,), (11,), (12,), (13,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])

        original_values = sorted([row.value for row in df.collect()])
        result_values = sorted([row.value for row in result_df.collect()])

        assert original_values == result_values

    def test_empty_dataframe(self):
        """Empty DataFrame -> returns empty DataFrame"""
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df([], schema)

        result_df = cap_outliers(df, ["value"])
        assert result_df.count() == 0

    def test_float_values(self):
        """Test floats"""
        data = [(1.5,), (2.5,), (3.5,), (4.5,), (100.0,)]
        schema = StructType([StructField("value", FloatType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        assert 100.0 not in result_data

    def test_negative_outliers(self):
        """Test negative outliers"""
        data = [(-100,), (1,), (2,), (3,), (4,), (5,)]
        schema = StructType([StructField("value", IntegerType(), True)])
        df = self._create_df(data, schema)

        result_df = cap_outliers(df, ["value"])
        result_data = [row.value for row in result_df.collect()]

        assert -100 not in result_data
