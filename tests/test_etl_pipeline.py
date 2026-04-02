import os
import sys
from pathlib import Path

import pyspark.sql.functions as F
import pytest
from pyspark.sql import DataFrame

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_DAEMON"] = "false"

from src.etl_pipeline.pipeline import etl_pipeline
from src.etl_pipeline.spark_config import get_spark_session

OUTLIER_VALUE = 99999
MISSING = ""


@pytest.fixture(scope="session")
def spark_session():
    spark = get_spark_session()
    yield spark
    spark.stop()


@pytest.fixture(scope="function")
def test_csv(spark_session):
    raw_warehouse = spark_session.conf.get("spark.sql.warehouse.dir")
    # FIXME
    if raw_warehouse.startswith("file:/"):
        raw_warehouse = raw_warehouse.replace("file:/", "")

    warehouse_path = Path(raw_warehouse)
    print(f"Raw path: {warehouse_path}")

    csv_path = warehouse_path.parent / "data" / "test.csv"

    # Contains missing values (CustomerID/Description), outliers (99999)
    # WARNING: do not use big tables here as collect() is being called later, which might lead to OOM
    csv_content = f"""InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/1/2010 8:26,2.55,17850,United Kingdom
536365,71053,WHITE METAL LANTERN,{OUTLIER_VALUE},12/1/2010 8:26,{OUTLIER_VALUE},17850,United Kingdom
536365,84406B,CREAM CUPID HEARTS COAT HANGER,8,12/1/2010 8:26,2.75,{MISSING},United Kingdom
536365,84029G,KNITTED UNION FLAG HOT WATER BOTTLE,6,12/1/2010 8:26,3.39,17850,United Kingdom
536365,84029E,RED WOOLLY HOTTIE WHITE HEART.,6,12/1/2010 8:26,{OUTLIER_VALUE},17850,United Kingdom
536365,22752,{MISSING},2,12/1/2010 8:26,7.65,17850,United Kingdom
536365,21730,GLASS STAR FROSTED T-LIGHT HOLDER,6,12/1/2010 8:26,4.25,{MISSING},United Kingdom
536366,22633,HAND WARMER UNION JACK,6,12/1/2010 8:28,1.85,17850,United Kingdom
536366,22632,HAND WARMER RED POLKA DOT,{OUTLIER_VALUE},12/1/2010 8:28,1.85,17850,United Kingdom
536367,84879,{MISSING},32,12/1/2010 8:34,1.69,{MISSING},United Kingdom
536367,22745,POPPY'S PLAYHOUSE BEDROOM ,6,12/1/2010 8:34,2.1,13047,United Kingdom
536367,22748,POPPY'S PLAYHOUSE KITCHEN,6,12/1/2010 8:34,2.1,13047,United Kingdom
536367,22749,FELTCRAFT PRINCESS CHARLOTTE DOLL,8,12/1/2010 8:34,3.75,13047,United Kingdom
536367,22310,{MISSING},6,12/1/2010 8:34,1.65,13047,United Kingdom
536367,84969,BOX OF 6 ASSORTED COLOUR TEASPOONS,6,12/1/2010 8:34,4.25,13047,United Kingdom
"""
    print(f"Writing CSV to: {csv_path.absolute()}")
    print(f"Parent exists: {csv_path.parent.exists()}")
    csv_path.touch(exist_ok=True)
    csv_path.write_text(csv_content)

    yield csv_path

    if csv_path.exists():
        csv_path.unlink()


class TestETLPipeline:
    @pytest.fixture(autouse=True)
    def _setup_spark(self, spark_session):
        self.spark = spark_session

    def _read_layer_data(self, layer_name):
        raw_warehouse = self.spark.conf.get("spark.sql.warehouse.dir")
        # FIXME: I don't think just removing file:/ from path is reliable
        if raw_warehouse.startswith("file:/"):
            raw_warehouse = raw_warehouse.replace("file:/", "")

        warehouse_path = Path(raw_warehouse)
        path = str(warehouse_path / "gold")
        print(f"Parquet path: {path}")
        try:
            return self.spark.read.parquet(path)
        except Exception:
            raise RuntimeError(f"Unable to read parquet: {layer_name}\nPath: {path}")

    # TODO: consider splitting this test into multiple ones
    def test_pipeline_cleans_outliers_and_nulls(self, spark_session, test_csv):
        """
        Ensure pipeline handles outliers and nulls
        """
        etl_pipeline("test", self.spark)

        gold_df: DataFrame = self._read_layer_data("gold")

        null_counts_df = gold_df.select(
            [
                F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
                for c in gold_df.columns
            ]
        )
        null_counts = zip(gold_df.columns, null_counts_df.collect()[0])
        for col, count in null_counts:
            assert count == 0, f"Missing values found in Gold layer:\nColumn {col}"

        # Outliers
        max_value = gold_df.select(F.max("Quantity")).collect()[0][0]
        assert max_value < OUTLIER_VALUE, "Outliers found in Gold layer"
        max_value = gold_df.select(F.max("UnitPrice")).collect()[0][0]
        assert max_value < OUTLIER_VALUE, "Outliers found in Gold layer"

        expected_derived_fields = [  # not an exhaustive check
            "Revenue",
            "IsQ4",
            "IsHolidayPeriod",
            "Cumulative_Purchase_Count",
            "Is_First_Purchase",
            "Client_Rolling_Rev_30D",
            "Monthly_Revenue",
        ]
        field_names = [f.name for f in gold_df.schema.fields]
        for exp_field in expected_derived_fields:
            assert exp_field in field_names, (
                f"Expected field not found in Gold Layer: {exp_field}"
            )
