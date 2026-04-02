from src.etl_pipeline.bronze import bronze_layer
from src.etl_pipeline.gold import gold_layer
from src.etl_pipeline.silver import silver_layer


def etl_pipeline(table_name, spark):
    warehouse_path = spark.conf.get("spark.sql.warehouse.dir")
    bronze_rows = bronze_layer(
        # TODO: change source path to a variable later (or kafka topic)
        spark,
        source_path=f"{warehouse_path}/../data/{table_name}.csv",
        target_path=f"{warehouse_path}/bronze",
    )
    silver_rows = silver_layer(
        spark,
        source_path=f"{warehouse_path}/bronze",
        target_path=f"{warehouse_path}/silver",
    )
    assert bronze_rows <= silver_rows
    gold_rows = gold_layer(
        spark,
        source_path=f"{warehouse_path}/silver",
        target_path=f"{warehouse_path}/gold",
    )
    assert silver_rows <= gold_rows
    # TODO: tests
    # TODO: vectorize features
