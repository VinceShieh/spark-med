$SPARK_HOME/bin/spark-submit \
    --class SparkCT \
    --master spark://sr443:7077 \
    --driver-memory 180g \
    --total-executor-cores 144 \
    --executor-memory 180g \
    --conf spark.kryoserializer.buffer.max=2047m \
    --conf spark.driver.maxResultSize=20g \
    target/scala-2.11/spark-med_2.11-0.0.1.jar \
    hdfs://sr443/data/ctdata/* \
    100 \
    0.0001 \
    144 
