
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object TrafficDataProcessor1 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Traffic Congestion Prediction")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Load and clean data
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/traffic_congestion_fake_dataset.csv")

    // Trim and clean date/time columns
    val dfTrimmed = df
      .withColumn("date", trim(col("date")))
      .withColumn("time", trim(col("time")))

    // Combine date and time into timestamp
    val dfWithTimestamp = dfTrimmed
      .withColumn("datetime_string", concat_ws(" ", col("date").cast("string"), date_format(col("time"), "HH:mm")))
      .withColumn("timestamp", to_timestamp(col("datetime_string"), "dd-MM-yyyy HH:mm"))

      .withColumn("hour", hour(col("timestamp")))
      .withColumn("is_congested", when(col("congestion_level") >= 3, 1).otherwise(0)) // Binary label

    // Index location
    val locationIndexer = new StringIndexer()
      .setInputCol("location")
      .setOutputCol("location_index")

    val indexedDF = locationIndexer.fit(dfWithTimestamp).transform(dfWithTimestamp)

    // Handle nulls before assembling
    val cleanedDF = indexedDF.na.fill(0, Seq("hour", "location_index"))

    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "location_index"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val output = assembler.transform(cleanedDF)

    // Train logistic regression model
    val lr = new LogisticRegression()
      .setLabelCol("is_congested")
      .setFeaturesCol("features")

    val model = lr.fit(output)

    // Make predictions
    val predictions = model.transform(output)
    predictions.select("timestamp", "location", "congestion_level", "prediction", "probability", "is_congested").show(false)

    // Evaluate model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("is_congested")
      .setRawPredictionCol("prediction")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Model Accuracy: $accuracy")

    spark.stop()
  }
}
