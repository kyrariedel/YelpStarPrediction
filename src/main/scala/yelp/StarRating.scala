package yelp

import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, NGram, SQLTransformer, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.functions.{abs, avg, col, least, greatest, lit, round => sparkRound, when}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

/**
  * This program predicts Yelp review star ratings from review text using Spark MLlib.
  * Text is turned into TF-IDF features (unigrams + bigrams), then a decision tree is compared to a random forest regressor.
  */

object StarRating {

  def main(args: Array[String]): Unit = {
    val logger = LogManager.getRootLogger
    if (args.length < 2 || args.length > 7) {
      logger.error("Usage:\nyelp.StarRating <input dir> <output dir> " +
        "[numFeatures=2048] [trainFraction=0.8] [maxDepth=8] [numTrees=20] [seed=42]")
      System.exit(1)
    }
    val inputPath = args(0)
    val outputPath = args(1)
    val numFeatures = if (args.length > 2) args(2).toInt else 2048
    val trainFraction = if (args.length > 3) args(3).toDouble else 0.8
    val maxDepth = if (args.length > 4) args(4).toInt else 8
    val numTrees = if (args.length > 5) args(5).toInt else 20
    val seed = if (args.length > 6) args(6).toLong else 42L

    val spark = SparkSession.builder()
      .appName("Yelp Star Rating Regression")
      .getOrCreate()

    val t0 = System.nanoTime()

    // Load review JSON; keep id, text, and stars as the label
    val reviews = spark.read.json(inputPath)
      .select(
        col("review_id"),
        col("text"),
        col("stars").cast("double").alias("label")
      )
      .filter(col("text").isNotNull && col("label").isNotNull)

    val n = reviews.count()
    logger.info(s"Loaded $n reviews from $inputPath")
    val Array(trainRaw, testRaw) = reviews.randomSplit(Array(trainFraction, 1.0 - trainFraction), seed)

    // tokenize + lowercase, drop stopwords, add bigrams so "not good" stays together
    // HashingTF = bag of words of fixed size; IDF down-weights common terms (fit on train)
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val stopwords = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("unigrams")
    val bigrams = new NGram()
      .setN(2)
      .setInputCol("unigrams")
      .setOutputCol("bigrams")
    // https://www.crowdstrike.com/en-us/blog/deep-dive-into-custom-spark-transformers-for-machine-learning-pipelines/
    val concatTokens = new SQLTransformer()
      .setStatement("SELECT *, concat(unigrams, bigrams) AS tokens FROM __THIS__")
    val hashingTF = new HashingTF()
      .setInputCol("tokens")
      .setOutputCol("rawFeatures")
      .setNumFeatures(numFeatures)
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
    val featurizer = new Pipeline().setStages(
      Array(tokenizer, stopwords, bigrams, concatTokens, hashingTF, idf)
    )

    val featModel = featurizer.fit(trainRaw)
    val train = featModel.transform(trainRaw).cache()
    val test = featModel.transform(testRaw)
    val trainCount = train.count()
    val testCount = test.count()
    logger.info(s"Train=$trainCount Test=$testCount numFeatures=$numFeatures maxDepth=$maxDepth numTrees=$numTrees")

    // single tree vs ensemble
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxDepth(maxDepth)
      .setSeed(seed)

    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxDepth(maxDepth)
      .setNumTrees(numTrees)
      .setFeatureSubsetStrategy("auto")
      .setSeed(seed)

    val tDt0 = System.nanoTime()
    val dtModel = dt.fit(train)
    val dtTrainSec = (System.nanoTime() - tDt0) / 1e9
    val dtPred = dtModel.transform(test)
    val dtMetrics = evaluate(dtPred, "dt")

    val tRf0 = System.nanoTime()
    val rfModel = rf.fit(train)
    val rfTrainSec = (System.nanoTime() - tRf0) / 1e9
    val rfPred = rfModel.transform(test)
    val rfMetrics = evaluate(rfPred, "rf")

    train.unpersist()

    val totalSec = (System.nanoTime() - t0) / 1e9
    val dtDepth = dtModel.asInstanceOf[DecisionTreeRegressionModel].depth
    val rfTrees = rfModel.asInstanceOf[RandomForestRegressionModel].getNumTrees

    val summary = Seq(
      s"input=$inputPath",
      s"reviews=$n",
      s"train=$trainCount",
      s"test=$testCount",
      s"numFeatures=$numFeatures",
      s"trainFraction=$trainFraction",
      s"maxDepth=$maxDepth",
      s"numTrees=$numTrees",
      s"seed=$seed",
      s"dt_depth=$dtDepth",
      s"rf_trees=$rfTrees",
      s"dt_rmse=${dtMetrics.rmse}",
      s"dt_mae=${dtMetrics.mae}",
      s"dt_rounded_accuracy=${dtMetrics.roundedAccuracy}",
      s"dt_within1_accuracy=${dtMetrics.within1Accuracy}",
      s"dt_bucket3_accuracy=${dtMetrics.bucket3Accuracy}",
      s"dt_train_seconds=$dtTrainSec",
      s"rf_rmse=${rfMetrics.rmse}",
      s"rf_mae=${rfMetrics.mae}",
      s"rf_rounded_accuracy=${rfMetrics.roundedAccuracy}",
      s"rf_within1_accuracy=${rfMetrics.within1Accuracy}",
      s"rf_bucket3_accuracy=${rfMetrics.bucket3Accuracy}",
      s"rf_train_seconds=$rfTrainSec",
      s"total_seconds=$totalSec"
    )

    summary.foreach(logger.info)
    spark.sparkContext.parallelize(summary, 1).saveAsTextFile(s"$outputPath/metrics")

    // sample of 20 test rows, worst RF errors first (for the report)
    val sample = dtPred
      .select(
        col("review_id"),
        col("label").alias("actual"),
        col("prediction").alias("pred_dt")
      )
      .join(
        rfPred.select(col("review_id"), col("prediction").alias("pred_rf")),
        Seq("review_id")
      )
      .withColumn("pred_dt_rounded", clampStar(sparkRound(col("pred_dt"))))
      .withColumn("pred_rf_rounded", clampStar(sparkRound(col("pred_rf"))))
      .orderBy(abs(col("actual") - col("pred_rf")).desc)
      .limit(20)

    val sampleLines = sample.collect().map { r =>
      f"${r.getString(0)}\t${r.getDouble(1)}%.1f\t${r.getDouble(2)}%.3f\t${r.getDouble(3)}%.3f\t${r.getDouble(4)}%.0f\t${r.getDouble(5)}%.0f"
    }
    val sampleOut = Seq("review_id\tactual\tpred_dt\tpred_rf\tpred_dt_rounded\tpred_rf_rounded") ++ sampleLines
    spark.sparkContext.parallelize(sampleOut, 1).saveAsTextFile(s"$outputPath/sample_predictions")

    writePredictions(dtPred, s"$outputPath/predictions_dt")
    writePredictions(rfPred, s"$outputPath/predictions_rf")
    spark.stop()
  }

  // round/clamp helper so predicted stars stay in [1, 5]
  private def clampStar(c: org.apache.spark.sql.Column) = greatest(lit(1.0), least(lit(5.0), c))

  private case class Metrics(
      rmse: Double,
      mae: Double,
      roundedAccuracy: Double,
      within1Accuracy: Double,
      bucket3Accuracy: Double
  )

  // 1-2 / 3 / 4-5
  private def starBucket(c: org.apache.spark.sql.Column) =
    when(c <= 2.0, lit(0)).when(c <= 3.0, lit(1)).otherwise(lit(2))

  // RMSE and MAE; also exact rounded star, within 1 star, and 3-class bucket
  private def evaluate(pred: DataFrame, tag: String): Metrics = {
    val evaluatorRmse = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val evaluatorMae = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    val scored = pred
      .withColumn("rounded", clampStar(sparkRound(col("prediction"))))
      .withColumn("correct", (col("rounded") === col("label")).cast("double"))
      .withColumn("within1", (abs(col("rounded") - col("label")) <= 1.0).cast("double"))
      .withColumn("bucket_ok", (starBucket(col("rounded")) === starBucket(col("label"))).cast("double"))

    val rmse = evaluatorRmse.evaluate(pred)
    val mae = evaluatorMae.evaluate(pred)
    val roundedAccuracy = scored.agg(avg(col("correct"))).first().getDouble(0)
    val within1Accuracy = scored.agg(avg(col("within1"))).first().getDouble(0)
    val bucket3Accuracy = scored.agg(avg(col("bucket_ok"))).first().getDouble(0)
    LogManager.getRootLogger.info(
      f"$tag rmse=$rmse%.4f mae=$mae%.4f rounded_accuracy=$roundedAccuracy%.4f " +
        f"within1=$within1Accuracy%.4f bucket3=$bucket3Accuracy%.4f"
    )
    Metrics(rmse, mae, roundedAccuracy, within1Accuracy, bucket3Accuracy)
  }

  // CSV: review_id, actual, prediction, rounded prediction
  private def writePredictions(pred: DataFrame, path: String): Unit = {
    pred
      .select(
        col("review_id"),
        col("label").alias("actual"),
        col("prediction"),
        clampStar(sparkRound(col("prediction"))).alias("prediction_rounded")
      )
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(path)
  }
}
