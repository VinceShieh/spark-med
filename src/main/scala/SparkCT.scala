
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
//import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.{GradientBoostedTrees => NewGBT}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree.CGradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.loss.LogLoss
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}


object SparkCT extends App {

  override def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("SparkCT").setMaster("local[4]")
    val sc = new SparkContext(conf)

    // Set numIterations large enough so that it stops early.
    //   val numIterations = 20
    //   val trainRdd = sc.parallelize(OldGBTSuite.trainData, 2).map(_.asML)

    //val data: DataFrame = spark.read.format("libsvm").load("data/ctdata/*")
//    val data = MLUtils.loadLibSVMFile(sc, "data/ctdata/*").repartition(144)

    val rawData = sc.textFile("data/5line", 4).map(_.split("\\s")).map(x => {
      if (x(0).toInt > 3)
        x(0) = "1"
      else
        x(0) = "-1"

      (x(0).toInt, x.drop(1).map(_.toDouble))
    }).repartition(4)


    val data: RDD[LabeledPoint] = rawData.map{case(label, v) => LabeledPoint(label, Vectors.dense(v))}

    // Split the data into training and test sets (30% held out for testing).
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.treeStrategy.algo = Classification
    boostingStrategy.numIterations = 50 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.impurity = Variance
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    boostingStrategy.validationTol = 0.001

    val model = CGradientBoostedTrees.trainWithValidation(trainingData, testData, boostingStrategy)
    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test with validation Error = " + testErr)

/*
    val model2 = CGradientBoostedTrees.train(trainingData, boostingStrategy)

    val labelAndPreds2 = testData.map { point =>
      val prediction = model2.predict(point.features)
      (point.label, prediction)
    }

    val testErr2 = labelAndPreds2.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test without validation Error = " + testErr2)
    //println("Learned classification GBT model:\n" + model.toDebugString)
*/

  }
}

