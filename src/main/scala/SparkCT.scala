
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
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
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}


object SparkCT extends App {

  override def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("SparkCT")
    val sc = new SparkContext(conf)

    // Set numIterations large enough so that it stops early.
    //   val numIterations = 20
    //   val trainRdd = sc.parallelize(OldGBTSuite.trainData, 2).map(_.asML)

    //val data: DataFrame = spark.read.format("libsvm").load("data/ctdata/*")
    val data = MLUtils.loadLibSVMFile(sc, args(0)).repartition(args(3).toInt)
/*
    val rawData = sc.textFile(args(0)).map(_.split("\\s")).map(x => {
      if (x(0).toInt > 0)
        x(0) = "1"
      else
        x(0) = "-1"

      val v = x.drop(1).map(_.split(":")).map( x => (x(0).toInt - 1, x(1).toDouble)).sortBy(_._1)
      (x(0).toInt, v)
    }).repartition(args(3).toInt)

    val length = rawData.map(_._2.last._1).max + 1

    val data: RDD[LabeledPoint] = rawData.map{case(label, v) => LabeledPoint(label, Vectors.sparse(length,
 v.map(_._1), v.map(_._2)))}
*/
    // Split the data into training and test sets (30% held out for testing).
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.treeStrategy.algo = Classification
    boostingStrategy.numIterations = args(1).toInt // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.impurity = Gini
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    boostingStrategy.validationTol = args(2).toDouble
/*
    val model = CGradientBoostedTrees.trainWithValidation(trainingData, testData, boostingStrategy)
    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test with validation Error = " + testErr)

*/
    var bestErr = 1.0
    var bestModel: GradientBoostedTreesModel = null
    var bestNumTree = 0
    for (numTree <- Array(50, 100, 150, 200, 250, 300, 350, 400)) {
      boostingStrategy.numIterations = numTree
      val modelGBT: GradientBoostedTreesModel = CGradientBoostedTrees.train(trainingData, boostingStrategy)

      val labelAndPreds2 = testData.map { point =>
        val prediction = modelGBT.predict(point.features)
        (point.label, prediction)
      }

      val testErr2 = labelAndPreds2.filter(r => r._1 != r._2).count.toDouble / testData.count()
      println("GBT Test without validation Error = " + testErr2 + ", with numTree:" + numTree)
      if (bestErr < testErr2) {
        bestModel = modelGBT
        bestNumTree = numTree
      }
      //println("Learned classification GBT model:\n" + model.toDebugString)
    }
    println("GBT Test without validation best Error = " + bestErr + ", with numTree:" + bestNumTree)
    /*
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = args(1).toInt // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val modelRF = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("GBT Test without validation Error = " + testErr2 + "RF Test Error = " + testErr)
    */
  }
}

