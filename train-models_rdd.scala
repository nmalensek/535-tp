import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
//import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

val fullDataset = spark.read.format("parquet").load("/project/data2")
val labeled = fullDataset.select("AirlineDelay", "features").withColumnRenamed("AirlineDelay", "label")
//val labeled_rdd = labeled.rdd.map(row => LabeledPoint(row.getAs[Double]("label"), row.getAs[Vector]("features")))

val labeled_rdd3 = labeled.rdd.map(row => LabeledPoint(row.getAs[Float]("label"), org.apache.spark.mllib.linalg.Vectors.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense)))
val model = LinearRegressionWithSGD.train(labeled_rdd3, 50, 0.001) // arbitrary hyperparameters for now
val valsAndPreds = labeled_rdd4.map{ pt =>
	val prediction = model.predict(pt.features)
	(pt.label, prediction)
	}
val MSE = valsAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
val weights = model.weights
val model2 = LinearRegressionWithSGD.train(labeled_rdd4, 10, 0.1, 1.0, weights)
