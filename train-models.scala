import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.udf

object TrainModels {
    def main(args: Array[String]) {
        if (args.length != 3) {
            println("Incorrect number of arguments. Usage:")
            println("<input dir> <output dir> <local | yarn>")
            return
        }

        //pre-processed files that contains feature column
        val inputPath = args(0)
        val outputPath = args(1)
        val specifiedMaster = args(2)

        val spark = SparkSession.builder.appName("TeamDenverFlights").master(specifiedMaster).getOrCreate()
        import spark.implicits._

        //hard code paths for now?
        //Remove US and FL airlines because they're not in the testing set, so we shouldn't train them
        val fullDataset = spark.read.format("parquet").load(inputPath + "/data_with_features/")
            .where($"Reporting_Airline" =!= "US").where($"Reporting_Airline" =!= "FL").where($"Reporting_Airline" =!= "YX")
            .where($"Reporting_Airline" =!= "OH").where($"Reporting_Airline" =!= "G4")

        val with_delay = fullDataset.withColumn("isDelayed", isDelayed(fullDataset("AirlineDelay"), fullDataset("NonAirlineDelay")))
        
        val airlineIds = spark.read.format("parquet").load(inputPath + "/airline_codes/")
        val filteredAirlines = airlineIds.where($"Reporting_Airline" =!= "US").where($"Reporting_Airline" =!= "FL").where($"Reporting_Airline" =!= "YX")
            .where($"Reporting_Airline" =!= "OH").where($"Reporting_Airline" =!= "G4")
        
        val airlineArray = filteredAirlines.select("id").rdd.map(x => x(0)).collect()
        val airportIds = spark.read.format("parquet").load(inputPath + "/airport_codes/")

        //get all flights from 2013-01-01 to 2016-12-31 for training and flights from 2017-01-01 onward for testing.
        val fullTraining = fullDataset.where($"FlightDate" < 1483254000).withColumnRenamed("AirlineDelay", "label")
        val fullTesting = fullDataset.where($"FlightDate" >= 1483254000).withColumnRenamed("AirlineDelay", "label")

        val trainProb = with_delay.where($"FlightDate" < 1483254000).withColumnRenamed("isDelayed", "label")
        val testProb = with_delay.where($"FlightDate" < 1483254000).withColumnRenamed("isDelayed", "label")

        var regressionModel = new LinearRegression().setMaxIter(10)
        var probabilityModel = new LinearRegression().setMaxIter(10)

        val nonAirlineDelayTraining = fullDataset.where($"FlightDate" < 1483254000).withColumnRenamed("NonAirlineDelay", "label")
        val allDelay = regressionModel.fit(nonAirlineDelayTraining)
        allDelay.save("/project/models/nonAirlineDelays")

        //for each Airline in airlineIds, filter the data by AirlineId, select the ML input columns, train the model, save the trained model to disk (use its save() method)
        for (airlineId <- airlineArray) {
            val filteredTraining = fullTraining.where($"AirlineId" === airlineId).select("label","features")

            val trainedModel = regressionModel.fit(filteredTraining)

            trainedModel.save("/project/models/amountModel" + airlineId)

            val filteredProbability = trainProb.where($"AirlineId" === airlineId).select("label","features")

            val trainedProbModel = probabilityModel.fit(filteredProbability)

            trainedProbModel.save("/project/models/probabilityModel" + airlineId)
        }   
    }

    def isDelayed = udf((x: Float, y: Float) => if(x > 0.0 || y > 0.0) 1.0 else 0.0)
}

//fullDataset example rows
//+----------+-----------------+-------------+------+------------+----+-------------+------------+---------------+---------+----------------+--------------------+
//|FlightDate|Reporting_Airline|    AirlineId|Origin|    OriginId|Dest|       DestId|AirlineDelay|NonAirlineDelay|Cancelled|CancellationCode|            features|
//+----------+-----------------+-------------+------+------------+----+-------------+------------+---------------+---------+----------------+--------------------+
//|1543302000|               YX|1322849927168|   EWR|635655159808| DTW|1554778161152|         0.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   EWR|635655159808| IND|1511828488192|        98.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   BUF|455266533376| EWR| 635655159808|         0.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   EWR|635655159808| MSP|1589137899520|        35.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   EWR|635655159808| MSP|1589137899520|         0.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   EWR|635655159808| PIT|1468878815232|        38.0|            6.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   EWR|635655159808| PIT|1468878815232|         0.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   CLT|858993459201| EWR| 635655159808|        26.0|            2.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   CMH|326417514496| EWR| 635655159808|         0.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//|1543302000|               YX|1322849927168|   EWR|635655159808| PIT|1468878815232|        21.0|            0.0|      0.0|            null|[1.543302E9,1.322...|
//+----------+-----------------+-------------+------+------------+----+-------------+------------+---------------+---------+----------------+--------------------+

//mlInputTesting example rows
//+-----+--------------------+                                                    
//|label|            features|
//+-----+--------------------+
//|  0.0|[1.543302E9,1.322...|
//| 98.0|[1.543302E9,1.322...|
//|  0.0|[1.543302E9,1.322...|
//| 35.0|[1.543302E9,1.322...|
//|  0.0|[1.543302E9,1.322...|
//| 38.0|[1.543302E9,1.322...|
//|  0.0|[1.543302E9,1.322...|
//| 26.0|[1.543302E9,1.322...|
//|  0.0|[1.543302E9,1.322...|
//| 21.0|[1.543302E9,1.322...|
//+-----+--------------------+

// airline lookup table
//+-----------------+-------------+
//|Reporting_Airline|           id|
//+-----------------+-------------+
//|               UA|  42949672960|
//|               NK|  94489280512|
//|               AA| 128849018880|
//|               EV| 309237645312|
//|               B6| 360777252864|
//|               DL| 558345748480|
//|               OO| 695784701952|
//|               F9| 704374636544|
//|               YV| 755914244096|
//|               US| 790273982464|
//|               MQ| 867583393792|
//|               OH| 919123001344|
//|               HA|1022202216448|
//|               G4|1228360646656|
//|               YX|1322849927168|
//|               AS|1434519076864|
//|               FL|1503238553600|
//|               VX|1580547964928|
//|               WN|1649267441664|
//|               9E|1709396983808|
//+-----------------+-------------+
