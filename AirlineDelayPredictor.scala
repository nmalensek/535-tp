import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

object AirlineDelayPredictor {
    def main(args: Array[String]) {
        if (args.length != 2) {
            println("Incorrect number of arguments. Usage:")
            println("<input dir> <output dir> <local | yarn>")
            return
        }

        //pre-processed files that contains feature column
        val savedModelRootPath = args(0)
        val specifiedMaster = args(1)

        val spark = SparkSession.builder.appName("TeamDenverFlights").master(specifiedMaster).getOrCreate()

        val transformer = new VectorAssembler().setInputCols(Array("FlightDate","AirlineId","OriginId","DestId")).setOutputCol("features")

        case class FlightInput(date: String, origin: String, destination: String, AirlineId: Long, delay: Double)
        case class ModelResults(prediction: Double, AirlineId: Long, AirlineChanceOfDelay: Double)

        val airlineLookup = spark.read.load("/535/tp/airline_codes")
        val airportLookup = spark.read.load("/535/tp/airport_codes")

        val airline1ProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/airlineModel")
        val model1 = LinearRegressionModel.load(savedModelRootPath + "/model1")
        val model2 = LinearRegressionModel.load(savedModelRootPath + "/model2")
        val modeln = LinearRegressionModel.load(savedModelRootPath + "/modeln")
        
        while(true) {
            val input = scala.io.StdIn.readLine("Please enter your desired flight date (YYYY-MM-dd format), origin airport, and destination airport:\n")

            val preparedInput = convertInput(input, airportLookup, airlineLookup)

            val modelInput = transformer.transform(preparedInput)

            val model1Prediction = model1.transform(modelInput.where($"AirlineId" === 42949672960L).select("label", "features"))
            //val airline1DelayProbability = airline1ProbabilityModel.transform(modelInput.where($"AirlineId" === 42949672960L).select("label", "features"))
            val model1Results = ModelResults(model1Prediction.first().getDouble(2), 42949672960L, airline1DelayProbability.first().getDouble(2))

            //...do for remaining models...

            val allResults = Seq(model1Results).toDF //model2results, ... etc.
            val resultsWithAirlines = allResults.join(airlineLookup, allResults("AirlineId") === airlineLookup("id")).select(allResults("prediction"), airlineLookup("Reporting_Airline"))
            
            resultsWithAirlines.withColumnRenamed("prediction", "PredictedDelayAmount").orderBy("PredictedDelayAmount").show(10)
        }
    }

    def convertInput(input: String, airports: DataFrame, airlines: DataFrame) : DataFrame = {
        val splitInput = input.split(" ")
        val flightInputDF = Seq(
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 42949672960L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 94489280512L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 128849018880L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 309237645312L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 360777252864L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 558345748480L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 695784701952L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 704374636544L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 755914244096L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 790273982464L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 867583393792L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 919123001344L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1022202216448L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1228360646656L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1322849927168L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1434519076864L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1503238553600L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1580547964928L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1649267441664L, 999.99),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, 1709396983808L, 999.99)
        ).toDF
        val castDF = flightInputDF.withColumn("date", 'date cast "date")
        
        val dateDF = castDF.withColumn("FlightDate", unix_timestamp(castDF("date")))
        
        val originDF = dateDF.join(airports, dateDF("origin") === airports("Origin"))
        .select(dateDF("*"), airports("id")).withColumnRenamed("id", "OriginId")
        
        val transformedDF = originDF.join(airports, originDF("destination") === airports("Origin"))
        .select(originDF("*"), airports("id")).withColumnRenamed("id", "DestId")
        .withColumnRenamed("delay", "label")
        return transformedDF
    }
}