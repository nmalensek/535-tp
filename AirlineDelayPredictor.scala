import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.unix_timestamp
import org.apache.spark.sql.functions.lit
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame

object Predictor {
        val ENDEAVOR = 1709396983808L
        val AMERICAN = 128849018880L
        val UNITED = 42949672960L
        val SPIRIT = 94489280512L
        val EXPRESSJET = 309237645312L
        val JETBLUE = 360777252864L
        val DELTA = 558345748480L
        val SKYWEST = 695784701952L
        val FRONTIER = 704374636544L
        val MESA = 755914244096L
        val ENVOY = 867583393792L
        val HAWAIIAN = 1022202216448L
        val ALASKA = 1434519076864L
        val VIRGIN = 1580547964928L
        val SOUTHWEST = 1649267441664L
        val NONE = 9999999999999L

        val spark = SparkSession.builder.appName("TeamDenverFlights").master("local").getOrCreate()
        import spark.implicits._

        case class FlightInput(date: String, origin: String, destination: String, AirlineId: Long, delay: Double, isDelayed: Double)
        case class ModelResults(prediction: Double, AirlineId: Long, AirlineChanceOfDelay: Double)
    
    def main(args: Array[String]) {
        if (args.length != 1) {
            println("Incorrect number of arguments. Usage:")
            println("<input dir>")
            return
        }

        //pre-processed files that contains feature column
        val savedModelRootPath = args(0)

        val transformer = new VectorAssembler().setInputCols(Array("FlightDate","AirlineId","OriginId","DestId")).setOutputCol("features")

        val airlineLookup = spark.read.load("/535/tp/airline_codes")
        val airportLookup = spark.read.load("/535/tp/airport_codes")

        airlineLookup.cache()
        airportLookup.cache()

        val unitedDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel42949672960")
        val spiritDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel94489280512")
        val americanDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel128849018880")
        val expressDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel309237645312")
        val jetBlueDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel360777252864")
        val deltaDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel558345748480")
        val skywestDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel695784701952")
        val frontierDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel704374636544")
        val mesaDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel755914244096")
        val envoyDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel867583393792")
        val hawaiianDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel1022202216448")
        val alaskaDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel1434519076864")
        val virginDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel1580547964928")
        val southwestDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel1649267441664")
        val endeavorDelayAmount = LinearRegressionModel.load(savedModelRootPath + "/amountModel1709396983808")

        val unitedProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel42949672960")
        val spiritProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel94489280512")
        val americanProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel128849018880")
        val expressProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel309237645312")
        val jetBlueProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel360777252864")
        val deltaProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel558345748480")
        val skywestProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel695784701952")
        val frontierProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel704374636544")
        val mesaProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel755914244096")
        val envoyProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel867583393792")
        val hawaiianProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel1022202216448")
        val alaskaProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel1434519076864")
        val virginProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel1580547964928")
        val southwestProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel1649267441664")
        val endeavorProbabilityModel = LinearRegressionModel.load(savedModelRootPath + "/probabilityModel1709396983808")

        val nonAirlineDelayModel = LinearRegressionModel.load(savedModelRootPath + "/nonAirlineDelays")
        //val cancellationModel = LinearRegressionModel.load(savedModelRootPath + "/cancellations")
        
        while(true) {
            val input = scala.io.StdIn.readLine("Please enter your desired flight date (YYYY-MM-dd format), origin airport, and destination airport:\n")
            println("Searching for flights matching the input " + input)
            val preparedInput = convertInput(input, airportLookup, airlineLookup)

            val modelInput = transformer.transform(preparedInput)

            val endeavorResults = giveInputToModel(endeavorDelayAmount, endeavorProbabilityModel, ENDEAVOR, modelInput)
            val americanResults = giveInputToModel(americanDelayAmount, americanProbabilityModel, AMERICAN, modelInput)
            val unitedResults = giveInputToModel(unitedDelayAmount, unitedProbabilityModel, UNITED, modelInput)
            val spiritResults = giveInputToModel(spiritDelayAmount, spiritProbabilityModel, SPIRIT, modelInput)
            val expressResults = giveInputToModel(expressDelayAmount, expressProbabilityModel, EXPRESSJET, modelInput)
            val jetBlueResults = giveInputToModel(jetBlueDelayAmount, jetBlueProbabilityModel, JETBLUE, modelInput)
            val deltaResults = giveInputToModel(deltaDelayAmount, deltaProbabilityModel, DELTA, modelInput)
            val skywestResults = giveInputToModel(skywestDelayAmount, skywestProbabilityModel, SKYWEST, modelInput)
            val frontierResults = giveInputToModel(frontierDelayAmount, frontierProbabilityModel, FRONTIER, modelInput)
            val mesaResults = giveInputToModel(mesaDelayAmount, mesaProbabilityModel, MESA, modelInput)
            val envoyResults = giveInputToModel(envoyDelayAmount, envoyProbabilityModel, ENVOY, modelInput)
            val hawaiianResults = giveInputToModel(hawaiianDelayAmount, hawaiianProbabilityModel, HAWAIIAN, modelInput)
            val alaskaResults = giveInputToModel(alaskaDelayAmount, alaskaProbabilityModel, ALASKA, modelInput)
            val virginResults = giveInputToModel(virginDelayAmount, virginProbabilityModel, VIRGIN, modelInput)
            val southwestResults = giveInputToModel(southwestDelayAmount, southwestProbabilityModel, SOUTHWEST, modelInput)

            val nonAirlineDelayResults = giveInputToModel(nonAirlineDelayModel, alaskaProbabilityModel, NONE, modelInput)

            val allResults = Seq(endeavorResults, americanResults, unitedResults, spiritResults, expressResults, jetBlueResults, deltaResults, 
            skywestResults, frontierResults, mesaResults, envoyResults, hawaiianResults, alaskaResults, virginResults, southwestResults, nonAirlineDelayResults).toDF

            val resultsWithAirlines = allResults.join(airlineLookup, allResults("AirlineId") === airlineLookup("id"), "left_outer").select(allResults("*"), airlineLookup("Reporting_Airline"))
            
            resultsWithAirlines.withColumnRenamed("prediction", "PredictedDelayAmount").orderBy("PredictedDelayAmount").show(20)
        }
    }

    def convertInput(input: String, airports: DataFrame, airlines: DataFrame) : DataFrame = {
        val splitInput = input.split(" ")
        val flightInputDF = Seq(
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, UNITED, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, SPIRIT, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, AMERICAN, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, EXPRESSJET, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, JETBLUE, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, DELTA, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, SKYWEST, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, FRONTIER, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, MESA, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, ENVOY, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, HAWAIIAN, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, ALASKA, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, VIRGIN, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, SOUTHWEST, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, ENDEAVOR, 999.99, 0.0),
            FlightInput(splitInput(0), splitInput(1).toUpperCase, splitInput(2).toUpperCase, NONE, 999.99, 0.0)
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

    def giveInputToModel(amountModel: LinearRegressionModel, probabilityModel: LinearRegressionModel, airline: Long, input: DataFrame) : ModelResults = {
            //predict the amount of delay
            val delayPrediction = amountModel.transform(input.where($"AirlineId" === airline).select("label", "features"))
            
            //predict likelihood of delay
            val probabilityPrediction = probabilityModel.transform(input.where($"AirlineId" === airline).withColumnRenamed("label", "Delay")
                .withColumn("label", lit(0.0)).select("label", "features")).withColumnRenamed("prediction", "prob_prediction")

            //combine delay amount and likelihood results into one dataframe, eventually join all airlines' results and display them to user in order
            val delayValue = if (delayPrediction.count > 0) delayPrediction.first().getDouble(2) else 100.0
            val probValue = if(probabilityPrediction.count > 0) probabilityPrediction.first().getDouble(2) else 100.0
            val results = ModelResults(delayValue, airline, probValue)

            return results
    }
}