
# First we shall Import the Required Classes for ML

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Next We will Create Spark Session , This is the Entry point of any Spark Functionality.

spark = SparkSession.builder.appName("California Housing Regression").getOrCreate()
print("Spark Session Created Succesfully")


# Next we shall Load the Housing Data from Git Repository , We willl infer the Schema Automatically and assume first
# first is a header

data_url = "https://raw.githubusercontent.com/RajanChettri/sparkML/refs/heads/main/housing.csv"


try:
        data = spark.read.csv( "/user/hadoop/housing.csv" , header=True , inferSchema=True)
        print("Dataset Loaded Succesfully")
        print("Schema:")
        data.printSchema()
        print("First 5 Rows: ")
        data.show(5)


        # drop rows with missing values

        data = data.dropna()

        # The Goal of this App is to Predict the Median House Value


        feature_cols = data.columns
        feature_cols.remove("median_house_value")
        feature_cols.remove("ocean_proximity")

        # Next Use VectorAssembler to Combine the Columns.

        assembler = VectorAssembler( inputCols= feature_cols , outputCol ="features" )

        # Next Transform the DataFrame with the new 'features' column.
        assembled_data = assembler.transform( data )

        # for training and testing we shall use 80 / 20 split.

        (training_data , test_data) = assembled_data.randomSplit( [0.8 ,0.2] , seed=42 )

        # for prediction we will create and train the Linear Regression Model

        lr = LinearRegression( featuresCol="features",labelCol="median_house_value")

        lr_model = lr.fit( training_data )

        # Next Make the Prediction on the Test Data

        predictions = lr_model.transform( test_data )

        # Print Prediction vs Actuals")
        predictions.select("median_house_value","prediction").show(5,truncate=False)


except Exception as e :
        print(f"Error Loading data from URL: {e}")
