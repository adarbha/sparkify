
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import Window

from pyspark.sql.functions import count, col, to_date, udf, sum
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


if __name__ == '__main__':

    spark = SparkSession \
            .builder \
            .appName('Sparkify_ml_model') \
            .getOrCreate()


    '''Load and Clean Dataset'''
    event_data = "s3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json"
    df = spark.read.json(event_data)

    # Remove nas in userId and sessionId anyway
    df = df.where(col('userId').isNotNull() | col('sessionId').isNotNull())

    '''Add a churn column to the dataframe for the churned (canceled) users'''
    add_churn = udf(lambda x : 1 if x is not None else 0, IntegerType())

    df_canceled_users = df.filter("page = 'Cancel'")\
                      .select('userId')\
                      .distinct()\
                      .withColumn('churn', add_churn(col('userId')))


    # Join df_canceled_users with df
    df = df.join(df_canceled_users, on = ['userId'], how = 'left')

    # Fill null in churn column with 0s
    df = df.fillna({'churn': 0})

    '''Feature engineering'''

     # Paid free users
    add_level = udf(lambda x: 1 if x == "free" else 0, IntegerType())
    df = df.withColumn("level_num", add_level(df.level))

    # Ads served
    add_ad_served = udf(lambda x: 1 if x == "Roll Advert" else 0, IntegerType())
    df = df.withColumn("ad_served", add_ad_served(df.page))

    # Number of songs played
    add_next_song = udf(lambda x: 1 if x == "NextSong" else 0, IntegerType())
    df = df.withColumn("song", add_next_song(df.page))

    # Songs added to playlist
    add_to_playlist = udf(lambda x: 1 if x == "Add to Playlist" else 0, IntegerType())
    df = df.withColumn("to_playlist", add_to_playlist(df.page))

    # Adding friends
    add_friend = udf(lambda x: 1 if x == "Add Friend" else 0, IntegerType())
    df = df.withColumn("friend", add_friend(df.page))

    # Accessing help and settings
    add_help = udf(lambda x: 1 if x in ["Help", "Settings"] else 0, IntegerType())
    df = df.withColumn("help", add_help(df.page))

    # Thumbs ratio
    # Add thums_up
    add_tu = udf(lambda x:1 if x == "Thumbs Up" else 0, IntegerType())
    df = df.withColumn("tu", add_tu(df.page))

    # Add thumbs down
    add_td = udf(lambda x: 1 if x == "Thumbs Down" else 0, IntegerType())
    df = df.withColumn("td", add_td(df.page))

    df = df.withColumn("thumbs_ratio", df.tu / df.td)

    '''Aggregate the data into a master_df with features aggregated'''

    master_df = df.groupBy(['userId','churn','gender'])\
                    .agg({'ad_served':'sum',
                          'length':'sum',
                          'artist':'count',
                          'to_playlist':'sum',
                          'friend':'sum',
                          'help':'sum',
                          'tu':'sum',
                          'td':'sum',
                          'level':'avg'
                        }).withColumnRenamed('sum(length)', 'sum_length')\
                          .withColumnRenamed('count(artist)', 'sum_song')\
                          .withColumnRenamed('sum(to_playlist)', 'sum_playlist')\
                          .withColumnRenamed('sum(friend)','sum_friend')\
                          .withColumnRenamed('sum(help)','sum_help')\
                          .withColumnRenamed('sum(tu)', 'sum_tu')\
                          .withColumnRenamed('sum(td)', 'sum_td')\
                          .withColumnRenamed('sum(ad_served)', 'sum_ad_served')\
                          .withColumnRenamed('avg(level)','avg_level')

    # Need to convert gender to numeric (1 = M and 0 = F)
    convert_gender = udf(lambda x: 1 if x == "M" else 0, IntegerType())

    # Need to create a tu/td ratio. Call this thumbs_ratio
    master_df = master_df.withColumn('gender_num', convert_gender(master_df.gender))\
                         .withColumn('thumbs_ratio', master_df.sum_tu / master_df.sum_td)
    
    # Avg_level has NAs and need to be filled in with 0s
    master_df = master_df.fillna({'avg_level':0.0})

    # Switch churn to label - this allows classifier to pick the right column up for the classification task
    master_df = master_df.withColumnRenamed('churn', 'label')

    
    # Prepare the data for creating a pipeline for the model
    assembler = VectorAssembler(inputCols = ["sum_song", "sum_friend", "sum_help", "sum_ad_served", "sum_length", "sum_playlist",
                                        "thumbs_ratio","gender_num", "avg_level"],
                            outputCol = "features", handleInvalid="skip")

    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

    '''Modeling'''

    # Split the dataset
    rest, validate = master_df.randomSplit([0.8, 0.2], seed = 42)

    # Using Gradient Boosted Tree Classifier
    gdbt = GBTClassifier(featuresCol="scaledFeatures")
    pipeline = Pipeline(stages = [assembler, scaler, gdbt])
    model = pipeline.fit(rest)

    # Run the model on validation set
    result = model.transform(validate)

    # Model Evaluation
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
    f1_score = evaluator.evaluate(result, {evaluator.metricName: "f1"})
    print("f1: {}".format(f1_score))


    # Write results
    with open("results.txt", 'w') as f:
        f.write("GDBT-f1 :{}".format(f1_score))



    ## Stop spark session
    spark.stop()
