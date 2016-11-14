package com.machine.learning

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * Created by M.Kumar on 11/14/2016.
  */
object twitterStreaming extends App{

  // println("Hello World")

  val spark = SparkSession
    .builder
    .appName("Recommendation Engine")
    .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
    .master("local[*]")
    .getOrCreate()
  println(s">>>Spark version : ${spark.version}")

  val sentenceData = spark.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (0, "I wish Java could use case classes"),
    (1, "Logistic regression models are neat")
  )).toDF("label", "sentence")

  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  val wordsData = tokenizer.transform(sentenceData)
  val hashingTF = new HashingTF()
    .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
  val featurizedData = hashingTF.transform(wordsData)
  // alternatively, CountVectorizer can also be used to get term frequency vectors

  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)
  val rescaledData = idfModel.transform(featurizedData)
  rescaledData.select("features", "label").take(3).foreach(println)

}
