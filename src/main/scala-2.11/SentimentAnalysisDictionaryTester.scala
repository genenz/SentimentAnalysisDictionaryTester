import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable.HashMap

/**
  * Created by gene on 3/1/16.
  */
object SentimentAnalysisDictionaryTester {
  def main(args: Array[String]): Unit= {
    val conf = new SparkConf().setMaster("local[*]").setAppName("SentimentAnalysisDictionary")
    val sc = new SparkContext(conf)
    val sqlc = new SQLContext(sc)

    // Bring in the Sentiment Dictionary.
    // Use spark to bring in the dictionary and manipulate it.
    // Then turn it into a HashMap
    val dictSourceFile = getClass.getResource("/AFINN-111.txt")
    println(dictSourceFile.toString)
    val sentimentDictionary = sc.textFile(dictSourceFile.toString, 1)
    val sentDictPairedRDD = sentimentDictionary.map(line => {
      val splitLine = line.split("\t")
      (splitLine(0), splitLine(1).toLong)
    })
    val dictArray = sentDictPairedRDD.collect()
    var dictionaryMap = new HashMap[String,Long]()

    // Populate the Dictionary
    dictArray.foreach(item => dictionaryMap += item)

    // Bring in the training set that we'll be working with
    val sentimentSource = getClass.getResource("/training_edited.txt")
    val sentimentSourceDF = sqlc.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(sentimentSource.toString)
    val textSentimentDF = sentimentSourceDF.filter("id IS NOT NULL").dropDuplicates(Array("text")).select("text", "sentiment")
    textSentimentDF.cache()

    val sentimentMatches = textSentimentDF.map(row => {
      var sentimentCount = 0L
      var sentiment = 0L
      //val words = row(0).toString().split(" ")
      val words = tokenize(row(0).toString)
      for (word <- words){
        if (dictionaryMap.contains(word)){
          sentimentCount = sentimentCount + dictionaryMap(word)
          if (sentimentCount > 0)
            sentiment = 1
          else if (sentimentCount < 0)
            sentiment = -1
          else
            sentiment = 0
        }
      }
      (sentiment,row(1).toString.toLong)
    })

    println("Model Accuracy: " + 100 * sentimentMatches.filter(x => x._1 == x._2).count / sentimentMatches.count + "%")

  }

  def tokenize(line: String): Array[String] = {
    line.toLowerCase.replaceAll("""[\p{Punct}]""", " ").replaceAll(" +", " ").trim.split(" ")
  }
}
