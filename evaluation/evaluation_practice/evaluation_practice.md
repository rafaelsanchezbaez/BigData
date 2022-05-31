# Evaluative practice
### 1. Load in an Iris.csv dataframe found at https://github.com/jcromerohdz/iris, prepare the necessary data cleaning to be processed by the following algorithm (Important, this cleaning must be done through a Scala script in Spark).
a. Use Spark's Mllib library the Machine Learning multilayer perceptron algorithm
  
```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

val data  = spark.read.option("header","true").format("csv").load("Iris.csv")
val df = data.withColumn("sepal_length",$"sepal_length".cast("double")).withColumn("sepal_width",$"sepal_width".cast("double")).withColumn("petal_length", $"petal_length".cast("double")).withColumn("petal_width", $"petal_width".cast("double"))
```
  
### 2. What are the column names?  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic1.jpg) 
  
### 3. How is the scheme?  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic2.jpg) 
  
### 4. Print the first 5 columns.  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic3.jpg) 
  
### 5. Use the describe() method to learn more about the data in the DataFrame.  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic4.jpg) 
  
### 6. Make the pertinent transformation for the categorical data which will be our labels to classify.
```scala
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")

val features = assembler.transform(df)

val indexerLabel = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(features)

val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

val splits = features.randomSplit(Array(0.7, 0.3),seed = 1234)

val training = splits(0)

val test = splits(1)

val layers = Array[Int](4, 5, 4, 3)
```
  
### 7. Build the classification model and explain its architecture.
```scala
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)

val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)

val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))

val model = pipeline.fit(training)
```
  
### 8. Print the model results
```scala
val predictions = model.transform(test)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)

println("Error = " + (1.0 - accuracy))
```  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic5.jpg)  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic6.jpg) 
   
### Conclusion
This type of algorithms help us to make predictions, based on the data that we have previously of some situation, as in this practice that tries to predict what species each flower is according to the petal and sepal in terms of their size. . Basically, what it does is an algorithm that takes the data set as a basis to create the model and once it is done, it tests it to see if it was right and the precision tells us how much this model succeeds or fails. These models can help us, depending on the case, to make decisions that can prevent losses for a company or profits, as the case may be.
