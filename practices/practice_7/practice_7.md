# Practice 7 - Naive Bayes  
### What is Naive Bayes?
It is a probabilistic classifier based on Bayes' theorem and some additional simplifying assumptions.
- The model is called naïve because it treats all the proposed prediction variables as independent of each other. Naive Bayesian is a fast and scalable algorithm that computes conditional probabilities for combinations of attributes and the target attribute.
  
### Steps to carry out the algorithm
- The data set in a frequency table.
- Create a probability table calculating those corresponding to the occurrence of the various events.
- The Naive Bayes equation is used to calculate the posterior probability of each class.
- The class with the highest posterior probability is the result of the prediction.
  
### Algorithm formula with the meaning of its variables
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_7/pic1.jpg) 
  
### Strengths
- An easy and fast way to predict classes, for binary and multiclass classification problems.
- The algorithm performs better than other classification models, even with less training data.
- The decoupling of the class conditional feature distributions means that each distribution can be estimated independently as if it had only one dimension.
  
### Weak points
- Naive Bayes algorithms are known to be poor estimators. Therefore, the odds obtained should not be taken too seriously.
- The Naive assumption of independence will most likely not reflect what the data is like in the real world.
- When the test data set has a feature that has not been observed in the training set, the model will assign it a probability of zero and it will be useless to make predictions.
  
### Example
```scala
//Import the necessary libraries

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

//Load the data by specifying the file path

val data = spark.read.format("libsvm").load("C:/spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

println ("Numero de lineas en el archivo de datos:" + data.count ())

//Show the first 20 lines by default

data.show()

//Randomly partition the data set into training set and test set according to the given weights. You can also specify a seed

val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3), 100L)
// The result is the type of the array, and the array stores the data of type DataSet

//Append to training set (fit operation) to train a Bayesian model
val naiveBayesModel = new NaiveBayes().fit(trainingData)

//The model calls transform() to make predictions and generate a new DataFrame.

val predictions = naiveBayesModel.transform(testData)

//Output of prediction results data
predictions.show()

//Evaluation of model accuracy

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
// precision
val precision = evaluator.evaluate (predictions) 

//print error rate
println ("tasa de error =" + (1-precision))
```
  
Result  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_7/pic2.jpg) 
