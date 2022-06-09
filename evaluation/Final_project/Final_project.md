![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTyEeQKr8i8LV46EPfadUhj83K8z9xG37VHYA&usqp=CAU)
##### INSTITUTO TECNOLÓGICO DE TIJUANA
##### SUBDIRECCIÓN ACADÉMICA
##### DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN
##### SEMESTRE
Enero – Julio 2022
##### CARRERA
Ing. Tecnologías de la información y comunicación.
##### MATERIA Y SERIE
Datos masivos.
##### TÍTULO
Proyecto final.
##### UNIDAD A EVALUAR
Cuarta unidad
##### NOMBRE Y NÚMERO DE CONTROL DEL ALUMNO
Rafael Sánchez Báez 1621254
ALONSO VILLEGAS LUIS ANTONIO 18212139

## Index

## Introduction
## Theoretical framework of algorithms
#### Support vector machine (SVM)
Support vector machine (SVM) is a supervised learning algorithm used in many classification and regression problems, including medical signal processing, natural language processing, and image and speech recognition applications.
The goal of the SVM algorithm is to find a hyperplane that best separates two different classes of data points. "As best as possible" implies the hyperplane with the widest margin between the two classes, represented by the plus and minus signs in the figure below. The margin is defined as the maximum width of the region parallel to the hyperplane that has no interior data points. The algorithm can only find this hyperplane in problems that allow linear separation; in most practical problems, the algorithm maximizes the soft margin by allowing a small number of misclassifications.[3]
![](https://miro.medium.com/max/600/0*9jEWNXTAao7phK-5.png)
![](https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png)

 Figure 1 Possible hyperplanes[7] Figure 2
 #### Decision tree
 A decision tree is a graphical and analytical way of representing all the events (happenings) that can arise from a decision made at a certain moment. They help us make the most "right" decision, from a probabilistic point of view, given a range of possible decisions. These trees allow you to examine the results and visually determine how the model flows. Visual results help look for specific subgroups and relationships that we might not find with more traditional statisticians.
Decision trees are a statistical technique for segmentation, stratification, prediction, data reduction and variable filtering, interaction identification, category merging, and discretization of continuous variables[5].
![](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2020/05/Parts-of-a-Decision-Tree.jpg)

#### Logistic regression.
Logistic regression is a statistical instrument for multivariate analysis, for both explanatory and predictive use. Its use is useful when there is a dichotomous dependent variable (an attribute whose presence or presence we have scored with the values ​​zero and one, respectively) and a set of predictor or independent variables, which can be quantitative (called covariates or covariates) or categorical. In the latter case, it is required that they be transformed into “dummy” variables, that is, simulated variables.
The purpose of the analysis is to: predict the probability that a certain “event” will happen to someone: for example, being unemployed =1 or not being unemployed = 0, being poor = 1 or not poor = 0, graduating from a sociology degree =1 or not received = 0).
Determine which variables weigh more to increase or decrease the probability that the event in question will happen to someone

This assignment of the probability of occurrence of the event to a certain subject, as well as the determination of the weight that each of the dependent variables in this probability, are based on the characteristics of the subjects to whom, effectively, these occur or not. events.
For example, logistic regression will take into account the values ​​assumed in a series of variables (age, sex, educational level, position in the household, migratory origin, etc.) by the subjects who are effectively unemployed (=1) and those who they are not (=0). Based on this, it will predict for each of the subjects – regardless of their real and current status – a certain probability of being unemployed (that is, of having a value of 1 in the dependent variable). Let's say, if someone is a young non-head of household, with low education and of male sex and migrant origin (although he is employed), the model will predict a high probability of being unemployed (since the unemployment rate of the group thus defined is high), generating a variable with those estimated probabilities. And it will proceed to classify it as unoccupied in a new variable, which will be the result of the prediction[6].
#### Multilayer perceptron
The Perceptron Multilayer model is made up of an input layer, hidden layers and an output layer (Figure 4) which are made up of a series of neurons that are responsible for receiving, processing and sending data to other neurons, processing the information through different mathematical functions.

![](https://www.researchgate.net/profile/V-Botti/publication/228815505/figure/fig1/AS:669385609994246@1536605374299/Figura-3-Ejemplo-de-perceptron-multicapa.png)

![](https://www.researchgate.net/profile/Henry-Paz/publication/281380920/figure/fig2/AS:391418657951750@1470332888950/Estructura-de-un-Perceptron-multicapa-Este-modelo-se-compone-de-la-siguiente-manera-o.png)


Figure 4[9]. Architecture of a Perceptron Multilayer Neural Network
The neurons of the input layer receive the digital levels that an image pixel presents in its different multi-spectral bands, therefore, there will be a direct relationship between the number of neurons of the input layer and the number of bands of the input layer. image to classify. For their part, the hidden layers are responsible for representing the level of complexity that may exist in the relationship between the input layer and the output layer. The usual number of hidden layers is between one and two, so that it is possible to solve the complex separability of the covers. Finally, the output layer is responsible for producing the classification result of the neural network; For this reason, the number of neurons that make up this layer is directly related to the number of coverage classes to be identified.

## Implementation
In order to compare these four algorithms, Apache Spark was used with the Scala language, Scala is a modern multi-paradigm programming language designed to express common programming patterns in a concise, elegant, and type-safe way. It easily integrates features of object-oriented and functional languages[2].
apache spark
Apache Spark is today one of the most influential and important technologies in the world of Big Data. It is an open cluster computational system, unified analysis engine, ultrafast for Big Data and Machine Learning[1].
In short, Spark is a general-purpose data processing engine, a set of tools with APIs that data scientists and application developers embed in their applications to quickly query, analyze, and transform data. It is capable of handling several petabytes of data at a time, distributed across a group of thousands of cooperating physical or virtual servers[4].
##Results
#### Algorithm Decision Tree
``` scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer,VectorAssembler}
import org.apache.log4j._
import org.apache.spark.sql.SparkSession

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
data.printSchema()
data.show()


val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)

val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
val features = assembler.transform(data)


val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)


val iterations=30
var z = new Array[Double](iterations)

var y=0

while(y < iterations){
val Array(trainingData, testData) = features.randomSplit(Array(0.7, 0.3))

val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test accuracy ${y+1} = ${(accuracy)}")
z(y)=accuracy
y=y+1
}
val sum=z.sum
val mean = sum/iterations
```
Accuracy 30 Iterations of the Decision Tree algorithm

![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_4/evaluation/Final_project/Decision_Tree1.png?raw=true)

Average over 30 Decision Tree iterations

![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_4/evaluation/Final_project/Decision_Tree2.png?raw=true)

We introduce the algorithm inside a while so that it performs the iterations automatically, we also create a vector to store the precision in each iteration and at the end with the .sum function we add all the values of our array, then we divide the result of the sum between the total number of iterations, which in this case were 30, and gave us 89.03% accuracy as a result.

SVM algorithm

``` scala
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.StandardScaler

var data = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter",";").csv("bank-full.csv").cache()

val cols = Array("age", "feature_job", "feature_marital", "feature_education", "feature_default", "balance", "feature_loan","day", "feature_month", "duration", "campaign", "pdays", "previous", "feature_poutcome")

val discrete_cols = Array("job","marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y")

val indexer = new StringIndexer()
discrete_cols.foreach{
  case(i) => {
    indexer.setInputCol(i)
    indexer.setOutputCol(s"feature_$i")
    data = indexer.fit(data).transform(data)
    data = data.drop(i)
  }
}

val vecass = new VectorAssembler()
vecass.setInputCols(cols)
vecass.setOutputCol("features")

val label_x_features = vecass.transform(data).select("feature_y", "features")

val scaler = new StandardScaler()
scaler.setInputCol("features")
scaler.setOutputCol("scaled_features")

val scaled = scaler.fit(label_x_features).transform(label_x_features).select("feature_y", "scaled_features")

val libsvm = scaled.rdd.map( row => LabeledPoint(row.getAs[Double](0),Vectors.fromML(row.getAs[DenseVector](1))))

val iterations=30
var z = new Array[Double](iterations)

var y=0
while(y < iterations){
val splits = libsvm.randomSplit(Array(0.7, 0.3))
val training = splits(0).cache()
val test = splits(1).cache()

val model = SVMWithSGD.train(training, 100)

val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
val prediction = model.predict(features)
(prediction, label)
}

val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println(s"Test accuracy $y: $accuracy")
z(y)=accuracy
y=y+1
}
val sum=z.sum
val mean = sum/iterations

```
Accuracy 30 SVM iterations



``` scala

```

![]()