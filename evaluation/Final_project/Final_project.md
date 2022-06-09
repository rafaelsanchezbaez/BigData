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

![](http://www.scielo.org.co/img/revistas/rcien/v18n2/v18n2a10-fig02.jpg)

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

![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_4/evaluation/Final_project/SVM1.png?raw=true)

Average accuracy over 30 SVM iterations

![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_4/evaluation/Final_project/SVM2.png?raw=true)

We introduce the algorithm inside a while so that it performs the iterations automatically, we also create a vector to store the precision in each iteration and at the end with the .sum function we add all the values of our array, then we divide the result of the sum between the total number of iterations, which in this case were 30, and gave us 89.11% accuracy as a result.

#### Logistic Regression Algorithm

``` scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val data  = spark.read.option("header","true").option("inferSchema","true").option("delimiter", ";").format("csv").load("bank-full.csv")

val label = new StringIndexer().setInputCol("y").setOutputCol("label")
val labeltransform = label.fit(data).transform(data)

val assembler = new VectorAssembler().setInputCols (Array ("balance", "day", "duration", "pdays", "previous")).setOutputCol("features")
val data2 = assembler.transform(labeltransform)
data2.show(1)

val training = data2.select("features", "label")
training.show(1)

val iterations=30
var z = new Array[Double](iterations)

var y=0

while(y < iterations){
val splits = training.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
println("training set = ", train.count())
println("test set = ", test.count())

val lr = new  LogisticRegression().setMaxIter(10).setRegParam(0.1)
val model = lr.fit(train)
val result = model.transform(test)
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Coefficients: ${model.coefficients}")
println(s"Intecept: ${model.intercept}")
println(s"Accuraccy = ${evaluator.evaluate(result)}")
z(y)=evaluator.evaluate(result)
y=y+1
}
val sum=z.sum
val mean = sum/iterations
var h=0
```
Logistic Regression 30 iterations

![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_4/evaluation/Final_project/LR1.png?raw=true)

Average accuracy over 30 iterations of Logistic Regression

![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_4/evaluation/Final_project/LR2.png?raw=true)

We introduce the algorithm inside a while so that it performs the iterations automatically, we also create a vector to store the precision in each iteration and at the end with the .sum function we add all the values of our array, then we divide the result of the sum between the total number of iterations, which in this case were 30, and gave us 88.52% accuracy as a result.

#### Multilayer Perceptron Algorithm

``` scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),0).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))

val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)

val cambio = fea.withColumnRenamed("y", "label")
val training= cambio.select("label","features")

val iterations=30
var z = new Array[Double](iterations)

var y=0

while(y < iterations){
val split = training.randomSplit(Array(0.7, 0.3))
val train = split(0)
val test = split(1)

val layers = Array[Int](5, 2, 2, 4)

val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

val model = trainer.fit(train)

val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy ${y+1} = ${evaluator.evaluate(predictionAndLabels)}")
z(y)=evaluator.evaluate(predictionAndLabels)
y=y+1
}
val sum=z.sum
val mean = sum/iterations
```
Multilayer Perceptron accuracy 30 iterations

![]()

Average precision in 30 iterations of Multilayer Perceptron

![]()

We introduce the algorithm inside a while so that it performs the iterations automatically, we also create a vector to store the precision in each iteration and at the end with the .sum function we add all the values of our array, then we divide the result of the sum between the total number of iterations, which in this case were 30, and gave us 88.26% accuracy as a result.
|Test accuracy|
|          | Test accuracy |                    |                    |                     |                       |
|----------|:-------------:|:------------------:|:------------------:|:-------------------:|:---------------------:|
|          |   Iteration  |    Decision Tree   |         SVM        | Logistic Regression | Multilayer Perceptron |
|          |       1       | 0.8909529730125064 | 0.8933736717827627 | 0.8791007615700058  | 0.8840047044986769    |
|          |       2       | 0.8884981684981685 | 0.8889454915944982 | 0.8781627958744528  | 0.8798080472499077    |
|          |       3       | 0.8939080289812213 | 0.8910635155096012 | 0.8871132516053707  | 0.881525226013415     |
|          |       4       | 0.8896365099100417 | 0.890473390179359  | 0.8832390329928723  | 0.879742145820984     |
|          |       5       | 0.8896398069067954 | 0.8932497233493176 | 0.8857609745227661  | 0.8822187890826161    |
|          |       6       | 0.8941167861846919 | 0.8897585468507716 | 0.8860232337254329  | 0.8824410333208537    |
|          |       7       | 0.8889460910494962 | 0.8870707367185207 | 0.8821465890431408  | 0.8831796223100571    |
|          |       8       | 0.8900214196026295 | 0.8880684310571721 | 0.8835001095610255  | 0.883184138492269     |
|          |       9       | 0.8917357819905213 | 0.8940402247855664 | 0.8874935362340253  | 0.8814034053217366    |
|          |       10      | 0.891198224852071  | 0.8910927865686056 | 0.8845107696831581  | 0.886097399248508     |
|          |       11      | 0.8862086663245106 | 0.8912531291415108 | 0.8859242925395076  | 0.8829974996322989    |
|          |       12      | 0.8904964747356052 | 0.8912350892790991 | 0.8873652562880399  | 0.8813484792964456    |
|          |       13      | 0.8889298892988929 | 0.8857901726427623 | 0.8852897473997028  | 0.8809347181008902    |
|          |       14      | 0.8958873656909967 | 0.8894420852247766 | 0.8841899315223117  | 0.8847409826332195    |
|          |       15      | 0.8900285735218697 | 0.8941767661838604 | 0.882773356911288   | 0.884856434922023     |
|          |       16      | 0.8908890521675239 | 0.8940436994911704 | 0.8861740606776408  | 0.8860042656468339    |
|          |       17      | 0.8892486570019869 | 0.8922986564863079 | 0.8852580764946156  | 0.8827704722056187    |
|          |       18      | 0.8906409694845963 | 0.8891997349628211 | 0.8887231503579952  | 0.8836021903211484    |
|          |       19      | 0.8924968706280834 | 0.8907079646017699 | 0.8840179997096821  | 0.8851824274013402    |
|          |       20      | 0.8906539974760597 | 0.892701048951049  | 0.8806575261683621  | 0.8859324043877853    |
|          |       21      | 0.8830755568711611 | 0.8930697435143676 | 0.884981684981685   | 0.882126462313046     |
|          |       22      | 0.8915752002373183 | 0.8896967934268887 | 0.8849591033020296  | 0.886687623364281     |
|          |       23      | 0.8902978472427012 | 0.8929778169531365 | 0.881508875739645   | 0.8814895947426068    |
|          |       24      | 0.8970631209818819 | 0.8889712971890529 | 0.8855783308931185  | 0.882570435554241     |
|          |       25      | 0.8883241252302025 | 0.8893424372866261 | 0.8871922262000438  | 0.8862422083704363    |
|          |       26      | 0.8884788545280095 | 0.8913390396053967 | 0.8851906158357771  | 0.8798298995527531    |
|          |       27      | 0.888650011120172  | 0.8906469500924215 | 0.8837329018973379  | 0.8821391484942886    |
|          |       28      | 0.8865527488855869 | 0.8914780292942743 | 0.8832132722113075  | 0.8805290227773696    |
|          |       29      | 0.8899970665884424 | 0.8968084322252524 | 0.8823311336842886  | 0.8792402630995492    |
|          |       30      | 0.8932489140837812 | 0.8931090214516603 | 0.8831273904089438  | 0.8776200135226504    |
| Average |               | 0.8903799251029177 | 0.8911808142133462 | 0.8843079996011859  | 0.8826816352565953    |
|          |               | 89.03%             | 89.11%             | 88.43%              | 88.26%                |

