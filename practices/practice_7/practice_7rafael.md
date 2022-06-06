# Naive Bayes
#### What is Naive Bayes?
It is a probabilistic classifier based on Bayes' theorem and some additional simplifying assumptions.

- The model is called naïve because it treats all the proposed predictor variables as independent of each other. Naive Bayesian is a fast and scalable algorithm that computes conditional probabilities for combinations of attributes and the target attribute.
#### Steps to carry out the algorithm
- The data set in a frequency table.
- Create a probability table calculating those corresponding to the occurrence of the various events.
- The Naive Bayes equation is used to calculate the posterior probability of each class.
- The class with the highest posterior probability is the outcome of the prediction.
#### Algorithm formula with the meaning of its variables
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_7/pic1.jpg?raw=true)

#### Strengths
- A quick and easy way to predict classes, for binary and multiclass classification problems.
- The algorithm performs better than other classification models, even with less training data.
- The decoupling of the class conditional feature distributions means that each distribution can be estimated independently as if it had only one dimension.
#### Weak points
- Naive Bayes algorithms are known to be poor estimators. Therefore, the odds that are obtained should not be taken very seriously.
- The Naive assumption of independence will most likely not reflect what the data is like in the real world.
- When the test data set has a feature that has not been observed in the training set, the model will assign it a probability of zero and predictions will be useless.
#### Example
``` scala
//Importar las librerias necesarias

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

//Cargar los datos especificando la ruta del archivo

val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")

println ("Numero de lineas en el archivo de datos:" + data.count ())

//Mostrar las primeras 20 líneas por defecto

data.show()

//Divida aleatoriamente el conjunto de datos en conjunto de entrenamiento y conjunto de prueba de acuerdo con los pesos proporcionados. También puede especificar una seed

val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3), 100L)
// El resultado es el tipo de la matriz, y la matriz almacena los datos de tipo DataSet

//Incorporar al conjunto de entrenamiento (operación de ajuste) para entrenar un modelo bayesiano
val naiveBayesModel = new NaiveBayes().fit(trainingData)

//El modelo llama a transform() para hacer predicciones y generar un nuevo DataFrame.

val predictions = naiveBayesModel.transform(testData)

//Salida de datos de resultados de predicción
predictions.show()

//Evaluación de la precisión del modelo

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
// Precisión
val precision = evaluator.evaluate (predictions) 

//Imprimir la tasa de error
println ("tasa de error =" + (1-precision))
``` 
