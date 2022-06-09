![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática y Ingeniería en Tecnologías de la Información y Comunicación.
### Proyecto / Tarea / Práctica:
#### Decision Tree Classifier
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno
- Alonso Villegas Luis Antonio
- Rafael Sanchez Baez
### Fecha:
#### Tijuana Baja California a 07 de 04 2022 

## Decision trees
Decision trees and their sets are popular methods for classification and regression learning tasks.automatic. Decision trees are widely used because they are easy to interpret, handle categorical features,
they extend to multiclass classification settings, do not require feature scaling, and can capture nonlinearities and feature interactions. Ensemble-of-trees algorithms, such as random forests and boosting, are among the better for classification and regression tasks.

A decision tree has a structure similar to a flowchart where an internal node represents a characteristic or attribute, the branch represents a decision rule and each node or leaf represents the result. The top node of a decision tree is known as the root node.

#### The basic idea behind any decision tree problem is the following:
- Select the best attribute using an attribute or feature selection measure.
- Make that attribute a decision node and split the dataset into smaller subsets.
- Start building the tree by repeating this process recursively for each attribute until one of the following conditions is matched:
   - All variables belong to the same attribute value.
   - No more attributes left
   - There are no more cases.

Decision tree-based learning is a commonly used method in data mining. The goal is to create a model that predicts the value of a target variable based on various input variables.

![](http://dataanalyticsedge.com/wp-content/uploads/2018/01/1-1.jpg)

A decision tree is a simple representation for classifying examples. Decision tree-based learning is one of the most efficient techniques for supervised classification. For this section, all features are assumed to have finite discrete domains, and there is a single target feature called the classification. Each element of the classification domain is called a class. A decision tree or classification tree is a tree in which each internal (non-leaf) node is labeled with an input function. Arcs from a node labeled with a feature are labeled with each of the possible values ​​of the feature. Each leaf of the tree is marked with a class or a probability distribution over the classes.

A tree can be "learned" by partitioning the initial set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion ends when the subset at a node all has the same value of the target variable, or when the partition no longer adds value to the predictions. This top-down induction process of decision trees is an example of a greedy algorithm, and is by far the most common strategy for learning decision trees from data.
In data mining, decision trees can also be described as the combination of mathematical and computational techniques to aid in the description, categorization, and generalization of a given set of data.
The data comes in records of the form:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/9354e3bfc0c65eb88a0bf7b6b625dcdbc9e74248)

The dependent variable, Y, is the target variable that we are trying to understand, classify, or generalize. The vector x is made up of the input variables, x1, x2, x3 etc., that are used for that task.

### Example  
  
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```


### Explanatory video
<https://www.youtube.com/watch?v=JcI5E2Ng6r4&ab_channel=IntuitiveMachineLearning>

<https://www.youtube.com/watch?v=Ih3U8Rju5ck>

### References.
<https://es.wikipedia.org/wiki/Aprendizaje_basado_en_%C3%A1rboles_de_decisi%C3%B3n>
<https://www.youtube.com/watch?v=JcI5E2Ng6r4&ab_channel=IntuitiveMachineLearning>
<https://spark.apache.org/docs/2.4.7/mllib-decision-tree.html>
<https://spark.apache.org/docs/2.4.7/ml-classification-regression.html?fbclid=IwAR3QHShNZQ-gTK3XzVKacVE7NORmYZqX_74qqDw_Yr2lx1sA-nEJJcPh0Kw#decision-tree-classifier>
