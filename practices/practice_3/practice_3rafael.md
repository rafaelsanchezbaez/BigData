# Random forest
Random Forest or Random Forest, is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both classification and regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and improve model performance. As the name suggests, "Random Forest is a classifier that contains a series of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on a decision tree, the random forest takes the prediction of each tree and based on the majority votes of the predictions, predicts the final result.
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_3/pic1.jpg?raw=true)
The higher number of trees in the forest leads to higher accuracy and avoids the problem of overfitting.
### Prediction
To make a prediction on a new instance, a random forest must aggregate the predictions from its set of decision trees. This aggregation is done differently for classification and regression. Classification: Majority vote. The prediction of each tree is counted as one vote for a class. The tag is predicted to be the class that receives the most votes. Regression: Average. Each tree predicts an actual value. The label is predicted to be the average of the tree predictions.
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_3/pic2.jpg?raw=true)
##### Why use random forest?
- It takes less training time compared to other algorithms. 
- Predicts the output with high accuracy, even for the large dataset that runs efficiently. 
- It can also maintain accuracy when a large proportion of data is missing.
##### random forest apps
- Banking: The banking sector mainly uses this algorithm to identify loan risk.
- Medicine: With the help of this algorithm, disease trends and disease risks can be identified. -Land use: We can identify areas of similar land use using this algorithm.
- Marketing: Marketing trends can be identified using this algorithm.
##### Advantage
- Random Forest is capable of performing classification and regression tasks. -It is capable of handling large data sets with high dimensionality. -Improves the accuracy of the model and avoids the problem of overfitting.
##### Disadvantages
- Although random forest can be used for both classification and regression tasks, it is not more suitable for regression tasks.
### Classification Example
``` scala
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println(s"Test Error = $testErr")
println(s"Learned classification forest model:\n ${model.toDebugString}")

// Save and load model
model.save(sc, "target/tmp/myRandomForestClassificationModel")
val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
```
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_3/Ramdomforest.png?raw=true)
### Regression Example
``` scala
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 3 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "variance"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println(s"Test Mean Squared Error = $testMSE")
println(s"Learned regression forest model:\n ${model.toDebugString}")

// Save and load model
model.save(sc, "target/tmp/myRandomForestRegressionModel")
val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")
```
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_3/Regresion.png?raw=true)


