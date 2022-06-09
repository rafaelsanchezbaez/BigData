# Gradient boosted tree classifier
##### Introduction
Gradient-boosted trees (GBTs) are a classification and regression method based on the use of decision trees. The classification system trains the decision trees to minimize failures.
##### Decision Tree
The classification system is responsible for creating these decision trees that will allow accurate predictions to be obtained. For these predictions to be correct, the system takes care of iterating the trees predicting results to obtain better results.
##### Basic Algorithm
Gradient boosting iteratively trains a sequence of decision trees. At each iteration, the algorithm uses the current set to predict the label for each training instance, and then compares the prediction to the true label. The dataset is relabeled to put more emphasis on training instances with poor predictions. Therefore, in the next iteration, the decision tree will help correct the previous errors.
The specific mechanism for relabeling instances is defined by a loss function (discussed below). With each iteration, the GBTs further reduce this loss function on the training data.
##### Data Input/Output
Once the iteration and training process is done, the classification system takes the selected Data Frame, runs the model and makes the prediction.
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_4/pic2.jpg?raw=true)
##### Usage Tips
We include some guidelines for using GBT by analyzing the various parameters. We omit some decision tree parameters as they are covered in the decision tree guide.
- loss – See the previous section for information on loss and its applicability to tasks (classification vs. regression). Different losses can give significantly different results, depending on the data set.
- numIterations: This sets the number of trees in the set. Each iteration produces a tree. Increasing this number makes the model more expressive, which improves the accuracy of the training data. However, the accuracy of the test time may be affected if it is too large.
- learningRate: It is not necessary to adjust this parameter. If the behavior of the algorithm seems unstable, lowering this value can improve stability.
- algo: The algorithm or task (classification vs. regression) is configured using the Strategy tree parameter.
##### Example
``` scala
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a GradientBoostedTrees model.
// The defaultParams for Classification use LogLoss by default.
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
// Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println(s"Test Error = $testErr")
println(s"Learned classification GBT model:\n ${model.toDebugString}")

// Save and load model
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
val sameModel = GradientBoostedTreesModel.load(sc,
  "target/tmp/myGradientBoostingClassificationModel")
``` 
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_4/Gradient%20boosted%20tree%20classifier.png?raw=true)

As a result, 3 trees are obtained with a total of 2 classes for each tree.