# Multilayer perceptron classifier.
The Multilayer Perceptron Classifier (MLPC) is a classifier based on the feed-forward artificial neural network. The MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer of the network. The input layer nodes represent the input data.
All other nodes map inputs to outputs by linearly combining the inputs with the node weights w and bias b and applying an activation function. This can be written in matrix form for the MLPC with K+1 layers as follows:
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_5/pic1.jpg?raw=true)
The nodes in the intermediate layers use the sigmoid (logistic) function:
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_5/pic2.jpg?raw=true)
The output layer nodes use the softmax function:
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_5/pic3.jpg?raw=true)
##### Multilayer perceptron example:
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_5/pic4.jpg?raw=true)
### Multilayer perceptron classifier.
The number of nodes N in the output layer corresponds to the number of classes. MLPC uses backpropagation to learn the model. We use the logistic loss function for optimization and L-BFGS as optimization routine.
Simple perceptron type Artificial Neural Network with n input neurons, m neurons in its hidden layer and one output neuron.
Layers can be classified into three types:
Input layer: Made up of those neurons that introduce input patterns into the network. No processing occurs in these neurons. Hidden layers: Formed by those neurons whose inputs come from previous layers and whose outputs pass to neurons of later layers. Output layer: Neurons whose output values correspond to the outputs of the entire network.
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_5/pic5.jpg?raw=true)
### Applications.
The multilayer perceptron (hereinafter MLP, MultiLayer Perceptron) is used to solve problems of pattern association, image segmentation, data compression, etc.
### Example.
``` scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame. || Carga los datos almacenados en formato LIBSVM como DataFrame.

//val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
val data = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_multiclass_classification_data.txt")

// Split the data into train and test || Divide los datos
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// specify layers for the neural network: || especificar capas para la red neuronal:
// input layer of size 4 (features), two intermediate of size 5 and 4 || capa de entrada de tamano 4 (features), dos intermedias de tamano 5 y 4
// and output of size 3 (classes) || y salida de tamano 3 (classes) 
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters || Crea el trainer y establece sus parametros.
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

// train the model || entrena el model
val model = trainer.fit(train)

// compute accuracy on the test set || precision de calculo en el conjunto de prueba
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
``` 
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unit_2/practices/practice_5/pic6.jpg?raw=true)
