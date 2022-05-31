# Practice 6 - Linear Support Vector Machine

### Linear Support Vector Machine
Support vector machine (SVM) is a supervised learning algorithm used in many classification and regression problems, including medical signal processing, natural language processing, and image and speech recognition applications.

### How does it work?
Support Vector Machines (created by Vladimir Vapnik) constitute a learning-based method for solving classification and regression problems. In both cases, this resolution is based on a first phase of training (where they are informed with multiple examples already solved, in the form of pairs {problem, solution}) and a second phase of use for problem resolution. In it, SVMs become a "black box" that provides an answer (output) to a given problem (input).

### Objective
The goal of the SVM algorithm is to find a hyperplane that best separates two different classes of data points. The algorithm can only find this hyperplane in problems that allow linear separation; in most practical problems, the algorithm maximizes the soft margin by allowing a small number of misclassifications.

### Kernel
The simplest way to perform the separation is by a straight line, a straight plane, or an N-dimensional hyperplane. Unfortunately, the universes to study are not usually presented in idyllic two-dimensional cases as in the previous example, but an SVM algorithm must deal with: More than two predictor variables. Non-linear separation curves. Cases where the data sets cannot be completely separated. Classifications in more than two categories.
Due to the computational limitations of linear learning machines, they cannot be used in most real-world applications. The representation by means of Kernel functions offers a solution to this problem, projecting the information to a higher dimensional feature space which increases the computational capacity of linear learning machines. That is, we map the input space X to a new feature space of higher dimensionality (Hilbert).

### Kernel Types
- Polynomial-homogeneous.
- Perceptron.
- Gaussian radial basis function.

### Characteristic
SVMs are basically classifiers for 2 classes. You can change the formulation of the QP algorithm to allow multiclass classification. More commonly, the data is "intelligently" split into two parts in different ways and an SVM is trained for each way of splitting. Multiclass classification is done by combining the output of all classifiers.

### Advantage
- Training is relatively easy. There is no local optimum, as in neural networks.
- They scale relatively well for data in high dimensional spaces.
- The trade-off between classifier complexity and error can be controlled explicitly.
- Non-traditional data such as strings and trees can be used as input to the SVM, instead of feature vectors.

### Disadvantages
- A “good” kernel function is needed, that is, efficient methodologies are needed to tune the initialization parameters of the SVM.

### Applications
- Optical character recognition.
- Face detection for digital cameras to focus correctly.
- Spam filters for email.
- Recognition of images on board satellites (knowing which parts of an image have clouds, land, water, ice, etc.)

### Example
```scala
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)

// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
```  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/practices/practice_6/pic1.jpg)  
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/practices/practice_6/pic2.jpg)
