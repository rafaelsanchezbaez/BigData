# Practice evaluation 3
#### Import a simple Spark session.
```scala
``` 
#### Use lines of code to minimize errors
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
``` 
#### Create an instance of the Spark session
```scala
val spark = SparkSession.builder().getOrCreate()
``` 
#### Importar la librería de Kmeans para el algoritmo de agrupamiento.
```scala
import org.apache.spark.ml.clustering.KMeans
``` 
#### Load the Wholesale Customers Data dataset
```scala
val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").
load("Wholesale customers data.csv")
``` 
#### Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
```scala
val feature_data=dataset.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper","Delicassen")

``` 
#### Import Vector Assembler and Vector
```scala
import org.apache.spark.ml.feature.VectorIndexer 
import org.apache.spark.ml.feature.VectorAssembler
``` 
#### Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
```scala
val assembler = new VectorAssembler().setInputCols(Array("Fresh",
"Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
``` 
#### Use the assembler object to transform feature_data
```scala
val features = assembler.transform(feature_data)
``` 
#### Create a Kmeans model with K=3
```scala
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(features)
``` 
#### Evaluate the clusters using Within Set Sum of Squared Errors WSSSE and print the centroids.
```scala
val WSSSE = model.computeCost(features)
println(s"Within Set Sum of Squared Errors = $WSSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)
``` 
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic1.jpg?raw=true)