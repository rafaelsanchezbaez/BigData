# Evaluation practice
### Instructions
Develop the following statements in Spark using the Scala programming language.
### Objective:
The objective of this practical exam is to try to group customers from specific regions of a wholesale distributor. This is based on the sales of some product categories.
The data sources are in the repository: https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Wholesale%20customers%20data.csv

### 1. Import a simple Spark session.
```scala
import org.apache.spark.sql.SparkSession
```

### 2. Use lines of code to minimize errors
```scala
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

### 3. Create an instance of the Spark session
```scala
spark val = SparkSession.builder().getOrCreate()
```

### 4. Import the Kmeans library for the clustering algorithm.
```scala
import org.apache.spark.ml.clustering.KMeans
```

### 5. Load the Wholesale Customer Dataset
```scala
dataset val = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale_customers_data.csv")
```

### 6. Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
```scala
val feature_data=dataset.select("Fresh", "Milk", "Supermarket", "Frozen", "Paper_Detergents","Delicassen")
```


### 7. Import Vector Assembler and Vector
```scala
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
```

### 8. Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
```scala
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
```

### 9. Use the assembler object to transform feature_data
```scala
val features = assembler.transform(feature_data)
```

### 10.Create a Kmeans model with K=3
```scala
val kmeans = new KMeans().setK(3).setSeed(1L)
model val = kmeans.fit(features)
```

### 11.Evaluate the clusters using the sum of squared errors within the WSSSE set and print the centroids.
```scala
val WSSSE = model.computeCost(features)
println(s"Within the stated sum of squared errors = $WSSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)
```
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/Unit_3/evaluation/evaluation_practice/pic1Unit3.jpg)
