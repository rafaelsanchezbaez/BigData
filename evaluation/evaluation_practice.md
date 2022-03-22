
# Evaluative practice

### Instructions
Answer the following questions with Spark DataFrames and Scala using the "CSV"
Netflix_2011_2016.csv found in the spark-dataframes folder.
 
#### 1. Start a simple Spark session.
```Scala
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```
 
#### 2. Load the Netflix Stock CSV file, have Spark infer the data types.
```Scala
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
```
 
#### 3. What are the names of the columns?
```Scala
df.columns
"Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"
```
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p1.jpg)
 
#### 4. How is the scheme?
```Scala
df.printSchema()
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p2.jpg) 
 
#### 5. Print the first 5 columns.
```Scala
df.show(5)
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p3.jpg) 
 
#### 6. Use describe() to learn about the DataFrame.
```Scala
df.describe().show()
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p4.jpg)

#### 7. Create a new dataframe with a new column called “HV Ratio” which is the ratio of the price in the “High” column to the “Volume” column of shares traded for one day. Hint - is an operation
```Scala
val df_HV = df.withColumn("HV Ratio",df("High")/df("Volume"))
df_HV.show(5)
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p5.jpg)

#### 8. Which day had the highest peak in the “Open” column?
```Scala
df_HV.orderBy($"High".desc).show(1)
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p6.jpg)
 
#### 9. What is the meaning of the “Close” column in the context of financial information,
explain it, there is no need to code anything?
```s
It is how much was obtained at the end of that day.
```
 
#### 10. What is the maximum and minimum of the “Volume” column?
```Scala
df_HV.select(max("Volume")).show()
df_HV.select(min("Volume")).show()
``` 
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p7.jpg)

   
#### 11. With Scala/Spark Syntax $ answer the following:
##### a. How many days was the “Close” column under $600?
```Scala
df_HV.filter($"Close"<600).count()
```
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p8.jpg)
 
 
##### b. What percentage of the time was the "High" column greater than $500?
```Scala
(df_HV.filter($"High">500).count()*1.0/df_HV.count())*100
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p9.jpg)
 
 
##### c. What is the Pearson correlation between the “High” column and the “Volume” column?
```Scala
df_HV.select(corr("High","Volume")).show()
```
![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p10.jpg)

  
##### d. What is the maximum of the “High” column per year?
```Scala
val dfyear = df_HV.withColumn("Year",year(df_HV("Date")))
val maxyear = dfyear.select($"Year",$"High").groupBy("Year").max()
maxyear.orderBy($"Year").select($"Year",$"max(High)").show()
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p11.jpg)

 
##### e. What is the average of the “Close” column for each calendar month?
```Scala
val dfmonth = df_HV.withColumn("Month",month(df_HV("Date")))
val meanmonth = dfmonth.select($"Month",$"Close").groupBy("Month").mean()
meanmonth.orderBy($"Month").select($"Month",$"avg(Close)").show()
```
 ![](https://github.com/rafaelsanchezbaez/Big_Data/blob/unidad_1/_images/Evaluation_Practice_p12.jpg)
