/*Instrucciones
Responder las siguientes preguntas con Spark DataFrames y Scala utilizando el “CSV”
Netflix_2011_2016.csv que se encuentra el la carpeta de spark-dataframes.
1. Comienza una simple sesión Spark.*/
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//2. Cargue el archivo Netflix Stock CSV, haga que Spark infiera los tipos de datos.
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//3. ¿Cuáles son los nombres de las columnas?
df.columns
"Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"
//4. ¿Cómo es el esquema?
df.printSchema()

//5. Imprime las primeras 5 columnas.
df.show(5)

//6. Usa describe () para aprender sobre el DataFrame.
df.describe().show()

/*7. Crea un nuevo dataframe con una columna nueva llamada “HV Ratio” que es la relación que
existe entre el precio de la columna “High” frente a la columna “Volumen” de acciones
negociadas por un día. Hint - es una operación*/
val df_HV = df.withColumn("HV Ratio",df("High")/df("Volume"))
df_HV.show(5)
//8. ¿Qué día tuvo el pico mas alto en la columna “Open”?
df_HV.orderBy($"High".desc).show(1)

/*9. ¿Cuál es el significado de la columna Cerrar “Close” en el contexto de información financiera,
expliquelo no hay que codificar nada?*/

//10. ¿Cuál es el máximo y mínimo de la columna “Volumen”?
df_HV.select(max("Volume")).show()
df_HV.select(min("Volume")).show()
//11. Con Sintaxis Scala/Spark $ conteste los siguiente:

//a. ¿Cuántos días fue la columna “Close” inferior a $ 600?
df_HV.filter($"Close"<600).count()
//b. ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
(df_HV.filter($"High">500).count()*1.0/df_HV.count())*100
//c. ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
df_HV.select(corr("High","Volume")).show()
//d. ¿Cuál es el máximo de la columna “High” por año?
val dfyear = df_HV.withColumn("Year",year(df_HV("Date")))
val maxyear = dfyear.select($"Year",$"High").groupBy("Year").max()
maxyear.orderBy($"Year").select($"Year",$"max(High)").show()
//e. ¿Cuál es el promedio de columna “Close” para cada mes del calendario?
val dfmonth = df_HV.withColumn("Month",month(df_HV("Date")))
val meanmonth = dfmonth.select($"Month",$"Close").groupBy("Month").mean()
meanmonth.orderBy($"Month").select($"Month",$"avg(Close)").show()
