# Practice 4
Given the pseudocode of the Fibonacci sequence in the provided link, implement Algorithm 1, Algorithm 2, Algorithm 3, Algorithm 4, Algorithm 5 with Scala
1. Algorithm 1
```scala
scala> val n=10 
n: Int = 10

scala> def fib1(n:Int):Int={
     |     if(n<2)
     |     {
     |         return n
     |     }
     |     else 
     |     {
     |         return(fib1(n-1)+fib1(n-2))
     |     }
     | }
fib1: (n: Int)Int

scala> println(fib1(n))
55
```
2. Algorithm 2
```scala
scala> val n = 8
n: Int = 8

scala> var phi=((1+math.sqrt(5))/2) 
phi: Double = 1.618033988749895

scala> var j=((math.pow(phi,n)-math.pow((1-phi),n))/(math.sqrt(5))) 
j: Double = 21.000000000000004

scala> 

scala> 

scala> def fib2(n:Double) : Double ={
     | if (n<2){
     | return n
     | }
     | else {
     | 
     |     return j
     | }
     | }
fib2: (n: Double)Double

scala> println(fib2(n))
21.000000000000004
```
3. Algorithm 3
```scala
scala> def fib3(n:Int):Int={
     | var n : Int = 7 
     | var a = 0 
     | var b = 1 
     | var c = 0 
     | 
     |     for(k <- 1 to n) {
     |         
     |         c = b + a 
     |         a = b 
     |         b = c   
     |     }
     |      return a  
     | }
fib3: (n: Int)Int

scala> println(fib3(n))
13
```
4. Algorithm 4
```scala
scala> def fib4(n:Int):Int={  
     | var n : Int = 6 
     | var a = 0 
     | var b = 1  
     | 
     | for (k <- 1 to n){
     |     b = b + a 
     |     a = b - a 
     | }
     | return a 
     | }
fib4: (n: Int)Int

scala> println(fib4(n))
8

```
5. Algorithm 5
```scala
scala> def fib5(n:Int):Int={
     |     var n = 7 
     |     var vector = new Array[Int](n+1) 
     |     vector(0) = 0 
     |     vector(1) = 1
     |     if(n< 2){ 
     |         return n 
     |     }
     |     for( k <- 2 to n){ 
     |         vector(k) = vector(k - 1) + vector(k - 2)  
     |         
     |     }
     |     return vector(n)
     | }
fib5: (n: Int)Int

scala> println(fib5(n))
13
```