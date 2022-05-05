1. Develop a scaling algorithm that calculates the radius of a circle
val circunfe = 40
val pi = 2 * 3.1416
val radio = circunfe / pi
println (radio)

2. Develop a scaling algorithm that tells me if a number is prime
val number = 13
var check = false
for(i <- 2 to number/2){
    if(number % i == 0){
        check = true
    }
}
if(check==true){
    println("is a not number prime")
}
else{
    println("is a number primer")
}

3. Given the variable var bird = "tweet", use string interpolation to print "I'm writing a tweet"
val bird = "tweet"
val interpolate = "I'm writing a " + bird

4. Given the variable var message = "Hello Luke I am your father!" uses slice to extract the "Luke" sequence

val message = "Hello Luke, I'm your father!"
message slice (6,10)

5.What is the difference between value (val) and a variable (var) in scala?
val is defined as a constant value and var is defined as a variable that can change without redefining it.

6. Given the tuple (2,4,5,1,2,3,3.1416,23) it returns the number 3.1416
val x = List (2,4,5,1,2,3,3.1416,23)
x (6)