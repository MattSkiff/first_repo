relu.func <- function(x) {
  sapply(
    FUN = function(x) {
      if(x<0) {
        return(0) #y
      } else {
        return(x)
      }
    },
    X = x,
    simplify = T
  )
}
x.vec <- seq(-1000,1000)/1000
y.vec <- relu.func(x.vec)
plot(type = 'l',x=x.vec,y=y.vec,xlab = "x",ylab = "y",main = "The Rectified Linear Unit")