heaviside.func <- function(x) {
  sapply(
    FUN = function(x) {
      if(x<0) {
        return(0) #y
      } else {
        return(1)
      }
    },
    X = x,
    simplify = T
  )
}
x.vec <- seq(-10000,10000)/10000
y.vec <- heaviside.func(x.vec)
plot(type = 'l',x=x.vec,y=y.vec,xlab = "x",ylab = "y",main = "The Heaviside Activation Function")