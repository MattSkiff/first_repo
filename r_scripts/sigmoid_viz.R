# author: matthew skiffington
# purpose: make sigmoid simple viz to go in dissertation

png(filename="fn.png", 
    type="cairo",
    units="px", 
    width=538, 
    height=480, 
    pointsize=12, 
    res=96)

sigmoid.func <- function(x) {
  sapply(
    FUN = function(x) {
      x = (exp(x)/(1+exp(x)))
    },
    X = x,
    simplify = T
  )
}
x.vec <- seq(-60000,60000)/10000
y.vec <- sigmoid.func(x.vec)
plot(type = 'l',x=x.vec,y=y.vec,xlab = "x",ylab = "y",main = "The Sigmoid Activation Function",ylim = c(0,1),xlim = c(-6,6))

dev.off()