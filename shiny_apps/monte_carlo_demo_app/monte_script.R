# From https://www.r-bloggers.com/estimation-of-the-number-pi-a-monte-carlo-simulation/
# Static non-shiny version
simulation = function(long){
  c = rep(0,long)
  numberIn = 0
  for(i in 1:long){
    x = runif(2,-1,1)
    if(sqrt(x[1]*x[1] + x[2]*x[2]) <= 1){
      numberIn = numberIn + 1
    }
    prop = numberIn / i
    piHat = prop *4
    c[i] = piHat
  }
  return(c)
}
size = 100000
res = simulation(size)
ini = 1
plot(res[ini:size], type = 'l')
lines(rep(pi, size)[ini:size], col = 'red')