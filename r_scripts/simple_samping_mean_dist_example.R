mt.mat <- matrix(ncol = 9,nrow = 100)
for (i in 1:100) {
  mt.mat[i,] <- runif(9,0,100)
  #mt.mat[i,] <- rpois(7,100)
}#random sampling from 0-100

means_nine.vec <- apply(mt.mat,1,mean)

mt_2.mat <- matrix(ncol = 50,nrow = 100)
for (i in 1:100) {
  mt_2.mat[i,] <- runif(50,0,100)
  #mt_2.mat[i,] <- rpois(7,100)
}#random sampling from 0-100

means_fifty.vec <- apply(mt_2.mat,1,mean)
unif.vec <- runif(10^6,0,100)

par(mfrow = c(1,3))
hist(means_nine.vec,xlim = c(0,100),col = 'blue',main = "Distribution of Sampling Mean \n Sampling with sample size = 9 \n 100 Samples",xlab = "Value of Means",breaks = 30)
hist(means_fifty.vec,xlim = c(0,100),col = 'red', main = "Distribution of Sampling Mean \n Sampling with sample size = 36 \n 100 Samples",xlab = "Values of Means",breaks = 30)
hist(unif.vec,xlim = c(0,100),col = 'red', main = "Distribution of Data we are Sampling from \n (Uniform Distribution) \n n = 10^6",xlab = "Values of Data",breaks = 30)

