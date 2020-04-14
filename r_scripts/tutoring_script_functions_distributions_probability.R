# challenge for adam - extend to rademacher distribution
rand_matrix_gaus.func <- function(n = 10,d = 10,sd = 1,u = 0) {
  return(matrix(rnorm(n*d,u,sd),nrow = n))
  # warning, message, stop functions
} 


# second challenge - adapt to calculate the p-norm for given p
# challenge for adam - use this function to build a table of norms for mpg - wrapped into a dataframe
vec_norm.func <- function(vec, norm_type) {
  if (length(vec) == 0) {
    stop("Please suppy a vector to calculate the norm of")
  }
  if (missing(norm_type)) {
    message("Setting norm type as L2")
    norm_type = "L2"
  }
  
    if (is.numeric(vec) == F & is.integer(vec) == F) {
      stop("Please supply either a numeric or integer vector")
    } else if (norm_type == "L0") {
      return(length(vec) - sum(vec == 0))
    } else if (norm_type == "L1") {
      return(sum(abs(vec)))
    } else if (norm_type == "L2") {
      return(sqrt(sum(vec ^ 2)))
    } else if (norm_type == "Max") {
      return(max(abs(vec)))
    } else {
      stop("Please supply a recognised norm type")
    }
  }

# always multiple ways to do things... https://www.r-bloggers.com/5-ways-to-measure-running-time-of-r-code/
# investigate efficiency of operations on matrices - is row wise or column wise more costly?
system.time(mean(apply(X = rand_matrix_gaus.func(n = 500,d = 10), MARGIN = 1,FUN = sd)))

# discrete dist
discrete_dist.func <- function(n=5) {
  p <- runif(n = n, min = 0,max = 1)
  sample.vec <- NULL
    for (i in 1:n) {
      if (p[i] <= 0.1) {
        sample.vec[i] <- 0
      } else if (p[i] <= 0.3) {
        sample.vec[i] <- 1
      }  else if (p[i] <= 0.6) {
          sample.vec[i] <- 2
        } else if (p[i] <= 1) {
          sample.vec[i] <- 5
        }
    }
  return(sample.vec)
}

# T3
mean(rbinom(n = 1*10^6,p = 0.6,size = 10))
var(rbinom(n = 1*10^6,p = 0.6,size = 10))
# q = numbers for which probability equal to or below is calculated
# size = number of trials
# prob = chance of success on each trial 
#  1) lower.tail = are you calculating probability up to the given quantile, or calculating probability
#  2) greater than or equal to the given quantile
pbinom(q = c(1,2,3,4,5,6,7,8,9,10),size = 10,prob = 0.6,lower.tail = T)

# density mass plot
barplot(table(rpois(1*10^6,lambda = 2))/(1*10^6))
ppois(lambda = 2, q=3) - sum(ppois(lambda = 2, q=c(2)))


barplot(table(rpois(1*10^6,lambda = 2))/(1*10^6),ylim = c(0,0.3))

ppois(lambda = 2, q=3) - sum(ppois(lambda = 2, q=c(0)))
                             
var(rpois(10000000,lambda = 2))
sd(rpois(10000000,lambda = 2))
