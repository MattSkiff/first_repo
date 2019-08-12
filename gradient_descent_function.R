# Gradient descent example
# Start with some function f(x) we wish to optimise
# This might be the log likelihood - which can often be optimised analytically 
# Or using more complex methods (e.g. the hessian, newton-raphson, fisher info)

# In practice this will usually be the loss likelihood of a NN
# To make this little script more generalisation, will use an AD package

devtools::install_github("mailund/dfdr")
library(dfdr)

simple_GD.func <- function (objective_function = function(x) x^2 - 2*x, learning_rate = 0.01, initial_guess_min = -100, initial_guess_max = 100, plot = T,plot_window = 100) {
  # NOTE: for dfdr to work, function must be expressed on single line (i.e. without brackets)
  # e.g. eq.func <- function(x) x^2 - 2*x
  require(dfdr)
  
  # check arguments
  if (!is.atomic(learning_rate) | !is.numeric(learning_rate)) {
    stop("Learning rate must be scalar and numeric")
  }
  if (!is.atomic(initial_guess_min) | !is.numeric(initial_guess_min)) {
    stop("Initial guess min must be scalar and numeric")
  }
  if (!is.atomic(initial_guess_max) | !is.numeric(initial_guess_max)) {
    stop("Initial guess max must be scalar and numeric")
  }
  #if (!is.atomic(lower_bound)) {
  #  stop("Min must be scalar")
  #}
  #if (!is.atomic(upper_bound)) {
  #  stop("Max must be scalar")
  #}
  
  # Step 1: Initialise param - for simple function, will be a scalar
  # Initial guess
  theta.vec <- c()
  theta.vec[1] <-  runif(n = 1,min = initial_guess_min,max = initial_guess_max)
  
  # Step 2: Set learning rate - completely arbitrary at this stage
  # Positive
  if (abs(learning_rate) >= 1 | abs(learning_rate) == 0) {
    message("Constant learning rate must be less than 1 (and positive) and non-zero - setting learning rate to 0.999")
    eta.num <- 0.999
  } else {
    eta.num <- abs(learning_rate)
  }
  
  # Partial derivative with respect to x
  # skipped warning and finally argument in try catch
  f <- tryCatch({
    d(objective_function,"x")
  }, error = function(e) {
    message("Function not differentiable via dfdr")
    message("Original warning:")
    stop(e)
  })
  
  theta_diff <- 10
  i <- 2
  # Looping until convergence
  while (theta_diff/eta.num > 0.01) {
    # updating param
    theta.vec[i] <- theta.vec[i-1] - eta.num*f(theta.vec[i-1])
    theta_diff <- abs(theta.vec[i]-theta.vec[i-1])
    i <- i + 1
    if (is.na(theta_diff)) {
      stop("GD diverged")
    }
    if (i > 10^8) {
      stop("Gradient Descent was unable to converge")
    }
  }
  
  # plotting function
  theta_last <- theta.vec[length(theta.vec)]
  if (plot) {
    curve(objective_function,
          from = theta_last-plot_window, 
          to = theta_last+plot_window,
          xlab = "theta",ylab = "f(theta)", 
          main = "Gradient Descent of Objective Function",
          ylim = c(objective_function(theta_last)-plot_window,objective_function(theta_last)+plot_window),
          sub = "theta values")
    mtext(body(objective_function))
    points(x = theta.vec,y = objective_function(theta.vec),col = 'red')
  }
  return(list(theta.vec = theta.vec,theta_last = theta_last,no_iterations = i))
}

  # illustrative examples
    func<- function(x) x^2 - 2*x
    curve(func,to = 10,from = -10)
      # converges
      simple_GD.func()
    # local minima
    func<- function(x) ((x^2 + 1)/x)
    curve(func,to = 10,from = -10)
      # will time out if initial guess less than 0 - no minima below 0
      simple_GD.func(func)
    # ad library can't handle below function
    #func<- function(x) (abs(x-6))
    #curve(func,to = 10,from = -10)
    #simple_GD.func(func)