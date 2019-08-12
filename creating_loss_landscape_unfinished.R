# install.packages("tidyr")
library(reshape2)
x_y_loss.df <- expand.grid(data.frame(theta1 = seq(0,1,by = 0.001), theta2 = seq(0,1,by = 0.001)))
x_y_loss.df$loss <- runif(nrow(x_y_loss.df),0,1)
x_y_loss.df$theta1 <- as.factor(x_y_loss.df$theta1)
x_y_loss.df$theta2 <- as.factor(x_y_loss.df$theta2)

x_y_loss_wide.df <- reshape(x_y_loss.df, idvar = "theta1", timevar = "theta2", direction = "wide")
persp(as.matrix(x_y_loss_wide.df),theta = -15)