# install.packages("ggplot2") #uncomment if need to install ggplot2
library(ggplot2)
distributions.df <- data.frame(data = c(rnorm(100,50,15),rnorm(100,80,5)),group = c(rep("big",100),rep("small",100)))
ggplot(data = distributions.df) +
  geom_histogram(mapping = aes(x = data,fill = group),group = "group",colour = "black") +
  theme_light() +
  labs(title = "Example Overlaid Histogram of Two Distributions", 
       caption = expression(N[1]==100~","~N[2]==100~","~mu[1]==50~","~mu[2]==80~","~sigma[1]==15~","~sigma[2]==5),
       subtitle = "Data sampled from two normal distributions with different locations with different spread",y = "Frequency",x = "What we are measuring (units)") +
  scale_fill_discrete(name = "Group",labels = c("Small","Big"))