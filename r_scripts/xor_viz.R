# author: matthew skiffington
# purpose: make xor problem simple viz to go in dissertation

library(ggplot2)
xor.df <- setNames(object = data.frame(
  rbind(
    c(0,1,"Class 1"),
    c(1,0,"Class 1"),
    c(1,1,"Class 2"),
    c(0,0,"Class 2")
  )
),
nm = c("x","y","Class")) 

ggplot(data = xor.df) + 
  geom_point(mapping = aes(x = x,y = y,fill = Class),colour = 'black',size = 5,shape=21 ,stroke = 2,) +
  geom_abline(intercept = -1.5, slope = 2, linetype, colour='red', size = 2) +
  geom_abline(intercept = 0, slope = 1.5, linetype, colour='blue', size = 2) +
  geom_abline(intercept = 5, slope = -2.5, linetype, colour='green', size = 2) +
  labs(title = "The XOR problem",subtitle = "Lines indicate arbitrary examples of linear classifiers") + 
  scale_fill_manual(values = c("black","white"),
                     labels = c("Class 1","Class 2")) +
  theme_light() +
  ggsave("cairo.png", type = "cairo") # avoids ugly aliasing 