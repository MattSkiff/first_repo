# author: matthew skiffington
# purpose: plain viz of vc4 (surpassing vc dimension) for linear classifiers to go in dissertation - 16 plots
# highlights when data isn't shattered
# randomly generates 3 points and fits a glm + plots decision boundary
# original source: glm code adapted from:
# glm code adapted from : https://stats.stackexchange.com/questions/6206/how-to-plot-decision-boundary-in-r-for-logistic-regression-model/6207
# plot code apated from: https://www.r-bloggers.com/beyond-basic-r-plotting-with-ggplot2-and-multiple-plots-in-one-figure/

library(ggplot2) # viz
library(cowplot) # multi-viz

# png(filename="vc_4.png", 
#     type="cairo",
#     units="px", 
#     width=1800, 
#     height=1800, 
#     pointsize=12, 
#     res=76)

# randomised plot generator
vc_4.func <- function(x) {
  rand_points.vec <- runif(n = 8,min = 0,max = 6)
  
  class_labels.vec <- c(
    "Class 1",
    "Class 1",
    "Class 2",
    "Class 2"
  )
  
  vc_ex.df <- data.frame(
    x = rand_points.vec[1:4],
    y = rand_points.vec[5:8],
    Class = class_labels.vec
  )
  
  model <- glm(Class ~.,family=binomial(link='logit'),data = vc_ex.df)
  slope.num <- coef(model)[2]/(-coef(model)[3])
  intercept.num <- coef(model)[1]/(-coef(model)[3])
  
  red_box.opt <- NULL
  
  shattered.bool <- sum(round(predict(model,type = 'response')) == (as.numeric(vc_ex.df$Class)-1)) == 4
  
  if(!shattered.bool) {
    red_box.geom <- theme(panel.border = element_rect(colour = "red", fill=NA, size=3))
  } else {
    red_box.geom <- NULL
  }
  
  
  g <- ggplot(data = vc_ex.df) + 
    geom_point(mapping = aes(x = x,y = y,fill = Class),colour = 'black',size = 2,shape=21 ,stroke = 0.5,) +
    geom_abline(intercept = intercept.num, slope = slope.num, linetype, colour='black', size = 1) +
    #labs(title = "Illustration of the VC Dimension of a Linear Classifier",subtitle = "Points randomly generated; GLM logistic decision boundary") + 
    #scale_fill_manual(values = c("black","white"),
    #                  labels = c("Class 1","Class 2")) +
    ylim(0,6) +
    xlim(0,6) +
    theme_light() +
    theme(axis.title=element_blank(),
          axis.text=element_blank()) +
    theme(legend.position = 'none') +
    theme(axis.ticks = element_blank()) +
    red_box.geom
  
  return(g)
}

vc_4_plots.ls <- suppressWarnings(lapply(FUN = vc_4.func,1:16)) # create plot list

title <- ggdraw() + 
  draw_label("Surpassing the Vapnik Chervonenkis Dimension \n of a Linear Classifier", 
             fontface='bold',
             size = 10)

sub <- ggdraw() + 
  draw_label("Binary data randomly generated. Logistic regression classifier \n fitted with decision boundary plotted in black.\nNon-shattered scenarios highlighted in red.", 
             size = 8)

plots.grid <- plot_grid(plotlist = vc_4_plots.ls,nrow = 4,ncol = 4) # create plot grid
plot_grid(title,plots.grid,sub,ncol = 1, rel_heights = c(0.1,0.9,0.1)) + 
  ggsave2("vc_4.png",
          width = 20,
          height = 20,
          units = 'cm',
          dpi = 600,
          type = "cairo-png") # final plot

# dev.off()


