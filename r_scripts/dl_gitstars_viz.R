# author: matthew skiffington
# # purpose: make simple sorted column chart for dissertation re:dl repos

library(ggplot2)
library(scales)

dl_gitstars.vec <- 
	c(
	'Pytorch' = 39600,
	'Keras' = 47200,
	'fast.ai' = 17400,
	'CNTK' = 16700,
	'Tensorflow' = 142000,
	'DL4J' = 11500,
	'Theano' = 9100,
	'Chainer' = 5300,
	'PlaidML' = 3200,
	'Caffe' = 30000,
	'Torch' = 8500,
	'Apache MxNet' = 18400,
	'Sonnet' = 8300,
	'Lasagne' = 3700,
	'TFLearn' = 9400
	)

framework_stars.df <- data.frame(Framework_Name = names(dl_gitstars.vec),Github_stars = dl_gitstars.vec)

g <- ggplot(data = framework_stars.df) +
	geom_col(mapping = aes(y = Github_stars,x = reorder(Framework_Name,-Github_stars))) +
	labs(title = "Github Stars of Popular Deep Learning Packages",
			 subtitle = "As of 13th March, 2020",
			 caption = "Torch, Caffee and Theano are not in active development",
			 x = "Framework",
			 y = "Github Stars") +
	theme_light() +
	theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)) +
	scale_y_continuous(labels = comma) 

png(filename = "G:/My Drive/Dissertation/figures/gitstars.png",
		width = 800, height = 600, units = "px", 
		res = 120,
		bg = "white",  
		type = "cairo-png")

g

dev.off()

g