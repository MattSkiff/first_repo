# author: matthew skiffington
# date: 09/03/2020
# purpose: produce timeline of nns for dissertation

# data #################

library(timelineS)
library(lubridate)

# old
Events_old <- c(
  "M&P Neuron \n Model* \n", #  1943-01-01
  "Dartmouth Summer AI \n", #   1955-08-31 
  "Perceptron paper* \n", #        1958-01-01
  "Perceptrons book* \n", #     1969-01-01
  "Hopfield (RNN)* \n", #       1982-01-01
  "Backpropagation \n", #       1986-10-09
  "MLP UAP \n proof \n", #      1993-02-09
  "LTSM \n", #                  1997-11-01
  "LeNet* \n" #                 1998-01-01
)

Event_Dates_old <- ymd(c(
  "1943-01-01", # MP Neuron
  "1955-08-31", # Dartmouth
  "1958-01-01", # P-paper
  "1969-01-01", # P-book
  "1982-01-01", # H-N
  "1986-10-09", # Backprop
  "1993-02-09", # MLP UAP
  "1997-11-01", # LTSM
  "1998-01-01"  # LeNet
))

#########################

# new
Events_new <- c(
  
  "AlexNet \n (ILSVRC) \n", #   2012-10-30
  "R-CNN \n", #                 2013-11-11
  "NiN \n", #                   2013-12-16
  "DRL-Atari \n", #             2013-12-19
  "VAE \n", #                   2013-12-20
  "GAN \n", #                   2014-06-10
  "VGG \n (ILSVRC) \n", #       2014-12-10
  "GoogLeNet \n (ILSVRC) \n", # 2014-12-10
  "YOLO \n", #                  2015-06-08
  "ResNet \n (ILSVRC) \n", #    2015-12-10
  "SqueezeNet \n", #            2016-02-24
  "Projection Nets \n", #       2017-08-09
  "AlphaGo \n (GO)", #          2017-10-19
  "StyleGAN \n",#               2018-12-12
  "AlphaStar (SC2) \n"  #       2019-10-30      
)

Event_Dates_new <- ymd(c(
  "2012-10-30", # AlexNet 
  "2013-11-11", # R-CNNs https://arxiv.org/abs/1311.2524
  "2013-12-16", # NiNs https://arxiv.org/abs/1312.4400
  "2013-12-19", # DRL-Atari
  "2013-12-20", # VAEs https://arxiv.org/abs/1312.6114
  "2014-06-10", # GANs https://arxiv.org/abs/1406.2661
  "2014-09-04", # VGG https://arxiv.org/abs/1409.1556
  "2014-09-17", # GoogLeNet https://arxiv.org/abs/1409.4842
  "2015-06-08", # YOLO https://arxiv.org/abs/1506.02640
  "2015-12-10", # ResNets https://arxiv.org/abs/1512.03385 
  "2016-02-24", # SqueezeNets https://arxiv.org/abs/1602.07360
  "2017-08-09", # ProjectionNet https://arxiv.org/abs/1708.00630
  "2017-10-19", # AlphaGo https://www.nature.com/articles/nature24270
  "2018-12-12", # StyleGAN https://arxiv.org/abs/1812.04948
  "2019-10-30"  # AlphaStar https://www.nature.com/articles/s41586-019-1724-z#citeas
))


# viz ###################

# dataframes
# old
timeline_old.df <- data.frame(
  Events = Events_old,
  Event_Dates = Event_Dates_old
)

# new
timeline_new.df <- data.frame(
  Events = Events_new,
  Event_Dates = Event_Dates_new)

#########################

# timelines
# old
timelineS(
  main = "Historical Events in Neural Network History",
  timeline_old.df,
  label.cex = 0.6
)

# new
timelineS(
  main = "Recent Events in Neural Network History",
  timeline_new.df,
  label.cex = 0.6)