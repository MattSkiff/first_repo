# something changing overtime
library(ggplot2)
library(xkcd)

# download.file("http://simonsoftware.se/other/xkcd.ttf",
#               dest="xkcd.ttf", mode="wb")
# 
# font_import(pattern = "[X/x]kcd", prompt=FALSE)
# fonts()
# fonttable()
# if(.Platform$OS.type != "unix") {
#   ## Register fonts for Windows bitmap output
#   loadfonts(device="win")
# } else {
#   loadfonts()
#}

over <- data.frame(Time = as.factor(c("Jan",
                            "Feb",
                            "Mar",
                            "Apr",
                            "May",
                            "Jun",
                            "Jul",
                            "Aug",
                            "sep",
                            "Oct",
                            "Nov",
                            "Dec")), #seq(ymd('2012-04-07'),ymd('2013-03-22'), by = '1 week'),
           Something = runif(12,200,800))

over$Time <- ordered(over$Time, levels = c("Jan",
                                           "Feb",
                                           "Mar",
                                           "Apr",
                                           "May",
                                           "Jun",
                                           "Jul",
                                           "Aug",
                                           "sep",
                                           "Oct",
                                           "Nov",
                                           "Dec"))

My_Theme = theme(
  axis.title.x = element_text(size = 16),
  axis.text.x = element_text(size = 14),
  axis.title.y = element_text(size = 16))

g <- ggplot(data = over) +
  geom_line(mapping = aes(x = Time,y = Something,group = 1)) +
  geom_point(mapping = aes(x = Time,y = Something,group = 1)) + 
  labs(title = "Something changing over time",
       subtitle = "A trend of some sort is observed",
       caption = "Mmm,Hmmm....") +
  theme_xkcd() + 
  My_Theme +
  theme(text = element_text(size = 16, family = "xkcd")) +
  annotate("text", x=over$Time[6], y = over$Something[6] - 75,label = "Something\nIn Particular!", family="xkcd", size = 6) +
  xkcd::xkcdaxis(xrange = range(as.numeric(over$Time)),yrange = c(0,1200)) 

g

png("something.png",width = 800,height = 800,type = "cairo")
g
dev.off()

