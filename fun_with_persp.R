# fun with persp


x <- seq(-10, 10, length= 100)
y <- x
z <- matrix(data = rep(c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,0.99),1000),nrow = 100,ncol = 100,byrow = F)
op <- par(bg = "white")
persp(x, y, z, theta = 30, phi = 30, expand = 0.5, col = "lightblue")
persp(x, y, z, theta = 30, phi = 30, expand = 0.5, col = "lightblue",
			ltheta = 120, shade = 0.75, ticktype = "detailed",
			xlab = "X", ylab = "Y", zlab = "Sinc( r )"
) -> res

x <- seq(-10, 10, length= 100)
y <- x
z <- matrix(data = c(rnorm(3333,0.09,0.001),rnorm(3333,0.39,0.005),rnorm(3334,0.89,0.010)),nrow = 100,ncol = 100,byrow = F)
op <- par(bg = "white")
persp(x, y, z, theta = 30, phi = 30, expand = 0.5, col = "lightblue")
persp(x, y, z, theta = 30, phi = 30, expand = 0.5, col = "lightblue",
			ltheta = 120, shade = 0.75, ticktype = "detailed",
			xlab = "X", ylab = "Y", zlab = "Sinc( r )"
) -> res

x <- seq(-10, 10, length= 100)
y <- x
z <- matrix(data = rep(dnorm(seq(-5,5,length = 100),mean = 0, sd = 1),100),nrow = 100,ncol = 100,byrow = F)
op <- par(bg = "white")
persp(x, y, z, theta = 30, phi = 30, expand = 0.5, col = "red")

#####
#https://stat.ethz.ch/pipermail/r-help/2003-September/038314.html
#
library(MASS)
bivn <- mvrnorm(1000, mu = c(0, 0), Sigma = matrix(c(1, .5, .5, 1), 2))

# now we do a kernel density estimate
bivn.kde <- kde2d(bivn[,1], bivn[,2], n = 50)

# now plot your results
contour(bivn.kde)
image(bivn.kde)
persp(bivn.kde, phi = 45, theta = 30)

# fancy contour with image
image(bivn.kde); contour(bivn.kde, add = T)

# fancy perspective
persp(bivn.kde, phi = 45, theta = 30, shade = 1, border = 1,col = 'red')

# from mass help
attach(geyser)
plot(duration[-272], duration[-1], xlim = c(0.5, 6),
		 ylim = c(1, 6),xlab = "previous duration", ylab = "duration")
f1 <- kde2d(duration[-272], duration[-1],
						h = rep(1.5, 2), n = 50, lims = c(0.5, 6, 0.5, 6))
contour(f1, xlab = "previous duration",
				ylab = "duration", levels  =  c(0.05, 0.1, 0.2, 0.4) )
f1 <- kde2d(duration[-272], duration[-1],
						h = rep(0.6, 2), n = 50, lims = c(0.5, 6, 0.5, 6))
contour(f1, xlab = "previous duration",
				ylab = "duration", levels  =  c(0.05, 0.1, 0.2, 0.4) )
f1 <- kde2d(duration[-272], duration[-1],
						h = rep(0.4, 2), n = 50, lims = c(0.5, 6, 0.5, 6))
contour(f1, xlab = "previous duration",
				ylab = "duration", levels  =  c(0.05, 0.1, 0.2, 0.4) )
detach("geyser")