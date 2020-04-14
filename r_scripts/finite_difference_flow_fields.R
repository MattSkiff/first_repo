#!/usr/bin/Rscript

#example of R batch file on windows
# example of writing to .Rout file and printing to windows console

# generate empty matrix
library(ggplot2)
library(reshape2)
library(gganimate)
require(png)
require(gifski)

finite_diff.func <- function(rows = 10,cols = 10,fixed_side = "left",head_value = 100, base_value = 0,min_change = 0.01) {
	
	flow_field.mat <- matrix(data = rep(0,rows*cols),nrow = rows, ncol = cols) #rep(rows*cols,0) #rnorm(rows*cols)
	flow_field.mat[1,] <- rep(head_value,ncol(flow_field.mat)) # replacing top values of matrix
	flow_field.mat[,ncol(flow_field.mat)] <- rep(head_value,nrow(flow_field.mat)) # replacing right hand side values of matrix w/higher water level
	flow_field.mat[nrow(flow_field.mat),1] <- base_value # setting lower water level
	
	message("Initial flow field")
	print(flow_field.mat) # show initial state
	flow_inital <- flow_field.mat
	#write(flow_field.mat, stdout()) #writes to console
	
	flow_fields.ls <- list()
	flow_animate.df <- data.frame()
	h_maps <- list()
	k = 1
	diff <- 10
	while (diff > min_change) {
		flow_old.mat <- flow_field.mat
		for (j in 1:nrow(flow_field.mat)) { # rows
			for (i in 1:ncol(flow_field.mat)) { # columns
				if (i != 1 & i != ncol(flow_field.mat) & j != 1 & j != nrow(flow_field.mat)) { 
					# not left or right columns or bottom or top columns
					flow_field.mat[j,i] <- (flow_field.mat[j+1,i]+flow_field.mat[j,i+1]+flow_field.mat[j,i-1]+flow_field.mat[j-1,i])/4 
					# h1+h2+h3+h4  - interior nodes
				} else if (j == nrow(flow_field.mat) & i != ncol(flow_field.mat) & i != 1) { 
					# is bottom row, not last column or first column
					flow_field.mat[j,i] <- (2*flow_field.mat[j-1,i]+flow_field.mat[j,i+1]+flow_field.mat[j,i-1])/4 # h1 + 2*h2 + h4 
					#} else if (j == nrow(flow_field.mat) & i == ncol(flow_field.mat)) { # corner case (only 1 node)
					#flow_field.mat[j,i] <- 3#(2*flow_field.mat[j,i-1]+2*flow_field.mat[j-1,i])/4 # 2*h1 + 2*h4 
					# this remains the same! As this is the impermeable right hand bound (represents water flow?)
				} else if (i == 1 & j != 1 & j != nrow(flow_field.mat)) { # is left hand column, not first or bottom row
					flow_field.mat[j,i] <- (flow_field.mat[j-1,i]+2*flow_field.mat[j,i+1]+flow_field.mat[j+1,i])/4 # h1 + 2*h2 + h4 
				}
			}
		}
		diff <- sum(abs(flow_field.mat)) - sum(abs(flow_old.mat))
		k <- k + 1
		flow_fields.ls[[k]] <- flow_field.mat
		flow_stacked.df <- melt(flow_field.mat)
		flow_stacked.df <- cbind(flow_stacked.df,rep(k,nrow(flow_stacked.df)))
		flow_animate.df <- rbind(flow_animate.df,flow_stacked.df)
	}
	
	flow_final <- flow_fields.ls[[k]]
	
	colnames(flow_animate.df) <- c("x","y","flow","iteration")
	results <- list(flow_animate.df,flow_inital,flow_final)
	return(results)
}

results <- finite_diff.func()

# visualised as the flip transformation of the matrix
g <- ggplot(data = results[[1]]) + 
	geom_tile(mapping = aes(x = x, y = y,fill = flow)) +
	transition_time(iteration) +
	labs(title = "Finite Difference Model of Water Flow", subtitle = "Iteration: {frame_time}") +
	theme_light() +
	scale_fill_viridis_c() +
	ease_aes("linear")

anim_save("outfile.gif",animate(g,nframes = 200,fps = 20,height = 800, width = 800))

#print("-Hello World!!",stdout())

print("Initial Conditions")
results[[2]]
print("Final Conditions")
results[[3]]