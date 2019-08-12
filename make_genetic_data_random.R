#df <- data.frame()
#
#for (i in 0:length(strings)/5000) {
#	df <- cbind(df,strings[c((5000*i):(5000*i)+5000)])
#	}


library(dplyr)
df <- mutate_all(as.data.frame(letters), .funs=toupper)
nucleotides <- df[c(1,3,7,20),]
strings<- as.character(sample(nucleotides,4*50*10^5,replace = T))
test.mat <- matrix(data = strings,nrow = 10^5)
test.df <- as.data.frame(test.mat)

#sapply(test.mat[1,],)

test_4.df <- 
data.frame('1' = apply(test.mat[,c(1:50)],FUN = paste,MARGIN = 1,collapse = "") ,
						 '2' = apply(test.mat[,c(51:100)],FUN = paste,MARGIN = 1,collapse = "") ,
						 '3' = apply(test.mat[,c(101:150)],FUN = paste,MARGIN = 1,collapse = "") ,
						 '4' = apply(test.mat[,c(151:200)],FUN = paste,MARGIN = 1,collapse = "") )
						 
write.table(test_4.df,file = "pete_test_data",row.names = F,col.names = F)