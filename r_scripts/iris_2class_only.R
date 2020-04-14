#https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv
iris<- read.csv("C:\\Users\\skiff\\Desktop\\iris_training.csv")
write.csv(iris[which(iris$virginica != 2),],"C:\\Users\\skiff\\Desktop\\iris_2.csv")