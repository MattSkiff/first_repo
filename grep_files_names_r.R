#  find -maxdepth 1 -type f ! -name flist.txt -printf  "%P\n" > flist.txt

library(stringr)
images_list <- read.csv("C:\\Users\\user\\Desktop\\PetImages\\train\\flist.csv",header = F)
rownames(images_list) <- c()

images_list$label <-as.integer(str_detect(images_list$V1,"1_"))

write.csv(images_list,"C:\\Users\\user\\Desktop\\PetImages\\train\\flist_label.csv", row.names = FALSE)