download.func <- function(i) {
  url = paste("http://people.cs.pitt.edu/~milos/courses/cs2710-Fall2017/Lectures/Class",i,".pdf",sep = "")
  destfile = paste("G:/My Drive/cs2710_foundations_ai_upitt/Class",i,".pdf",sep = "")
  download.file(url = url,destfile = destfile,quiet = T)
}
lapply(FUN = download.func,X = seq(25))