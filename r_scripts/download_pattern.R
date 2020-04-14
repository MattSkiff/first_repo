download.func <- function(i) {
  #url = paste("http://people.cs.pitt.edu/~milos/courses/cs2710-Fall2017/Lectures/Class",i,".pdf",sep = "")
  url = paste("https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec",i,".pdf",sep = "")
  #destfile = paste("G:/My Drive/cs2710_foundations_ai_upitt/Class",i,".pdf",sep = "")
  destfile = paste("G:/My Drive/csc321_2018_intro_to_nns_ml_utoronto/lec",i,".pdf",sep = "")
  download.file(url = url,destfile = destfile,quiet = T)
}
seq_0 <- sprintf("%02d", 1:23)
lapply(FUN = download.func,X = seq_0)