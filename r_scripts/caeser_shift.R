# updated: works for unicode character set (needs to be in wd)
# function only works for characters (no punctuation other than spaces), and converts everything to lowercase first

caeser_shift.func <- function(text,shift = 5,de_encrypt = F) {
  # round(runif(1,0,10),0)
  if (!is.atomic(text) | !is.character(text)) {
    stop("Please supply a character string!")
  }
  characters <- c(letters,LETTERS)
  #text <- tolower(text)
  x.vec <- strsplit(text,split = " ")
  letter.l <- strsplit(unlist(x.vec),split = "")
  indices.l <- lapply(letter.l,function(x) match(x,characters))
  if (de_encrypt) {
    indices_shifted.l <- lapply(indices.l,function(indices.l) indices.l - (shift+1)) # due to +1 later
    indices_normalised.l <- lapply(indices_shifted.l,function(indices_shifted.l) indices_shifted.l %% 52) # accounts for index shifts outside 1-26
    indices_non_zero.l <- lapply(indices_normalised.l,function(indices_normalised.l) indices_normalised.l + 1) # removes zero values, which break indexing
    return.l <- lapply(indices_non_zero.l,function(indices_non_zero.l) characters[indices_non_zero.l])
  } else {
    indices_shifted.l <- lapply(indices.l,function(indices.l) indices.l + (shift-1)) # due to +1 later
    indices_normalised.l <- lapply(indices_shifted.l,function(indices_shifted.l) indices_shifted.l %% 52) # accounts for index shifts outside 1-26
    indices_non_zero.l <- lapply(indices_normalised.l,function(indices_normalised.l) indices_normalised.l + 1) # removes zero values, which break indexing
    return.l <- lapply(indices_non_zero.l,function(indices_non_zero.l) characters[indices_non_zero.l])
  }
  return.l <- lapply(return.l,function(return.l) paste(return.l,collapse = ""))
  return.l <- lapply(return.l,function(return.l) paste(return.l," "))
  return(trimws(paste(unlist(return.l),collapse = ""),which = "right"))
}

# example encrypt
caeser_shift.func("hello my name is",5,F)
caeser_shift.func("the quick brown fox jumps over the lazy",5,F)

# example de_encrypt
caeser_shift.func(caeser_shift.func("Hello my name is",5,F),5,T)
caeser_shift.func(caeser_shift.func("The quick brown fox jumps over the lazy",5,F),5,T)