#------------twitteR

install.packages("twitteR")
library(twitteR)

# Change the next four lines based on your own consumer_key, consume_secret, access_token, and access_secret. 
consumer_key <- "mXoAbo8OqeBPNoJoJ0LhPwnQU"
consumer_secret <- "OrdHYUUr8qnK3mB175JHOIElTceDnDeLsZYzI4lzHTwHxh57PN"
access_token <- "1036633749893525504-rikF1mGQPFMHykT20d9VTvqrcGMu00"
access_secret <- "hwD8CWgJqstLwn7Zoa1Y94r3u7PfsQcZai0khudCGWDGW"

1y
tw = twitteR::searchTwitter('#Waikato', n = 10, since = '2016-11-08', retryOnRateLimit = 1e3)
d = twitteR::twListToDF(tw)

1tw_uD <- twListToDF(tw_user)

text3.csv <- write.csv(tw_uD$text,"text2.csv")

#-------------rtweet

install.packages("rtweet")
library("rtweet")
tweet_df<- get_timeline("waikato", 3200)

token <- create_token(
  app = "api_data_stream",
  consumer_key = "mXoAbo8OqeBPNoJoJ0LhPwnQU",
  consumer_secret = "OrdHYUUr8qnK3mB175JHOIElTceDnDeLsZYzI4lzHTwHxh57PN",
  access_token = "1036633749893525504-rikF1mGQPFMHykT20d9VTvqrcGMu00",
  access_secret = "hwD8CWgJqstLwn7Zoa1Y94r3u7PfsQcZai0khudCGWDGW")

## check to see if the token is loaded
identical(token, get_token())

tweet_df2<- get_timeline("waikato", 3200,tweet_mode = "extended",max_id = 961032538750828544)
text5.csv <- write.csv(tweet_df$text,"waikato.csv")

#-------images

library(magrittr)
library(dplyr)
library(tidyr)

tweet_df <- as.data.frame(tweet_df)

media_named_chr <- t(as.data.frame(subset(tweet_df2, is.na(media_url) == F)$media_url))
media_named_chr <- media_df[,1]
#media_vec <- as.vector(media_df)

for (media_named_chr in media_named_chr) {
  download.file(media_named_chr, destfile = basename(media_named_chr),mode = 'wb')
}

write.csv(media_urls.csv,subset(tweet_df, is.na(media_url) == F)$media_url)




