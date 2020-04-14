# update 18_07_19 - changed brackets around sum and squaring euclid norm p,q

library(shiny)
library(ggplot2)

ui <- fluidPage(title = "Euclidean Norms and Distances",
                h1("Euclidean Norms and Distances"),
                plotOutput("euclid"),
                tableOutput("stats"),
                tableOutput("data"),
                actionButton("generate","Generate New Points")
)

server <- function(input, output,session) {
  
output$euclid <- renderPlot({
  # two points in euclidean n-space (n = 2)
  req(rv$df)
  df <- rv$df
  ggplot(data = df) + 
    geom_point(mapping = aes(x = x, y = y,colour = id),size = 3) +
    geom_line(mapping = aes(x = x,y = y,group = id, colour = id)) +
    #labs(caption = ) + 
    theme_light(base_size = 22) + 
    labs(caption = paste0("p = ",round(rv$p[1],2),", ",round(rv$q[1],2), " | q = ",round(rv$p[2],2),",",round(rv$q[2],2))) +
    lims(x = c(min(df$x),max(df$x)),y = c(min(df$y),max(df$y))) +
    scale_colour_manual(values = c(1,2),labels=c("Line p","Line q"))
  }) 

output$stats <- renderTable({
  euclid_distance <- sqrt((rv$p[1]-rv$q[1])^2 + (rv$p[2]-rv$q[2])^2)
  l1 <- abs(rv$p[1] - rv$q[1]) + abs(rv$p[2] - rv$q[2])
  euclid_norm_p <- sqrt(sum(rv$p^2))
  euclid_norm_q <- sqrt(sum(rv$q^2))
  data.frame(`Euclidean Distance` = euclid_distance,
             `L1 Distance`  = l1,
             `Euclidean Norm P` = euclid_norm_p,
             `Euclidean Norm Q` = euclid_norm_q)
})

output$data <- renderTable({
  row.names(rv$df) <- c("origin p","origin q","point p","point q")
  rv$df
},rownames = T)

rv <- reactiveValues(p = c(1,1),
                     q = c(1,2),
                     df = rbind(c(0,0,1),c(0,0,2),data.frame(x = c(1,1),y = c(1,2),id = factor(c(1,2)))))

observeEvent(input$generate,{
  rv$p <- rnorm(2,0,10)
  rv$q <- rnorm(2,0,10)
  rv$df <- rbind(c(0,0,1),c(0,0,2),data.frame(x = rv$p,y = rv$q,id = c(1,2)))
  rv$df[,"id"] <- as.factor(rv$df[,"id"])
  
  # d <- 20 # d dimensions
  # N <- 10000 # N points
  # V = matrix(data = rnorm(N*d,0,10),nrow = N) # set of N points
  # C <- 20 # absolute constant C
  # episilon <- 0.5 # some error between 0 and 1
  # 
  # pairwise_distances <- 
  # 
  # UB <- (1-episilon)
  
})

session$onSessionEnded(stopApp)
}

)shinyApp(ui = ui,server = server)

#runApp() - uncomment to run app
#rsconnect::deployApp("C:\\Users\\Matthew\\Google Drive\\dev_working_folder\\shiny_apps\\euclid_app",account = 'mattskiff')
