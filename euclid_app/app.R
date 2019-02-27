library(shiny)
library(ggplot2)

ui <- fluidPage(title = "Euclidean Norms and Distances",
                h1("Euclidean Norms and Distances"),
                plotOutput("euclid"),
                tableOutput("stats"),
                actionButton("generate","Generate New Points")
)

server <- function(input, output) {
output$euclid <- renderPlot({
  # two points in euclidean n-space (n = 2)
  df <- rbind(c(0,0,1),c(0,0,2),data.frame(x = rv$p,y = rv$q,id = c(1,2)))
  ggplot(data = df) + 
    geom_point(mapping = aes(x = x, y = y,colour = id),size = 3) +
    geom_line(mapping = aes(x = x,y = y,group = id, colour = id)) +
    #labs(caption = ) + 
    theme_light(base_size = 22) + 
    labs(caption = paste0("p = ",round(rv$p[1],2),", ",round(rv$p[2],2), " | q = ",round(rv$q[1],2),",",round(rv$q[2],2))) +
    lims(x = c(-30,30),y = c(-30,30))
  }) 

output$stats <- renderTable({
  euclid_distance <- sqrt((rv$p[1]-rv$q[1])^2 + (rv$p[2]-rv$q[2])^2)
  l1 <- abs(rv$p[1] - rv$q[1]) + abs(rv$p[2] - rv$q[2])
  euclid_norm_p <- sqrt(sum(rv$p)^2)
  euclid_norm_q <- sqrt(sum(rv$q)^2)
  data.frame(`Euclidean Distance` = euclid_distance,
             `L1 Distance`  = l1,
             `Euclidean Norm P` = euclid_norm_p,
             `Euclidean Norm Q` = euclid_norm_q)
})

rv <- reactiveValues(p = c(1,1),q = c(1,2))

observeEvent(input$generate,{
  rv$p <- rnorm(2,0,10)
  rv$q <- rnorm(2,0,10)
  
  d <- 20 # d dimensions
  N <- 10000 # N points
  V = matrix(data = rnorm(N*d,0,10),nrow = N) # set of N points
  C <- 20 # absolute constant C
  episilon <- 0.5 # some error between 0 and 1
  
  pairwise_distances <- 
  
  UB <- (1-episilon)
  
  
})

}

shinyApp(ui = ui,server = server)

#runApp() - uncomment to run app
#rsconnect::deployApp('G:\\My Drive\\dev_working_folder',account = 'mattskiff')
