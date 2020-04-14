library(shiny)
library(ggplot2) # heatmap
library(gganimate) # to animate ggplot2 heatmap
library(reshape2) # to melt dfs
library(DT) # for tables that don't trail off the pages...
library(shinycssloaders) # loader as viz takes a while to load
require(gifski) # needed for gganimate
require(png) # needed for gganimate

ui <- fluidPage(
  
    titlePanel("Finite Difference Model for Water Flow Demonstration"),
    sidebarLayout(
      
      sidebarPanel(
        
        sliderInput("head_level", "Head Level:",
                    min = 20, max = 100,
                    value = 100),
        
        sliderInput("base_level", "Base level:",
                    min = 0, max = 50,
                    value = 0),
        
        sliderInput("rows", "Row-wise Nodes:",
                    min = 5, max = 30,
                    value = 10),
        
        sliderInput("cols", "Column-wise Nodes:",
                    min = 5, max = 30,
                    value = 10),
        
        numericInput("min_change", "Minimum Change",value = 0.1, min = 1*10^-12, max = 1),
        actionButton("Go", "Run / Update Simulation")),
      
        mainPanel(
          tabsetPanel(
            tabPanel("Flow Visualisation",imageOutput("finiteDifferenceGif") %>% withSpinner(color="#0dc5c1")),
            tabPanel("Starting Conditions",DTOutput("finiteDifference_initial")),
            tabPanel("Final Conditions",DTOutput("finiteDifference_final"))
          )
        )
      )
    )

server <- function(input,session, output) {
   
  finite_diff.func <- function(rows,cols,fixed_side = "left",head_value = 15, base_value = 1.5,min_change = 0.01) {
    
    flow_field.mat <- matrix(data = rep(0,rows*cols),nrow = rows, ncol = cols) #rep(rows*cols,0) #rnorm(rows*cols)
    flow_field.mat[1,] <- rep(head_value,ncol(flow_field.mat)) # replacing top values of matrix
    flow_field.mat[,ncol(flow_field.mat)] <- rep(head_value,nrow(flow_field.mat)) # replacing right hand side values of matrix w/higher water level
    flow_field.mat[nrow(flow_field.mat),1] <- base_value # setting lower water level
    
    message("Initial flow field")
    print(flow_field.mat) # show initial state
    flow_inital <- flow_field.mat
    #write(flow_field.mat, stdout())
    
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
  
  results <- eventReactive(input$Go, {
    finite_diff.func(head_value = input$head_level,
                     base_value = input$base_level,
                     min_change = input$min_change,
                     rows = as.integer(input$rows),
                     cols = as.integer(input$cols))
  })
  
  output$finiteDifference_initial <- renderDT(
    datatable(results()[[2]],colnames = NULL,
    options = list(scrollX = TRUE,dom = 't',bSort=FALSE))
  )
  
  output$finiteDifference_final <- renderDT(
    datatable(round(results()[[3]],2),colnames = NULL,
    options = list(scrollX = TRUE,dom = 't',bSort=FALSE))
  )
  
  output$finiteDifferenceGif <- renderImage({
    outfile <- tempfile(fileext = '.gif')
    g <- ggplot(data = results()[[1]]) +
      geom_tile(mapping = aes(x = x, y = y,fill = flow)) +
      transition_time(iteration) +
      labs(title = "Finite Difference Model of Water Flow", subtitle = "Iteration: {frame_time}") +
      theme_light() +
      scale_fill_viridis_c() +
      ease_aes("linear")
    
    anim_save("outfile.gif",animate(g))
    
    # Return a list containing the filename
    list(src = "outfile.gif",
         contentType = 'image/gif'
         # width = 400,
         # height = 300,
         # alt = "This is alternate text"
    )}, deleteFile = TRUE)
  
  #session$onSessionEnded(stopApp)
  
  }
# Run the application 
shinyApp(ui = ui, server = server)

#library(rsconnect)
#rsconnect::setAccountInfo()

#deployApp()

