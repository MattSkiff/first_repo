library(shiny)
library(ggplot2)
library(rgl)

#setwd("\\\\homes.cms.waikato.ac.nz/people/mks29/win/Desktop/shiny_app_demo")

ui <- fluidPage(
  
  # App title ----
  titlePanel("Simple Distributions Example"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Slider for the number of bins ----
      selectInput(inputId = "distribution_selector",
                  label = "Choose a distribution!:",
                  choices = c("Normal", "Poisson", "Binomial", "Beta")),
      
    #  sliderInput(inputId = "n", "Number of Observations", 1000, 10000, 0, step = 1000, round = FALSE,
    #              ticks = TRUE, animate = FALSE, sep = ",")
    #  #animationOptions(interval = 1000, loop = TRUE)
    #),
    actionButton("start_dist", "Start Loop"),
    actionButton("stop_dist", "Stop Loop"),
    h3(textOutput(outputId = "sampleSize_caption"))),
    
    # Main panel for displaying outputs ----
    mainPanel(
      # Output: Histogram ----
      plotOutput(outputId = "distPlot")
    )
  )
)

server <- function(input, output,session) {
  # Selecting distributions every 5s
  choices = c("Normal", "Poisson", "Binomial", "Beta")
  
  my<-reactiveValues(inc=0, timer=reactiveTimer(3000), started=FALSE)
  
    observeEvent(input$stop_dist, {my$started<-FALSE})
    observeEvent(input$start_dist, {my$started<-TRUE})
  
  # Looping through sample sizes
  values <- reactiveValues(n = 0,y = 1,dist = 1)
  observe({
    my$timer()
    if(isolate(my$started)) {
    if (isolate(values$y)  == 6) {
      values$y = 1
      values$n  = 1*10^isolate(values$y)
      if (isolate(values$dist) == 4) {
        values$dist = 0
      }
      values$dist = values$dist + 1
      updateSelectInput(session, "distribution_selector", selected = choices[isolate(values$dist)])
    } else {
      values$y = isolate(values$y) + 1
      values$n  = 1*10^isolate(values$y)
    }
    }
  })
  
  output$sampleSize_caption <- renderText({
    paste("Sample Size: ",values$n)
  })
  
  output$distPlot <- renderPlot({
    # generate data
    if (input$distribution_selector == "Normal"){
      data <- as.data.frame(rnorm(values$n,0,1))
    } else if (input$distribution_selector == "Poisson") {
      data <- as.data.frame(rpois(values$n,10))
    } else if (input$distribution_selector == "Binomial") {
      data <- as.data.frame(rbinom(values$n,values$n*3,0.5))
    } else if (input$distribution_selector == "Beta") {
      data <- as.data.frame(rbeta(values$n,5,1))
    } 
    colnames(data) <- c("x")
    k <- 1 + 3.22*log(length(data$x)) # sturge's rule
    #plot
    ggplot(data = data) + 
      geom_histogram(mapping = aes(x = x),bins = k) +
      ggtitle(input$distribution_selector) + 
      xlim(min(data$x),max(data$x)) + 
      theme_light()
  })
  
  session$onSessionEnded(stopApp)
}

shinyApp(ui, server)

