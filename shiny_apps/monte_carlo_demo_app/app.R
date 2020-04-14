library(shiny) # framework
library(ggplot2) # for graphics
library(ggforce) # for better circles in ggplot2
# APDAPTED FROM https://www.r-bloggers.com/estimation-of-the-number-pi-a-monte-carlo-simulation/
#setwd("\\\\homes.cms.waikato.ac.nz/people/mks29/win/Desktop/shiny_app_monte")
options(bitmapType='cairo') 
shinyOptions(shiny.trace = FALSE)
shinyOptions(shiny.reactlog = FALSE)
shinyOptions(shiny.stacktraceoffset = FALSE)

ui <- fluidPage(
  withMathJax(),
  tags$style(type = "text/css",
             ".recalculating {opacity: 1.0;}"),
  
  fluidRow(column(6,wellPanel(h3(
    "Estimating Pi using Monte Carlo Simulation"))),
          br(),
           column(6,h5(textOutput(outputId = "Information")))),
  
  fluidRow(column(3,
                  h5(textOutput(outputId = "Information2"))),
           column(3,
                  h5(textOutput(outputId = "Information3"))),
           column(3,
                  h5(textOutput(outputId = "Information4"))),
           column(3,
                  h5('$$A = \\pi r^2$$'),
                  h5('$$\\pi = \\frac{A}{r^2}$$'))),
  tags$style(type="text/css", "#foo {white-space: pre-wrap;}"),
  fluidRow(column(2,
    br(),
    actionButton("start_dist", "Start / Continue Loop")),
  column(2,
         br(),
         actionButton("stop_dist", "Pause Loop")),
  column(4,
         sliderInput(inputId = "num_points_slider",label = "Choose Number of Points to Generate (pause first!)",min = 0,max = 100000,step = 1000,value = 10)),
  column(4,
         br(),
         actionButton("gen_button","Generate"),
  helpText("100k points will take around 30s (simulation function not vectorised + R is not a graphics engine)"))),#,
  #column(2,)),
  fluidRow(column(6,
                  h5(textOutput(outputId = "info1"))),
           column(6, h5(textOutput(outputId = "info2")))),
  fluidRow(column(6, h5(textOutput(outputId = "info3"))),
  column(6, h5(textOutput(outputId = "info4")))),
  br(),
  hr(),
  fluidRow(column(
    6,
    plotOutput(
      outputId = "piPlot",
      #width = "600px",
      height = "100%"
    )
  ),
  column(
    6,
    plotOutput(
    outputId = "convergPlot",
    #width = "1200px",
    height = "100%"
    )),
  fluidRow(column(12,h3("")))
))

server <- function(input, output,session) {
    observeEvent(input$stop_dist, {
      rand$started <- FALSE
    })
    observeEvent(input$start_dist, {
      rand$started <- TRUE
    })
    rand <- reactiveValues(i = 0, df = setNames(data.frame(matrix(ncol = 2, nrow = 2,runif(4,-1,1))), c("x", "y")), timer = reactiveTimer(2000), started = FALSE) # carries timer, data and start status
    gen_points <- function(x) { # x vector, specifies no. df (length of vec) + length of each df / 2 (size of each element in vector), returns list of df with rand points 
      l <- list()
      for (i in 1:length(x)) {
        l[[i]] <- setNames(data.frame(matrix(ncol = 2, nrow = x[i]/2 ,runif(x[i],-1,1))), c("x", "y"))
      }
      return(l)
    }
    simulation = function(data){ # must be passed in as DF (not list)
      c = rep(0,nrow(data)/2)
      numberIn = 0
      for(i in 1:nrow(data)){
        if(sqrt(sum(data[i,]^2)) <= 1) { # euclid distance of point from 0,0
          numberIn = numberIn + 1
        }
        prop = numberIn / i # want proportion of points falling into circle
        piHat = prop*4 # as area of square is 2x2
        c[i] = piHat
      }
      return(c) # returns vector
    }
    
    sampleSizes <- c(10,50,100,150,200,300,400,500,750,1000,1250,1500,1750,2000,2500,3000,4000,5000)
    data <- gen_points(sampleSizes)
    

    observe({ # loops the data sets 
      if (isolate(rand$i) == 0) { # regenerating random data on each loop
        sampleSizes <- c(10,50,100,150,200,300,400,500,750,1000,1250,1500,1750,2000,2500,3000,4000,5000)
	  data <- gen_points(sampleSizes) 
      }
      rand$timer()
      if (rand$started) {
        if (isolate(rand$i) == length(sampleSizes)) {
          rand$i = 0
          rand$df <- isolate(rand$df[1:2,])
        } else {
          rand$i = isolate(rand$i) + 1
          rand$df <- data[[isolate(rand$i)]]
        }
      }
    })
    
    theme_set(theme_bw())
    theme_update(text = element_text(size=12),
                 panel.grid.major = element_blank(),
                 panel.grid.minor = element_blank(),
                 strip.background = element_blank())
    
    # gen points
    observeEvent(input$gen_button, {
      df <- gen_points(input$num_points_slider*2)
      rand$df <- df[[1]]
    })
    
    output$piPlot <- renderPlot({
      message('Rendering Pi Plot')
      df <- rand$df
      if (nrow(df) <= 5000) {
      ggplot(df) + 
        geom_circle(aes(x0 = 0, y0 = 0, r = 1)) +
        geom_point(mapping = aes(x = x, y = y,col = ifelse(sqrt((x^2)+(y^2)) <= 1,'blue','red')), size = 3,shape = 16) + # colours based on dist from point - using origin of 0,0
        geom_rect(mapping = aes(xmin = -1,xmax = 1,ymin = -1,ymax = 1),fill = NA,size = 0.5,color = 'red') +
        labs(title = "Randomly generating data within square",caption = "Finding proportion inside circle, versus outside") + 
        coord_cartesian(ylim = c(-1, 1),xlim = c(-1,1)) +
        scale_color_manual(labels = c("In", "Out"), values = c("blue", "red")) +
        theme(legend.title = element_blank(),
              legend.position = c(0.8, 0.2),
              legend.spacing.y = unit(0, "mm"),
              legend.background = element_rect(fill='white',size=0.5, linetype="solid", colour ="black")) 
      } else {
        ggplot(data.frame()) + 
          geom_point() + 
          xlim(-1,1) + 
          ylim(-1,1) + 
          annotate("text",x = 0, y = 0,label = "Too much data to render quickly!")
      }
      #}
    }, height = function() {
      session$clientData$output_piPlot_width
    })
# hat-tip to jcheng5 https://github.com/rstudio/shiny/issues/650)
    
    info <- reactiveValues(in_circle = 0, out = 0,pi = 0)
    
    output$convergPlot <- renderPlot({
      df <- rand$df
      res = as.data.frame(simulation(df)) # ggplot needs df
      colnames(res) <- "pi"
      info$in_circle = sum(ifelse(sqrt((df$x^2)+(df$y^2)) <= 1,1,0))
      info$out = nrow(rand$df) - info$in_circle
      info$pi = res$pi[nrow(res)]
      res$x <- seq(1:nrow(res))
      ini = 1
      ggplot(res) +
        geom_line(mapping = aes(y = pi, x = x)) +
        geom_hline(yintercept = pi,colour = 'red') + 
        labs(title = "Convergence of Pi",
             caption = paste("For Random Sample of n = ",nrow(res))) +
        ylab("Estimate of Pi") +
        xlab("Iterations (Number of Points)") +
        #coord_cartesian(ylim=c(pi-2, pi+2)) + # doesn't crop data
        theme(legend.position = "none") 
      
    }, height = function() {
      session$clientData$output_convergPlot_width
    })
    # hat-tip to jcheng5 https://github.com/rstudio/shiny/issues/650)
    
    output$Information <- renderText({
      "Here we estimate Pi (an irrational number which is not known exactly), using simulation methods. 
      As we know the formula for the area of a circle (below), if we can estimate the area of the circle, and we know the radius, we can estimate Pi!"
    })
    output$Information2 <- renderText({
      paste("The area of the square containing the circle here is 2 x 2 = 4. We get an estimate of the circles area by comparing the proportion of points inside the circle to outside. This will approximate the proportion of the area of the circle to the square.")
    })
    output$Information3 <- renderText({
      paste("This proportion is (right now)  ",
            round(info$in_circle/(info$out+info$in_circle),3),". We then multiply this proportion by the area of the square to get an estimate of the area of the circle: ",
            round(info$in_circle/(info$out+info$in_circle)*4,3))
    })
    output$Information4 <- renderText({
      paste("The radius is 1, so this term disappears from the equation. This means the area of the circle = pi, which we estimate by randomly generating points inside the square below.")
    })
    output$info1 <- renderText({
      paste("Sample Size: ",nrow(rand$df))
    })
    output$info2 <- renderText({
      paste("Number In: ",info$in_circle,"| Number Out: ",info$out)
    })
    output$info3 <- renderText({
      paste("Estimate of Pi from Sample: ", round(info$pi,6)," | Error: ",abs(round(info$pi - pi,6))," (",round(abs((info$pi - pi)/pi)*100,6),"%)")
    })
    output$info4 <- renderText({
      paste("Actual value of Pi: ", round(pi,6))
    })
  }

shinyApp(ui, server)
#runApp(display.mode = "showcase") - use command in console to run in display mode