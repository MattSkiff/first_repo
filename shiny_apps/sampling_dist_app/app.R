library(shiny)
library(ggplot2) # heatmap
library(gganimate) # to animate ggplot2 heatmap
library(reshape2) # to melt dfs
library(DT) # for tables that don't trail off the pages...
library(shinycssloaders) # loader as viz takes a while to load
require(gifski) # needed for gganimate
require(png) # needed for gganimate

ui <- fluidPage(
  
    titlePanel("Sampling Distribution of the Mean Demonstration"),
    fluidRow(column(4,
    								plotOutput("small_sample_plot")),
    				 column(4,
    				 			 plotOutput("large_sample_plot")),
    				 column(4,
    				 			 plotOutput("data_plot"))
    				 ),
    fluidRow(column(4,
    								selectInput(label = "Choose Data Distribution", inputId = "d_choice", 
    														choices = c("Normal" = "norm",
    																				"Binomial" = "binom",
    																				"Poisson" ="pois",
    																				"Uniform" = "unif",
    																				"Beta" = "beta",
    																				"Gamma" = "gamma"
    																				), 
    														selected = "Normal", multiple = FALSE,
    														selectize = TRUE)),
    				 column(4,
    				 			 sliderInput(label = "Sample Size 1", inputId = "ss_2",
    				 			 						min = 0, max = 100,
    				 			 						value = 10)),
    				 column(4,
    				 			 sliderInput(label = "Sample Size 2",inputId =  "ss_1",
    				 			 						min = 100, max = 10000,
    				 			 						value = 500)),
   	fluidRow(column(12,
   									conditionalPanel(
   										condition = "input.d_choice == norm",
   										fluidRow(column(6,
   															numericInput("norm_mean", "Mean", 0, min = NA, max = NA)),
   														 column(6,
   															numericInput("norm_sd", "Standard Deviation", 1, min = NA, max = NA)))),
   									conditionalPanel(
   										condition = "input.d_choice == binom",
   										fluidRow(column(6,
   																		numericInput("binom_trials", "Mean", 0, min = NA, max = NA)),
   														 column(6,
   														 			 numericInput("binom_prob ", "Standard Deviation", 1, min = NA, max = NA)))),
   									conditionalPanel(
   										condition = "input.d_choice == pois",
   										fluidRow(column(6,
   																		numericInput("norm_mean", "Mean", 0, min = NA, max = NA)),
   														 column(6,
   														 			 numericInput("norm_sd", "Standard Deviation", 1, min = NA, max = NA)))),
   									conditionalPanel(
   										condition = "input.d_choice == unif",
   										fluidRow(column(6,
   																		numericInput("norm_mean", "Mean", 0, min = NA, max = NA)),
   														 column(6,
   														 			 numericInput("norm_sd", "Standard Deviation", 1, min = NA, max = NA)))),
   									conditionalPanel(
   										condition = "input.d_choice == beta",
   										fluidRow(column(6,
   																		numericInput("norm_mean", "Mean", 0, min = NA, max = NA)),
   														 column(6,
   														 			 numericInput("norm_sd", "Standard Deviation", 1, min = NA, max = NA)))),
   									conditionalPanel(
   										condition = "input.d_choice == gamma",
   										fluidRow(column(6,
   																		numericInput("norm_mean", "Mean", 0, min = NA, max = NA)),
   														 column(6,
   														 			 numericInput("norm_sd", "Standard Deviation", 1, min = NA, max = NA))))
				))
      )
)

server <- function(input,session, output) {
   
	output$small_sample_plot <- renderPlot({

	}, height = function() {
		session$clientData$output_small_sample_plot_width
	})
	
	output$large_sample_plot <- renderPlot({
		
	}, height = function() {
		session$clientData$output_large_sample_plot_width
	})
	
	output$data_plot <- renderPlot({
		
	}, height = function() {
		session$clientData$output_data_plot_width
	})
	
  finite_diff.func <- function() {}
  results <- eventReactive(input$Go, {
  })
  
  }
# Run the application 
shinyApp(ui = ui, server = server)

#library(rsconnect)
#rsconnect::setAccountInfo()

#deployApp()

