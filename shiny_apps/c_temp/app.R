library(rvest) # scraping MoH
#library(rgdal)
#library(sp)
#library(leaflet)
library(shiny) # app
library(reshape2) # melting
library(dplyr) # wrangling
#library(rgeos)
library(magrittr) # piping
library(ggplot2) # core plot package
library(viridis) # nice colour scale
library(forcats) # factors
library(plotly) # interactive viz
library(shinythemes) # for web theme
library(DT) # for interactive data tables
library(readr) # to nicely read in data
library(lubridate) # now()

# set timezone

# set text size on x axis
text_size = 8

# https://datascott.com/blog/subtitles-with-ggplotly/

# https://datafinder.stats.govt.nz/search/?q=territorial%2Bclipped
# WGS 84 (EPSG:4326 Geographic)
# nz.sdf <- readOGR(dsn = ".", layer = "territorial-authority-2020-clipped-generalised")
# taCentroids <- gCentroid(nz.sdf,byid=TRUE)
# 
# nz.sdf@data$TA2020_V_1[1:67]
# 
# leaflet(data = nz.sdf) %>% 
#     addTiles() %>% 
#     addPolygons() %>%
#     setView(lng = 172.8859711, lat = -39.9005585, zoom = 5) %>%
#     addCircleMarkers(data = taCentroids,
#                      radius = ~ sqrt(quantity))

# Define UI for application 
## UI -----------
ui <- fluidPage(theme = shinytheme("simplex"),
                tags$head(
                    tags$meta(charset = "UTF-8"),
                    tags$meta(name = "description", content = "COVID19 Data from NZ - Plots and Tables"),
                    tags$meta(name = "keywords", content = "New Zealand, NZ, COVID19,Coronavirus,Data,Data Visualisation"),
                    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1.0"),
                    tags$meta(name = "og:image", content = "titlecard.jpg"),
                    tags$meta(name = "og:url", content = "https://mks29.shinyapps.io/covid_nz/"),
                    tags$meta(name = "og:type", content = "website"),
                    tags$meta(name = "og:title", content = "COVID19 NZ Shiny App")
                ),
                
                # setup shinyjs
                #useShinyjs(),
                
                ## Application title -------------
                titlePanel(paste("New Zealand COVID19 Cases")), # : ",as.Date(Sys.time() + 13*60*60),"(GMT+13)" #adjust +13 hours for GMT+13 in NZ
                h3("Check the Ministry of Health website for the most up-to-date information"),
                h5("Data Source: New Zealand Ministry of Health. Data includes both confirmed and probable cases."),
                #h5("Time Series Data Source: University of Hopkins Systems Science and Engineering Unit (pulls from World Health Organisation and other sources)"),
                #h5("WHO data will have a 1-2 day lag against Ministry of Health data"),
                
                
                # Header buttons --------------
                fluidRow(
                    column(3,
                           wellPanel(
                               actionButton(inputId = "updateButton",
                                            label = "Update")
                           )
                    ),
                    column(3,
                           wellPanel(
                               actionButton(inputId = "mohLink",
                                            label = "Ministry of Health Cases Page",
                                            onclick = "window.open('https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-current-cases', '_blank')")
                               #uiOutput("tab")
                           )
                    ),
                    column(3,
                           wellPanel(
                               downloadButton(outputId = "download",
                                              label = "Download Raw Case Data")
                               #uiOutput("tab")
                           )
                    ),
                    column(3,
                           wellPanel(
                               downloadButton(outputId = "download_ts",
                                              label = "Download Raw Time Series Data")
                               #uiOutput("tab")
                           )
                    )
                ),
                # header Table ----------
                fluidRow(
                    #tags$style(type = "text/css", "#info {white-space: pre-wrap;}"),
                    column(12,dataTableOutput("info"))  #textOutput("info") #DT::dataTableOutput("info")
                ),
                ## tabs ------------
                wellPanel(
                    fluidRow(
                        tabsetPanel(type = "tabs",
                                    tabPanel("Time Series",
                                             # line plot
                                             tags$br(),
                                             fluidRow(
                                                 column(6,plotlyOutput("tsPlot",height = 600)),
                                                 column(6,plotlyOutput("newcasesPlot",height = 600))
                                             )
                                    ),
                                    # tabPanel("Recoveries",
                                    #          #stacked barplots
                                    #          tags$br(),
                                    #          fluidRow(
                                    #              column(12,plotlyOutput("statusPlot",height = 600))
                                    #          )
                                    #),
                                    tabPanel("Bivariate",
                                             #stacked barplots
                                             tags$br(),
                                             fluidRow(
                                                 column(12,plotlyOutput("mainPlot",height = 600))
                                             ),
                                             fluidRow(
                                                 column(12,plotlyOutput("mainPlot2",height = 600))
                                             ),
                                             fluidRow(
                                                 column(12,plotlyOutput("mainPlot3",height = 600))
                                             )
                                    ),
                                    tabPanel("Univariate",
                                             # simple barplots
                                             tags$br(),
                                             fluidRow(
                                                 column(6,plotlyOutput("agePlot",height = 600)),
                                                 column(6,plotlyOutput("genderPlot",height = 600))
                                             ),
                                             fluidRow(
                                                 column(12,plotlyOutput("regionPlot",height = 600))
                                             )
                                    ),
                                    tabPanel("Additional Tables",
                                             tags$br(),
                                             fluidRow(
                                                 column(12,DT::dataTableOutput("regionTable"))
                                             ),
                                             fluidRow(
                                                 column(12,DT::dataTableOutput("ageTable"))
                                             ),
                                             fluidRow(
                                                 column(12,DT::dataTableOutput("genderTable"))
                                             )
                                    ),
                                    tabPanel("Raw Data",
                                             tags$br(),
                                             DT::dataTableOutput("rawData")
                                    ),
                                    tabPanel("About",
                                             tags$br(),
                                             uiOutput("about")
                                    )
                        )
                    )
                )
)

# Define server logic required to draw a histogram
server <- function(input, output,session) {
    
    rv <- reactiveValues()
    rv$run <- 0
    ## Main Scraping and Dataframe ------------
    covid_ts.df <- eventReactive(eventExpr = c(input$updateButton,rv),
                                 valueExpr = {
                                     # url <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
                                     # covid_ts.df <- read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
                                     # covid_ts.df <- covid_ts.df %>% filter(`Country/Region` == "New Zealand") %>% select(-c(Lat,Long,`Province/State`)) 
                                     # covid_ts.df <- covid_ts.df %>% rename(Country = `Country/Region`)
                                     # coivd_ts.df <- melt(covid_ts.df)
                                     # covid_ts.df <- coivd_ts.df %>% filter(value != 0)
                                     
                                     #save(covid_ts.df,file = "covid_ts.df")
                                     #load("covid_ts.df")
                                     #covid_ts.df$variable <- as.character(covid_ts.df$variable)
                                     #covid_ts.df$value[25] <- 102
                                     #covid_ts.df <- rbind(covid_ts.df,c("New Zealand","3/24/20",155))
                                     covid_ts.df <- read.csv(file = "covid_ts.csv",header = T)
                                     
                                     covid_ts.df$variable <- as.factor(covid_ts.df$variable)
                                     covid_ts.df$value <- as.numeric(covid_ts.df$value)
                                     covid_ts.df
                                     
                                     covid.lag <- lag(covid_ts.df$value,1)
                                     covid.lag[is.na(covid.lag)] <- 0
                                     
                                     covid_ts.df$new_cases <- covid_ts.df$value - covid.lag
                                     
                                     covid_ts.df
                                     
                                    # write.csv(x = covid_ts.df,file = "covid_ts.csv",quote = F,row.names = F)
                                 })
    
    covid.df <- eventReactive(eventExpr = c(input$updateButton,rv),
                              valueExpr = {
                                  # data gen
                              		#download.file("https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-current-cases",destfile="t.html")
                                  
                                  url <- "https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-current-cases"
                                  #<- url %>%
                                  
                                  
                                  covid.ls <- read_html(url) %>% # "23_03_2020.html" # for static 
                                      html_table()
                                  
                                  covid.df <- covid.ls[[1]]
                                  covid_p.df <- covid.ls[[2]]
                                  
                                  covid_p.df$Case <- paste(covid_p.df$Case,"probable")
                                  
                                  covid.df <- rbind(covid.df,covid_p.df)
                                
                                  levels(covid.df$Gender)[levels(covid.df$Gender) == ""] <- "Not Reported"
                                  levels(covid.df$Location)[levels(covid.df$Location) == ""] <- "Not Reported"
                                  levels(covid.df$Age)[levels(covid.df$Age) == ""] <- "Not Reported"
                                  
                                  covid.df <- covid.df %>%  mutate(Gender = recode(Gender, 
                                                                                   `Male` = "M",
                                                                                   `Female` = "F"))
                                  
                                  
                                  # sort levels by frequency of location
                                  covid.df$Location <- fct_recode(covid.df$Location, c("Hawkes Bay" = "Hawke’s Bay")) 
                                  covid.df$Location <- fct_infreq(covid.df$Location, ordered = NA)
                                  covid.df$Age <- fct_recode(covid.df$Age, c("60s" = "64")) #fct_infreq(covid.df$Age, ordered = NA)         
                                  #write.csv(covid.df,"covid19.csv")
                                  #write.csv(covid.df,"covid19.csv")
                                  covid.df$Age <- fct_relevel(covid.df$Age, c("Child","Teens","20s","30s","40s","50s","60s","70s")) #fct_infreq(covid.df$Age, ordered = NA)                                  #write.csv(covid.df,"covid19.csv")
                                  covid.df
                              })
    ## Time Series -------------------
    output$tsPlot <- renderPlotly({
        
        ## Cumulative Time Series ---------------
        ts.df <- covid_ts.df()
        
        # recode dates
        ts.df$variable <- as.Date(ts.df[,2],
                                  format = "%m/%d/%y")
        
        ts.g <- ggplot(data = ts.df) +
            geom_line(mapping = aes(x = variable,y = value,group = 1)) + # reorder(covid_main.df$Location,left_join(covid_main.df,order.df)$order)
            geom_point(mapping = aes(x = variable,y = value,group = 1)) +
            labs(title = "New Zealand COVID19 cases: Time Series (Cumulative)",subtitle = paste(Sys.time(),Sys.timezone()),x = "Date",y = "Cumulative Number of cases") +
            theme_bw() +
            #annotate(geom = "text", x = 1, y = max(ts.df$value)/2, label = paste0("N = ",nrow(ts.df)),color = "black") +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1,size = text_size)) +
        	scale_x_date(breaks = seq(min(ts.df$variable), max(ts.df$variable), by = "2 day"), minor_breaks = "1 day") #+
            #geom_text(data = tail(ts.df),aes(x = variable - 0.5,y = value + max(new_cases)/20,label = value))
        #scale_x_date(breaks = ts.df$variable[seq(1, length(ts.df$variable), by = 3)])
        
        ts.g %>% 
            ggplotly() %>% #tooltip = c("Number of cases")
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('NZ COVID19 cases - Time Series',
                                              '<br>',
                                              '<sup>',
                                              "Current to 24/03/2020 (showing values for last 5 days)",
                                              '</sup>')))
    })
    
    ## New Cases Time Series -------------------
    output$newcasesPlot <- renderPlotly({
        nc.df <- covid_ts.df()
        
        # recode dates
        nc.df$variable <- as.Date(nc.df[,2],
                                  format = "%m/%d/%y")
        
        nc.g <- ggplot(data = nc.df) +
            #geom_line(mapping = aes(x = variable,y = new_cases,group = 1)) + # reorder(covid_main.df$Location,left_join(covid_main.df,order.df)$order)
            geom_col(mapping = aes(x = variable,y = new_cases,group = 1)) +
            labs(title = "NZ COVID19: New cases",subtitle = paste(Sys.time(),Sys.timezone()),x = "Date",y = "Number of new cases") +
            theme_bw() +
            #annotate(geom = "text", x = 1, y = max(ts.df$value)/2, label = paste0("N = ",nrow(ts.df)),color = "black") +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1,size = text_size)) +
            scale_x_date(breaks = seq(min(nc.df$variable), max(nc.df$variable), by = "2 day"), minor_breaks = "1 day") #+
            #geom_text(data = tail(nc.df),aes(x = variable,y = new_cases + max(new_cases)/20,label = new_cases))
        #scale_x_date(breaks = ts.df$variable[seq(1, length(ts.df$variable), by = 3)])
        
        nc.g %>% 
            ggplotly() %>% #tooltip = c("Number of cases")
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('NZ COVID19 cases: New Cases',
                                              '<br>',
                                              '<sup>',
                                              "Current to 24/03/2020 (showing values for last 5 days)",
                                              '</sup>')))
    })
    ## Stacked Bar Charts ----------------
    output$mainPlot <- renderPlotly({
        covid_main.df <- covid.df() %>%
            group_by(Age,Location) %>%
            summarise(n = length(Case))
        
        main.g <- ggplot(data = covid_main.df) +
            geom_col(mapping = aes(x = Location,y = n,fill = Age)) + # reorder(covid_main.df$Location,left_join(covid_main.df,order.df)$order)
            labs(title = "NZ COVID19 cases - Region and Age",subtitle = paste(Sys.time(),Sys.timezone()),x = "Region",y = "Number of cases") +
            scale_fill_viridis(discrete = T) +
            theme_light() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust  = 1,,size = text_size))
        
        main.g %>% 
            ggplotly(tooltip = c("Region","Age","n")) %>% 
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('New Zealand COVID19 cases - Region and Age',
                                              '<br>',
                                              '<sup>',
                                              Sys.time() + 13*60*60,
                                              '</sup>')))
    })
    output$mainPlot2 <- renderPlotly({
        covid_main.df <- covid.df() %>%
            group_by(Age,Gender) %>%
            summarise(n = length(Case))
        
        main.g <- ggplot(data = covid_main.df) +
            geom_col(mapping = aes(x = Age,y = n,fill = Gender)) + # reorder(covid_main.df$Location,left_join(covid_main.df,order.df)$order)
            labs(title = "NZ COVID19 cases - Age and Gender",subtitle = paste(Sys.time(),Sys.timezone()),x = "Age",y = "Number of cases") +
            scale_fill_viridis(discrete = T) +
            theme_light() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust  = 1,,size = text_size))
        
        main.g %>% 
            ggplotly(tooltip = c("Gender","n")) %>% 
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('NZ COVID19 cases - Age and Gender',
                                              '<br>',
                                              '<sup>',
                                              Sys.time() + 13*60*60,
                                              '</sup>')))
    })
    output$mainPlot3 <- renderPlotly({
        covid_main.df <- covid.df() %>%
            group_by(Location,Gender) %>%
            summarise(n = length(Case))
        
        main.g <- ggplot(data = covid_main.df) +
            geom_col(mapping = aes(x = Location,y = n,fill = Gender)) + # reorder(covid_main.df$Location,left_join(covid_main.df,order.df)$order)
            labs(title = "NZ COVID19 cases by Region and Gender",subtitle = paste(Sys.time(),Sys.timezone()),x = "Region",y = "Number of cases") +
            scale_fill_viridis(discrete = T) +
            theme_light() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust  = 1,,size = text_size))
        
        main.g %>% 
            ggplotly(tooltip = c("Gender","n")) %>% 
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('NZ COVID19 cases - Region and Gender',
                                              '<br>',
                                              '<sup>',
                                              Sys.time() + 13*60*60,
                                              '</sup>')))
    })
    ## Plot - Age -------------------
    output$agePlot <- renderPlotly({
        covid_age.df <- covid.df() %>%
            group_by(Age) %>%
            summarise(n = length(Case))
        
        age.g <- ggplot(data = covid_age.df) +
            geom_col(mapping = aes(x = Age,y = n,fill = Age)) + # reorder(covid_age.df$Age, -n)
            labs(title = "NZ COVID19 cases - Age",subtitle = paste(Sys.time(),Sys.timezone()),x = "Age",y = "Number of cases") +
            scale_fill_viridis(discrete = T) +
            theme_light() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust  = 1,,size = text_size))
        
        age.g %>% ggplotly(tooltip = c("Age","n")) %>% 
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('New Zealand COVID19 cases by Age',
                                              '<br>',
                                              '<sup>',
                                              Sys.time() + 13*60*60,
                                              '</sup>')))
    })
    ## Plot - Region -------------------
    output$regionPlot <- renderPlotly({
        covid_region.df <- covid.df() %>%
            group_by(Location) %>%
            summarise(n = length(Case)) 
        
        region.g <- ggplot(data = covid_region.df) +
            geom_col(mapping = aes(x = reorder(covid_region.df$Location, -n),y = n,fill = Location,)) +
            labs(title = "NZ COVID19 cases - Region",subtitle = paste(Sys.time(),Sys.timezone()),x = "Location",y = "Number of cases") +
            theme_light() +
            scale_fill_viridis(discrete = T) +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust  = 1,,size = text_size))
        
        region.g %>% 
            ggplotly(tooltip = c("Location","n")) %>% 
            config(displayModeBar = F) %>% 
            layout(title = list(text = paste0('NZ COVID19 cases - Region',
                                              '<br>',
                                              '<sup>',
                                              Sys.time() + 13*60*60,
                                              '</sup>')))
    })
    ## Plot - Gender -------------------
    output$genderPlot <- renderPlotly({
        covid_gender.df <- covid.df() %>%
            group_by(Gender) %>%
            summarise(n = length(Case)) 
        
        gender.g <- ggplot(data = covid_gender.df) +
            geom_col(mapping = aes(x = reorder(covid_gender.df$Gender, -n),y = n,fill = Gender)) +
            labs(title = "NZ COVID19 cases - Gender",subtitle = paste(Sys.time(),Sys.timezone()),x = "Gender",y = "Number of cases") +
            scale_fill_viridis(discrete = T) +
            theme_light() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust  = 1,,size = text_size))
        
        gender.g %>% ggplotly(tooltip = c("Gender","n")) %>% 
            config(displayModeBar = F) %>%
            layout(title = list(text = paste0('New Zealand COVID19 cases by Gender',
                                              '<br>',
                                              '<sup>',
                                              Sys.time() + 13*60*60,
                                              '</sup>')))
    })
    ## Info -------------------
    
    # manually filled for now
    cases.df <- data.frame(`Total(Confirmed+Probable)` = as.integer(155),
                           Confirmed = as.integer(142),
                           Probable = as.integer(13),
                           Recovered = as.integer(12),
                           Community = as.integer(4))
    # colnames(cases.df) <- c("Total Cases" = "Total_Cases",
    #                         "Confirmed" = "Confirmed",
    #                         "Probable" = "Probable",
    #                         "Recovered" = "Recovered",
    #                         "Community" = "Community")
    
    output$info <- DT::renderDataTable({ 
        #DT::renderDataTable({ 
        #DT::dataTableOutput(cases.df)
        DT::datatable(cases.df, options = list(dom = 't'),rownames = FALSE,
                      colnames = c("Total Cases (Confirmed + Probable)","Confirmed","Probable","Recovered","Community"))
        #renderText({
        #paste("Confirmed cases:",nrow(covid.df()),"\n","Recovered cases: ",12,"\n","Community transmission cases:",4)
    })
    ## Tables -------------------
    output$rawData = DT::renderDataTable({
        DT::datatable(covid.df(),options = list(
            pageLength = 60))
    })
    output$regionTable = DT::renderDataTable({
        
        covid_region.df <- covid.df() %>%
            group_by(Location) %>%
            summarise(n = length(Case)) 
        
        DT::datatable(covid_region.df,options = list(
            pageLength = 60))
    })
    output$genderTable = DT::renderDataTable({
        
        covid_gender.df <- covid.df() %>%
            group_by(Gender) %>%
            summarise(n = length(Case)) 
        
        DT::datatable(covid_gender.df,options = list(
            pageLength = 60))
    })
    output$ageTable = DT::renderDataTable({
        
        covid_age.df <- covid.df() %>%
            group_by(Age) %>%
            summarise(n = length(Case))
        
        DT::datatable(covid_age.df,options = list(
            pageLength = 60))
    })
    
    # output$tab <- renderUI({
    #     url <- a("Ministry of Health Cases Page", href = "https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-current-cases")
    #     tags$div(class = "submit",
    #              tags$a(href = url, 
    #                     "Learn More", 
    #                     target = "_blank")
    #     )
    #     tagList("URL link:", url)
    # })
    
    # Downloadable csv of cases dataset ----
    output$download <- downloadHandler(
        filename = function() {
            paste("covid_19_cases_nz_",as.numeric(Sys.time()),".csv", sep = "")
        },
        content = function(file) {
            write.csv(covid.df(), file, row.names = FALSE)
        }
    )
    # Downloadable csv of time series dataset ----
    output$download_ts <- downloadHandler(
        filename = function() {
            paste("covid_19_timeseries_nz_",as.numeric(Sys.time()),".csv", sep = "")
        },
        content = function(file) {
            write.csv(covid_ts.df(), file, row.names = FALSE)
        }
    )
    
    output$about <- renderUI({
        HTML('Source Code: <a href = "https://github.com/MattSkiff/covid19_nz_data">Shiny App GitHub Repo</a><br> 
              Source MoH data: <a href = "https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-current-cases">Ministry of Health Confirmed Cases (web tables)</a><br>')
       # Source Time Series data: <a href = "https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv">John Hopkins University Centre for Systems Science and Engineering - Time Series Data Source</a><br>')       
    })
    
    # Trick file date creation update
    onStop(function() {
        
        # File name
        p <- paste0(getwd(), "/app.R")
        
        # Update file 'date creation'
        Sys.setFileTime(p, now())
        
    }) # onStop
}

# covid.ls <- url %>%
# 	read_html() %>%
# 	# html_nodes(xpath='//*[@id="mw-content-text"]/table[1]') %>%
# 	html_table()
# 
# covid.df <- covid.ls[[1]]
# nrow(covid.df)

# Run the application 
shinyApp(ui = ui, server = server)