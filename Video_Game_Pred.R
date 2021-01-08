##########################################################
# Predict video game sales by rating
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")

#load libraries
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(corrplot)
library(ggplot2)
library(readr)
library(Rborist)
library(gbm)


#download and read in csv from gitHub
VidGam <- "https://github.com/AndrewPikeGIS/EDX_Training/raw/master/Video_Games_Sales_as_at_22_Dec_2016.csv"

#read in csv file from github
dfVG <- read.csv(VidGam)

#check to see if NA's in critic score is temporal.
sort(unique(dfVG$Year_of_Release[is.na(dfVG$Critic_Score)]))

#drop all null values
dfVG_Cl <- dfVG %>% drop_na() %>% select(Name, Platform, Year_of_Release, Genre, 
                                         Publisher, Critic_Score, Critic_Count, User_Score,
                                         User_Count, Developer, Rating, Global_Sales)
#convert global sales numbers to true sales (not per million)

dfVG_Cl<- dfVG_Cl %>% mutate(Global_Sales = Global_Sales *1000000)

#Central Stats for Global Sales
MuGl <-mean(dfVG_Cl$Global_Sales)
SDGl <-sd(dfVG_Cl$Global_Sales)
MinGl <- min(dfVG_Cl$Global_Sales)
MaxGL <- max(dfVG_Cl$Global_Sales)

GLStats <- data.frame(c(MuGl, SDGl, MinGl, MaxGL), row.names = c("MEAN", "SD", "MIN", "MAX"))%>%
  rename(Sales = c.MuGl..SDGl..MinGl..MaxGL.)
knitr::kable(GLStats)

#plot histogram for sales with binwidth at 1 million
dfVG_Cl %>% ggplot(aes(Global_Sales)) +geom_histogram(binwidth = 1000000) +scale_y_continuous(trans = "log10")

#split up platform into handheld, console, PC
Handheld <-c("3DS", "DS", "GBA", "PSP", "PSV", "GG") 
Console <- c("DC", "X360", "GC", "GEN", "N64", "NES", "NG", 
             "PCFX", "PS", "PS2", "PS3", "PS4", "Wii", "Wiiu",
             "XB", "XOne")
PC<-c("PC")

#Assign a system type value to each game
dfVG_Cl <-dfVG_Cl %>% 
  mutate(SystemType = case_when(
    Platform %in% Handheld ~ "Handheld",
    Platform %in% Console ~ "Console",
    Platform %in% PC ~ "PC",
    TRUE ~ "other"), 
    User_Score = as.numeric(User_Score),
    Year_of_Release = as.numeric(as.character(Year_of_Release)),
    Date = as.Date(paste(as.character(Year_of_Release), 1, 1, sep = "-"))
  )


#plot number of games released per publisher to total sales by publisher.
dfVG_Cl %>% group_by(Publisher) %>% 
  summarize(TotalSales = sum(Global_Sales), gameCount = n())%>%
  arrange(desc(TotalSales)) %>%
  ggplot(aes(x=TotalSales, y =gameCount)) + geom_point()

#create vector of sales factor, total sales of publisher divided by mean sales
StandSales <- dfVG_Cl %>% group_by(Publisher) %>% 
  summarize(Salesfact = (sum(Global_Sales)/MuGl))

#add SalesFact to data frame
dfVG_Cl<- left_join(dfVG_Cl, StandSales, by = "Publisher")

#create count for number of games released per publisher
GamesReleased <- dfVG_Cl %>% group_by(Publisher) %>% 
  summarize(GamesReleased = n())

#add GamesReleased to data frame
dfVG_Cl<- left_join(dfVG_Cl, GamesReleased, by = "Publisher")

#final N/A drop
dfVG_Cl <- dfVG_Cl %>% drop_na()

#run correlation plot for columns that are not factors
Cols <- unlist(lapply(dfVG_Cl, is.numeric))
X<- cor(dfVG_Cl[, Cols])
corrplot(X, method = "number")

#run pairs plot 
pairs(dfVG_Cl[,Cols], pch = 19, lower.panel = NULL)

#run sales over time by genre
dfVG_Cl %>% group_by(Year_of_Release, Genre, Date) %>%
  summarize(Sales = sum(Global_Sales)) %>% 
  ggplot() + geom_area(aes(x=Date, y = Sales, color = Genre, fill = Genre), position = "stack")


#plot number of games released per publisher to total sales by publisher.
dfVG_Cl %>% group_by(Publisher) %>% 
  summarize(TotalSales = sum(Global_Sales), gameCount = n())%>%
  arrange(desc(TotalSales)) %>%
  ggplot(aes(x=TotalSales, y =gameCount)) + geom_point()

#create RMSE calc function
RMSE <- function(true_Sales, predicted_Sales){
  sqrt(mean((true_Sales - predicted_Sales)^2))
}

# Validation set will be 10% of Video Game data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = dfVG_Cl$Global_Sales, times = 1, p = 0.1, list = FALSE)
VG <- dfVG_Cl[-test_index,]
Validation <-dfVG_Cl[test_index,]

#Run and test results from Systype, Genre, User Count and Critic Score model
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ SystemType + Genre  + User_Count + Critic_Score,
                   method = "gbm", data = VG,
                   trControl = trainControl(method = "cv", number = 5))

#create the RMSE test  dataframe to hold rmse and r squared r
RMSETest <- data.frame( RMSE = c(min(train_gbm$results$RMSE)),
                        R_2 = c(min(train_gbm$results$Rsquared)), 
                        row.names = "SYSTYPE_GENRE_UC_CS")

#print out results table
knitr::kable(RMSETest)


#Run and test results from Systype, Genre, User Score and Critic Score model
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ SystemType + Genre  + User_Score + Critic_Score,
                   method = "gbm", data = VG,
                   trControl = trainControl(method = "cv", number = 5))

#assign new results to dataframe
GBMNew <- data.frame(RMSE = min(train_gbm$results$RMSE),
                     R_2 = min(train_gbm$results$Rsquared), 
                     row.names = "SYSTYPE_GENRE_US_CS")


#combine new results with final RMSE test results
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)


#Run and test results from Systype, Genre, User Count and Critic Count model
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ SystemType + Genre  + User_Count + Critic_Count,
                   method = "gbm", data = VG,
                   trControl = trainControl(method = "cv", number = 5))

#store new results to dataframe
GBMNew <- data.frame(RMSE = min(train_gbm$results$RMSE),
                     R_2 = min(train_gbm$results$Rsquared), 
                     row.names = "SYSTYPE_GENRE_UC_CC")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)


#Run and test results from Platform, Genre, User Count and Critic Score model
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ Platform + Genre  + User_Count + Critic_Score,
                   method = "gbm", data = VG,
                   trControl = trainControl(method = "cv", number = 5))

#store new results to dataframe
GBMNew <- data.frame(RMSE = min(train_gbm$results$RMSE),
                     R_2 = min(train_gbm$results$Rsquared), 
                     row.names = "PLATFORM_GENRE_UC_CS")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)


#Run and test results from Platform, Genre, User Count, Critic Score and Year of release model
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ Platform + Genre  + User_Count + Critic_Score +Year_of_Release,
                   method = "gbm", data = VG,
                   trControl = trainControl(method = "cv", number = 5))

#join new results to final results table
GBMNew <- data.frame(RMSE = min(train_gbm$results$RMSE),
                     R_2 = min(train_gbm$results$Rsquared), 
                     row.names = "PLATFORM_GENRE_UC_CS_YR")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)


#Run and test results from Platform, Genre, User Count, 
#Critic Score and Year of release model using random forest model to test a different algorithm
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ Platform + Genre  + User_Count +
                     Critic_Score +Year_of_Release,
                   method = "Rborist", data = VG,
                   tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)))

#join new results to final results table
GBMNew <- data.frame(RMSE = min(train_gbm$results$RMSE),
                     R_2 = min(train_gbm$results$Rsquared), 
                     row.names = "PLATFORM_GENRE_UC_CS_YR_RF")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)


#Run and test results from Platform, Genre, User Count, Critic Score, Year of release model and rating
set.seed(1, sample.kind="Rounding")
train_gbm <- train(Global_Sales ~ Platform + Genre  +
                     User_Count + Critic_Score +Year_of_Release +Rating,
                   method = "gbm", data = VG,
                   trControl = trainControl(method = "cv", number = 5))

#join new results to final results table
GBMNew <- data.frame(RMSE = min(train_gbm$results$RMSE),
                     R_2 = min(train_gbm$results$Rsquared), 
                     row.names = "PLATFORM_GENRE_UC_CS_YR_RT")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)



#Run and test results from Platform, Genre, User Count, Critic Score, Year of release model, rating, SalesFactor and games published
set.seed(1, sample.kind="Rounding")
train_gbmGRSF <- train(Global_Sales ~ Platform + Genre  + User_Count +
                       Critic_Score +Year_of_Release +Rating +GamesReleased +Salesfact,
                     method = "gbm", data = VG,
                     trControl = trainControl(method = "cv", number = 5))

#join new results to final results table
GBMNew <- data.frame(RMSE = min(train_gbmGRSF$results$RMSE),
                     R_2 = min(train_gbmGRSF$results$Rsquared), 
                     row.names = "PLATFORM_GENRE_UC_CS_YR_RT_GR_SF")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest)

#Run and test results from Platform, Genre, User Count, Critic Score, Year of release model, rating and SalesStandardidation
set.seed(1, sample.kind="Rounding")
train_gbmSF <- train(Global_Sales ~ Platform + Genre  +
                       User_Count + Critic_Score +Year_of_Release +Rating +Salesfact,
                     method = "gbm", data = VG,
                     trControl = trainControl(method = "cv", number = 5))

#join new results to final results table
GBMNew <- data.frame(RMSE = min(train_gbmSF$results$RMSE),
                     R_2 = min(train_gbmSF$results$Rsquared), 
                     row.names = "PLATFORM_GENRE_UC_CS_YR_RT_SF")

#join new results to final results table
RMSETest <- rbind(RMSETest, GBMNew)

#Check results of all models
knitr::kable(RMSETest) 

#Using final model predict sales for Validation set
y_hat_gbm <- predict(train_gbmSF, Validation, type = "raw")

Pred <- as.data.frame(y_hat_gbm)

#check RMSE for final predicted global sales on validation set.
RMSE(Validation$Global_Sales, y_hat_gbm)

#create new dataframe to compare rmse
Compare <- data.frame(Name = Validation$Name,
                      Year = Validation$Year_of_Release,
                      Platform = Validation$Platform,
                      Sales = Validation$Global_Sales,
                      Predicted = y_hat_gbm) %>% 
                      mutate(Diff = Sales - Predicted, 
                             DiffA = abs(Diff),
                             no0 = ifelse(Predicted <0, 0, Predicted))

#plot predicted against true
qplot(Compare$Sales, Compare$Predicted)+geom_abline(intercept = 0, slope = 1)

#check for normalit among the errors
Compare %>% ggplot(aes(x = Diff)) + geom_histogram()

#check for obvious trends among games with the highest residuals
HighErr <- Compare[Compare$DiffA>2000000,]
HighErr <- HighErr[order(-HighErr$DiffA),]

#calculate RMSE without hit or bust games
NoHitBust <- Compare[Compare$DiffA<2000000,]
RMSE(NoHitBust$Sales, NoHitBust$Predicted)

#exploreResults
train_gbmSF$bestTune
varImp(train_gbmSF)
