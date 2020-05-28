library(readxl)
library(lubridate)
library(MLmetrics)
library(data.table)
library(dplyr)
library(chron)
library(timeDate)
library(leaps)
library(mctest)
library(mlr)
library(xgboost)
library(MASS)
library(rvest)
library(xml2)
library(forecast)
library(ggplot2)

data = read_excel("data.xlsx")

# boxlam <- BoxCox.lambda(data$Load)
# data$boxload <- BoxCox(data$Load, lambda = boxlam)

# create variables like Trend, Month, Weekday, and polynomial and interaction terms
data$weekday = as.POSIXlt(data$Date)$wday+1
data$month = month(ymd(data$Date))
data$trend = c(1: length(data$Date))
data$WH = data$weekday*data$Hour
data$TP2 = data$Temperature**2
data$TP3 = data$Temperature**3
data$TM = data$Temperature*data$month
data$TP2M = data$TP2*data$month
data$TP3M = data$TP3*data$month
data$TH = data$Temperature*data$Hour
data$TP2H = data$TP2*data$Hour
data$TP3H = data$TP3*data$Hour

# create lag temperature variables
data = setDT(data)[, paste0('T', 1:72) := shift(Temperature, 1:72)][]

# create lag polynomial temperature variables
for (i in c(1:72)){
  data = setDT(data)[, paste0('T',i,'P2') := '^'(data[[paste0('T',i)]], 2)][]
  data = setDT(data)[, paste0('T',i,'P3') := '^'(data[[paste0('T',i)]], 3)][]
}

# create lag month and lag hour variables
data = setDT(data)[, paste0('M', 1:72) := shift(month, 1:72)][]
data = setDT(data)[, paste0('H', 1:72) := shift(Hour, 1:72)][]

# create lag interaction terms
for (i in c(1:72)){
  data = setDT(data)[, paste0('T',i,'M',i) := data[[paste0('T',i)]]*data[[paste0('M',i)]]][]
  data = setDT(data)[, paste0('T',i,'P2M',i) := data[[paste0('T',i,'P2')]]*data[[paste0('M',i)]]][]
  data = setDT(data)[, paste0('T',i,'P3M',i) := data[[paste0('T',i,'P3')]]*data[[paste0('M',i)]]][]
  data = setDT(data)[, paste0('T',i,'H',i) := data[[paste0('T',i)]]*data[[paste0('H',i)]]][]
  data = setDT(data)[, paste0('T',i,'P2H',i) := data[[paste0('T',i,'P2')]]*data[[paste0('H',i)]]][]
  data = setDT(data)[, paste0('T',i,'P3H',i) := data[[paste0('T',i,'P3')]]*data[[paste0('H',i)]]][]
}

# create Moving Average, Moving Minimum, and Moving Maximum temperature variables
for (i in c(1:49)){
  data = setDT(data)[, paste0('TDMA',i) := rowMeans(data[,(10+i):(33+i)])][]
  data = setDT(data)[, paste0('TDMin',i) := apply(data[,(10+i):(33+i)],1,FUN=min)][]
  data = setDT(data)[, paste0('TDMax',i) := apply(data[,(10+i):(33+i)],1,FUN=max)][]
}

# train, val,pred split
train = data[1:26304,]
val = data[26305:35064,]
pred = data[35065:43848,]

# baseline vanilla model
vanilla = lm(Load ~ trend + month + weekday + Hour + weekday:Hour +
               poly(Temperature,3) + Temperature:month + T2:month + T3:month +
               Temperature:Hour + T2:Hour + T3:Hour
             , data=train)
summary(vanilla)
MAPE(vanilla$fitted.values,train$Load[4:26304])

# xgboost
dtrain <- xgb.DMatrix(data = as.matrix(train[,c(2:3,5:232,377:955)]), label=train$Load) 
dtest <- xgb.DMatrix(data = as.matrix(val[,c(2:3,5:232,377:955)]))
dpred <- xgb.DMatrix(data = as.matrix(pred[,c(2:3,5:232,377:955)]))

params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 200, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

bst <- xgboost(params = params, data = dtrain, nthread = 2, nrounds = 200,print_every_n = 10, early_stopping_rounds = 10)

# importanceRaw <- xgb.importance(model = bst)
# importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]
# importanceClean

xgbpredtrain <- predict(bst, dtrain)
MAPE(xgbpredtrain,train$Load) #0.006608219

xgbpredval <- predict(bst, dtest)
MAPE(xgbpredval,val$Load) #0.04258473

# retrain using both train and val
dtrain.full <- xgb.DMatrix(data = as.matrix(data[1:35064,c(2:3,5:232,377:955)]), label=data[1:35064,]$Load) 

# xgbcv <- xgb.cv( params = params, data = dtrain.full, nrounds = 200, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

bst <- xgboost(params = params, data = dtrain.full, nthread = 2, nrounds = 200,print_every_n = 10, early_stopping_rounds = 10)

xgbpred <- predict(bst, dpred)
xgbtrain <- predict(bst, dtrain.full)
MAPE(xgbtrain,data[1:35064,]$Load) 

write.csv(xgbpred,"xgbpred.csv")
write.csv(xgbtrain,"xgbfitted.csv")
# vanilla + Recency effect Regression
enhanced = lm(Load~.-Date-M1-M2-M3-M4-M5-M6-
               -M7-M8-M9-M10-M11-M12-M13-
               -M14-M15-M16-M17-M18-M19-M20-
               -M21-M22-M23-M24-M25-M26-M27-
               -M28-M29-M30-M31-M32-M33-M34-
               -M35-M36-M37-M38-M39-M40-M41-
               -M42-M43-M44-M45-M46-M47-M48-
               -M49-M50-M51-M52-M53-M54-M55-
               -M56-M57-M58-M59-M60-M61-M62-
               -M63-M64-M65-M66-M67-M68-M69-
               -M70-M71-M72-H1-H2-H3-H4-
               -H5-H6-H7-H8-H9-H10-H11-
               -H12-H13-H14-H15-H16-H17-H18-
               -H19-H20-H21-H22-H23-H24-H25-
               -H26-H27-H28-H29-H30-H31-H32-
               -H33-H34-H35-H36-H37-H38-H39-
               -H40-H41-H42-H43-H44-H45-H46-
               -H47-H48-H49-H50-H51-H52-H53-
               -H54-H55-H56-H57-H58-H59-H60-
               -H61-H62-H63-H64-H65-H66-H67-
               -H68-H69-H70-H71-H72,
             data=train)
summary(enhanced)
MAPE(enhanced$fitted.values,train$Load[73:26304])
MAPE(enhanced$fitted.values,train$Load)





