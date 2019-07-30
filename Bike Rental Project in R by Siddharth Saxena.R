#Removing all objects stored in R
rm(list=ls())

#Setting up current work directory
setwd("S:/Data science/Bike Project")

#Checking the work directory
getwd()

#Loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees', "usdm")

#installing all packages in x
lapply(x, require, character.only = TRUE)
rm(x)

#Reading the csv file
bikedata = read.csv("day.csv", header = T)

#Understanding the structure of dataset
str(bikedata)

summary(bikedata)

######################################## Outlier Analysis ###############################################

numeric_index = sapply(bikedata,is.numeric) #selecting only numeric data

numeric_data = bikedata[,numeric_index]

cnames = colnames(numeric_data)

 for (i in 1:length(cnames))
 {
   assign(paste0("bcount",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(bikedata))+ 
            stat_boxplot(geom = "errorbar", width = 0.5) +
            geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                         outlier.size=1, notch=FALSE) +
            theme(legend.position="bottom")+
            labs(y=cnames[i],x="No of Bikes")+
            ggtitle(paste("Box plot of Bike count for",cnames[i])))
 }


# Plotting plots together
 gridExtra::grid.arrange(bcount1,bcount2,bcount3,ncol=3)
 gridExtra::grid.arrange(bcount4,bcount5,bcount6,ncol=3)
 gridExtra::grid.arrange(bcount7,bcount8,bcount9,ncol=3)
 gridExtra::grid.arrange(bcount10,bcount11,bcount12,ncol=3)
 gridExtra::grid.arrange(bcount13,bcount14,bcount15,ncol=3)

#After checking the plots we get to know that humidity, windspeed and casual have outliers
#Since casual is just a part of final count we will not take it in account
 
#Storing name of columns from where we need to remove outliers (humidity and windspeed)
 cnm = colnames(bikedata[,12:13])
 
#Loop to remove outliers from above columns
 for(i in cnm){
   print(i)
   val = bikedata[,i][bikedata[,i] %in% boxplot.stats(bikedata[,i])$out]
   print(length(val))
   bikedata = bikedata[which(!bikedata[,i] %in% val),]
 }
 
#So, there are two outlier values in humidity and 12 outlier values in windspeed which are removed
 
##################################### Missing Value Analysis ##########################################
#Count of missing values in the bikedata columns
apply(bikedata,2,function(x){sum(is.na(x))}) 
 
#Since there is no missing value in data, we will skip the missing value part
 
#################################### Feature Selection ################################################
corrgram(bikedata[,numeric_index], order = F,
          upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
 
bikedata = subset(bikedata, select = -c(instant,dteday,atemp, casual, registered))
 
#Removed instant because it doesnot depict any feature
#Removed dteday as it is of no use
#Removed atemp as it is highly correlated with temp
#Removed casual and registered s they are just the part of toal count of target variable

################################## Feature Scaling #####################################################

#Normalisation
cnm = c("season","yr","mnth","holiday","weekday","workingday","weathersit",
           "temp","hum", "windspeed", "cnt")

for(i in cnm){
  print(i)
  bikedata[,i] = (bikedata[,i] - min(bikedata[,i]))/
    (max(bikedata[,i] - min(bikedata[,i])))
}

################################## Model Development ####################################################

#Checking the multicollinearity
vif(bikedata[,-11])
vifcor(bikedata[,-11], th = 0.9)

#Using cross validation
train_in = sample(1:nrow(bikedata), 0.7 * nrow(bikedata))
train_data = bikedata[train_in,]
test_data = bikedata[-train_in,]

############################### Linear Regression ########################################################

model_lm = lm(cnt ~., data = train_data)

#Summary of the model
summary(model_lm)

#Prediction
LM_predict = predict(model_lm, test_data[,1:10])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

#Calculating MAPE
MAPE(test_data[,11], LM_predict)

#MAPE = 0.19269
#Accuracy = 80.73 percent

#Plotting the graph 
qplot(x = test_data[,11], y = LM_predict, data = test_data, color = I("red"), geom = "point", xlab = "Test Data", ylab = "Predictions")

##################################### Decision Tree #######################################################

treefit = rpart(cnt ~ . ,data = train_data, method = "anova")

summary(treefit)

predictions_tt = predict(treefit, test_data[,1:10])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_data[,11],predictions_tt)

#MAPE = 0.2433816
#Accuracy = 75.66 percent

qplot(x = test_data[,11], y = predictions_tt, data = test_data, color = I("red"), geom = "point", xlab = "Test Data", ylab = "Predictions")

##################################### Random Forest #######################################################

forestmodel = randomForest(cnt ~.,data=train_data)

summary(forestmodel)

forest_predictions = predict(forestmodel,test_data[,1:10])

MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test_data[,11],forest_predictions)

#MAPE = 0.1782885
#Accuracy = 82.17 percent

qplot(x = test_data[,11], y = forest_predictions, data = test_data, color = I("red"), geom = "point", xlab = "Test Data", ylab = "Predictions")

write.csv(forest_predictions, "Bike Data predictions.csv", row.names = F)
