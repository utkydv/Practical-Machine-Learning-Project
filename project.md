### Executive Summary

In this project, the data from accelerometers on the belt, forearm, arm,
and dumbell of 6 participants has been used. They were asked to perform
barbell lifts correctly and incorrectly in 5 different ways.The goal of
this project is to:  
a) predict the manner in which they did the exercise by building a
model  
b) explain how cross validation has been used  
c) find out expected out of sample rate  
d) use the prediction model to predict 20 different test cases

### Data Import

Here we assume that the required testing & training data has been
downloaded & placed in the present working directory.

    train<-read.csv("pml-training.csv",na.strings=c("NA","")) #converting all missing values to NA
    dim(train)

    ## [1] 19622   160

    test<-read.csv("pml-testing.csv")
    dim(test)

    ## [1]  20 160

### Data Pre-Processing

Since some of the modelling techniques that we are going to deploy (like
random forest) would require removal of missing values, we will first
remove columns that contain NAs.

    train <- train[,sapply(train, function(x) sum(is.na(x))==0)]
    dim(train)

    ## [1] 19622    60

The column dimension reduces to 60 from 160. We further reduce the
column dimension to 53 by removing first 7 columns like individual
identifiers or timestamps.

    train<-train[,-c(1:7)]
    dim(train)

    ## [1] 19622    53

We apply same treatment to test set.

    test <- test[,sapply(test, function(x) sum(is.na(x))==0)]
    test<-test[,-c(1:7)]
    dim(test)

    ## [1] 20 53

We would like to be sure that the columns in the train & test sets are
absolutely same. We check the names of the columns in the reduced sets.

    which(names(train)!=names(test))

    ## [1] 53

The only difference is the column number 53 i.e. column classe in train
set & problem\_id in test set, which is expected.

### Data Preparation

Now we use this train set to create new subsets for model training &
testing purpose.

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    inTrain<- createDataPartition(y=train$classe,p=0.7,list=FALSE)
    newtrain<- train[inTrain,]
    newtest <- train[-inTrain,]

### Prediction Model

This is a classification problem & we are going to try the following
models:  
\#\#\#\#1. CART  
For the purpose of cross validation, we would be using 4-fold validation
in order to get better estimate of in sample error rate:

    library(rpart)

    mod <- train(classe~.,method="rpart", data=newtrain, trControl=trainControl(method="cv",number=4))

The decision tree generated from the model looks like below:

    library(rattle)

    ## Rattle: A free graphical interface for data mining with R.
    ## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

![](project_files/figure-markdown_strict/unnamed-chunk-11-1.png)

Let us see the out of smaple error of this model

    pred <- predict(mod,newtest)
    confusionMatrix(pred,newtest$classe) 

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1416  416  278  185  110
    ##          B    7  202   24    3    5
    ##          C   89  280  399   77  191
    ##          D  152  240  325  611  159
    ##          E   10    1    0   88  617
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5514          
    ##                  95% CI : (0.5386, 0.5642)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.4259          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8459  0.17735   0.3889   0.6338   0.5702
    ## Specificity            0.7651  0.99178   0.8689   0.8220   0.9794
    ## Pos Pred Value         0.5888  0.83817   0.3851   0.4109   0.8617
    ## Neg Pred Value         0.9259  0.83398   0.8707   0.9197   0.9100
    ## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
    ## Detection Rate         0.2406  0.03432   0.0678   0.1038   0.1048
    ## Detection Prevalence   0.4087  0.04095   0.1760   0.2527   0.1217
    ## Balanced Accuracy      0.8055  0.58457   0.6289   0.7279   0.7748

#### 2. Random Forest(RF)

We will now run RF as our model. We won't specify cross validation
explicity here as RF inherently does cross validation while building the
trees.

    library(randomForest)

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    mod1<-train(classe~.,method="rf", data=newtrain) 

Let us see the out of smaple error for RF

    pred1 <- predict(mod1,newtest)
    confusionMatrix(pred1,newtest$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    3    0    0    0
    ##          B    2 1136    3    0    0
    ##          C    0    0 1023    8    0
    ##          D    0    0    0  956    0
    ##          E    0    0    0    0 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9973          
    ##                  95% CI : (0.9956, 0.9984)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9966          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9974   0.9971   0.9917   1.0000
    ## Specificity            0.9993   0.9989   0.9984   1.0000   1.0000
    ## Pos Pred Value         0.9982   0.9956   0.9922   1.0000   1.0000
    ## Neg Pred Value         0.9995   0.9994   0.9994   0.9984   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1930   0.1738   0.1624   0.1839
    ## Detection Prevalence   0.2846   0.1939   0.1752   0.1624   0.1839
    ## Balanced Accuracy      0.9990   0.9982   0.9977   0.9959   1.0000

#### 3. Generalized Boosted Regression Model(GBM)

Let us now try GBM as our prediction model. Please note we are not using
cross validation here as it takes a very long time to run the model with
cross validation. Those having latest configuration machines can icnlude
cross validation step.

    library(gbm)

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.1

    mod2<-train(classe~.,method="gbm", data=newtrain) 

Let us see the out of smaple error for GBM

    pred2 <- predict(mod2,newtest)

    ## Loading required package: plyr

    confusionMatrix(pred2,newtest$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1655   35    0    2    1
    ##          B   10 1077   24    4    5
    ##          C    5   26  995   31    2
    ##          D    3    0    7  927    5
    ##          E    1    1    0    0 1069
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9725         
    ##                  95% CI : (0.968, 0.9765)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9652         
    ##  Mcnemar's Test P-Value : 6.224e-07      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9886   0.9456   0.9698   0.9616   0.9880
    ## Specificity            0.9910   0.9909   0.9868   0.9970   0.9996
    ## Pos Pred Value         0.9776   0.9616   0.9396   0.9841   0.9981
    ## Neg Pred Value         0.9955   0.9870   0.9936   0.9925   0.9973
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2812   0.1830   0.1691   0.1575   0.1816
    ## Detection Prevalence   0.2877   0.1903   0.1799   0.1601   0.1820
    ## Balanced Accuracy      0.9898   0.9683   0.9783   0.9793   0.9938

From the above results, we observe, RF is giving us the best accuracy in
terms of out os sample error. Hence, we are selecting RF as our model
for prediction on the 20 observations shared with us.

### Test Set Prediction

    pred3 <- predict(mod1,test)
    pred3

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
