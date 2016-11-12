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

Let us see the out of sample error of this model

    pred <- predict(mod,newtest)
    confusionMatrix(pred,newtest$classe) 

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1532  483  489  437  144
    ##          B   24  392   31  173  142
    ##          C  115  264  506  354  292
    ##          D    0    0    0    0    0
    ##          E    3    0    0    0  504
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4986          
    ##                  95% CI : (0.4857, 0.5114)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3442          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9152  0.34416  0.49318   0.0000  0.46580
    ## Specificity            0.6312  0.92204  0.78905   1.0000  0.99938
    ## Pos Pred Value         0.4966  0.51444  0.33050      NaN  0.99408
    ## Neg Pred Value         0.9493  0.85419  0.88057   0.8362  0.89253
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2603  0.06661  0.08598   0.0000  0.08564
    ## Detection Prevalence   0.5242  0.12948  0.26015   0.0000  0.08615
    ## Balanced Accuracy      0.7732  0.63310  0.64111   0.5000  0.73259

Out of sample error: 0.5014444

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
    ##          A 1672    0    0    0    0
    ##          B    2 1139    1    0    0
    ##          C    0    0 1024    8    0
    ##          D    0    0    1  956    0
    ##          E    0    0    0    0 1082
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.998           
    ##                  95% CI : (0.9964, 0.9989)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9974          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   1.0000   0.9981   0.9917   1.0000
    ## Specificity            1.0000   0.9994   0.9984   0.9998   1.0000
    ## Pos Pred Value         1.0000   0.9974   0.9922   0.9990   1.0000
    ## Neg Pred Value         0.9995   1.0000   0.9996   0.9984   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1935   0.1740   0.1624   0.1839
    ## Detection Prevalence   0.2841   0.1941   0.1754   0.1626   0.1839
    ## Balanced Accuracy      0.9994   0.9997   0.9982   0.9957   1.0000

Out of sample error: 0.0020391

#### 3. Generalized Boosted Regression Model(GBM)

Let us now try GBM as our prediction model. Please note we are not using
cross validation here as it takes a very long time to run the model with
cross validation. Those having latest configuration machines can include
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

Let us see the out of sample error for GBM

    pred2 <- predict(mod2,newtest)

    ## Loading required package: plyr

    confusionMatrix(pred2,newtest$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1652   25    0    1    0
    ##          B   13 1091   23    4    7
    ##          C    7   22  996   31    8
    ##          D    1    0    6  925    5
    ##          E    1    1    1    3 1062
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.973          
    ##                  95% CI : (0.9685, 0.977)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9658         
    ##  Mcnemar's Test P-Value : 4.679e-06      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9869   0.9579   0.9708   0.9595   0.9815
    ## Specificity            0.9938   0.9901   0.9860   0.9976   0.9988
    ## Pos Pred Value         0.9845   0.9587   0.9361   0.9872   0.9944
    ## Neg Pred Value         0.9948   0.9899   0.9938   0.9921   0.9958
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2807   0.1854   0.1692   0.1572   0.1805
    ## Detection Prevalence   0.2851   0.1934   0.1808   0.1592   0.1815
    ## Balanced Accuracy      0.9903   0.9740   0.9784   0.9786   0.9901

Out of sample error: 0.0270178

From the above results, we observe, RF is giving us the best accuracy as
it has minimum out of sample error. Hence, we are selecting RF as our
model for prediction on the 20 observations shared with us.

### Test Set Prediction

    pred3 <- predict(mod1,test)
    pred3

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
