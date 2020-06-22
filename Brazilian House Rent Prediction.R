################################

#Harvardx - Data Science Capstone - "Prediciting House Rental Price"

#In this project, we will be creating a model to predict house rental prices 

#Student: Eline Morais

################################


# Install packages required
# Note: this process could take a couple of minutes
if(!require(readr)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(brnn)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Cubist)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(pls)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("data.table", repos = "http://cran.us.r-project.org")

#libraries required
library(tidyverse)
library(caret) 
library(readr)
library(httr)
library(dplyr)
library(ggplot2) 
library(ggthemes) 
library(gridExtra) 
library(stringr)
library(knitr)
library(lubridate)
library(rvest)
library(matrixStats)
library(purrr)


#The aim of this data analysis project is to create an automated model to predict property rental values
#without human bias using regression machine learning algorithms.

#The data source used for this project is obtained from
#https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent?select=houses_to_rent_v2.csv, or
#https://raw.githubusercontent.com/emoraiss/brazilian_rent/master/datasets_554905_1035602_houses_to_rent_v2.csv".

#This project is part of final evaluation students on Data Science Professional Certificate of Harvardx 
#- Casptone, where each student must choose a dataset on internet and use tools they learned during the 
#course, applying machine learning.

#Accessing the data - Github
tmp<- tempfile()
download.file("https://raw.githubusercontent.com/emoraiss/brazilian_rent/master/datasets_554905_1035602_houses_to_rent_v2.csv",tmp)
data<-read_csv(tmp)
file.remove(tmp)

#Data structure
dim(data)
names(data)
summary(data)

#Changing titles
colnames(data) <- gsub(pattern = "[[:punct:]]|R", replacement = "", colnames(data))
colnames(data) <- colnames(data) %>% str_trim() %>% str_replace_all(.,"\\s", "_")
colnames(data)

#looking floor
data %>% group_by(floor) %>% summarise(n=n()) %>% arrange(desc(n)) %>% head() %>% knitr::kable()

#Change floor = "-" to 0
data <- data %>% mutate(floor=as.numeric(ifelse(floor=="-", 0, floor)))

#Any NA?
sum(is.na(data))

#n_distinct - summary features 
data %>% summarise(Cities= n_distinct(city), Mean_area= mean(area), Room_options= n_distinct(rooms), Parking_options= n_distinct(parking_spaces),
                   Floor_options= n_distinct(floor), Mean_rental= mean(rent_amount)) %>% knitr::kable()

#prevalence by city
data %>% group_by(city) %>% summarise(n= n(), "%" = (n*100/10692)) %>% arrange(desc(n)) %>% knitr::kable()

#Create train and test set
# Validation set will be 10% of data
# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = data$rent_amount, times = 1, p = 0.1, list = FALSE)
build<- data[-test_index,]
validation <- data[test_index,]

#dimension build
dim(build)

#Data Analysis

#Area and Rent Amount

#Comparing area by city, we can see that São Paulo and Belo Horizonte have larger ranges, 
#but in general 50% of properties have areas between 50m2 and 200m2.
area <- build %>% group_by(city) %>%  qplot(city, area, data = ., geom = "boxplot") +  
  scale_y_log10()+theme_bw() +ggtitle("Area by City")+ 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
rent<- build %>% group_by(city) %>%  qplot(city, rent_amount, data = ., geom = "boxplot") +
  scale_y_log10() +theme_bw() +ggtitle("Rent by City") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
grid.arrange(area, rent, ncol = 2)

#Highest area
build %>% select(city, area, rent_amount)  %>% arrange(desc(area)) %>% head() %>% knitr::kable()

#Since by definition a mansion is around 2.000 m2, we will consider properties area>=2000 errors 
#and will remove them. Our dataset has 4 of them.
remove_index<- which(build$area>=2000)
build<- build[-remove_index,]
dim(build)

#Highest rent amount
build %>% select(city, area, rooms , bathroom, parking_spaces, property_tax,rent_amount)  %>%
  filter(rent_amount>10000) %>% arrange(desc(rent_amount)) %>% head(., 10) %>% knitr::kable()

#We see a small house in disagreement with the rest of the "select" group. Let us take rent
#values higher than 20k as outliers.
remove_index<- which(build$rent_amount>20000)
build<- build[-remove_index,]
dim(build)

#Now let us see their distributions: 
rent<- build %>% ggplot(aes(rent_amount)) +
  geom_histogram(binwidth= 0.1, color= "black", fill= "lightblue") + theme_bw()+
  scale_x_log10() + ggtitle("Rent distribution")
area<- build %>% ggplot(aes(area)) + geom_histogram(binwidth = 0.1, color= "black", fill= "lightblue") +
  scale_x_log10() +theme_bw()+ ggtitle("Area distribution")
grid.arrange(area,rent, ncol = 2)

#It would be expected that the larger the property higher the rent would be, once a common unit 
#of measure is value per square meter. Follow the scatter plot.
build %>% ggplot(aes(rent_amount, area)) +scale_x_log10()+scale_y_log10()+ geom_point() +
  theme_bw()+ geom_smooth(method= "lm") + ggtitle("Rent vs Area")

#We can see the same high correlation by city
build %>% ggplot(aes(rent_amount, area)) +scale_x_log10()+scale_y_log10()+ geom_point()+theme_bw()+ geom_smooth(method= "lm") + facet_wrap(~city) + ggtitle("Rent vs Area by City")

#correlation
build %>% group_by(city) %>% summarise(cor= cor(area, rent_amount)) %>% arrange(desc(cor)) %>%
  knitr:: kable()

#Rooms and Bathrooms

#We have seen at the first summary that the minimum and maximum number of rooms (between 1 and 13) 
#and bathrooms (between 1 and 10) are quite similar. Let us check how they are distributed and correlated. 
room<- build %>% group_by(rooms) %>% ggplot(aes(rooms)) + 
  geom_histogram(binwidth= 0.5,color= "black", fill= "lightblue") + scale_y_sqrt() + 
  theme_bw() + ggtitle("Rooms distribution")
bath<- build %>% group_by(bathroom) %>% ggplot(aes(bathroom)) + 
  geom_histogram(binwidth= 0.5, color= "black", fill= "lightblue") +scale_y_sqrt() + 
  theme_bw() + ggtitle("Bathrooms distribution")
grid.arrange(room, bath, ncol=2)

#Only about 2% of houses has more than 5 rooms or bathrooms. In a boxplot we can see the range of
#bathrooms for each quantity of room.
build  %>% ggplot(aes(rooms, bathroom, group= rooms)) +
  geom_boxplot() +theme_bw()+ggtitle("Rooms vs Bathroom") 

#We can think that a low standard apartment/house may have fewer bathrooms than rooms, in spite of high
#standard properties, in addition to having one bathroom per room, there may be a toilet, a maid's 
#bathroom, a leisure area bathroom, among others.

#Looking closer at some points in the graphic we can find other outliers. 
#Small houses (area) with a big quantity of bathroom or rooms, making no sense. Let us remove them.
build %>% filter(rooms==5 & bathroom==1|rooms==2 & bathroom>6) %>% select(city, area, rooms, bathroom, parking_spaces, floor, rent_amount) %>% knitr::kable()

remove_index<- which(build$rooms==5 & build$bathroom==1|build$rooms==2 & build$bathroom>6) 
build<- build[-remove_index,]
dim(build)

#Relation of room and bathroom with rental price.
room<- build %>% group_by(rooms) %>%  ggplot(aes(rooms, rent_amount, group= rooms)) + 
  geom_boxplot() +  scale_y_log10()+theme_bw() +ggtitle("Rooms vs Rental price")
bathroom<- build %>% group_by(bathroom) %>% ggplot(aes(bathroom, rent_amount, group= bathroom)) +
  geom_boxplot()+ scale_y_log10()+theme_bw() +ggtitle("Bathrooms vs Rental price")
grid.arrange(room, bathroom, ncol = 2)

#Parking Spaces

#We believe they are quite related to area and rental price as well, since highest pattern houses 
#may have more parking spaces. 
build %>% group_by(parking_spaces) %>% ggplot(aes(parking_spaces)) +
  geom_histogram(binwidth = 0.5, color="black", fill="lightblue") +scale_y_sqrt()+ 
  theme_bw() + ggtitle("Parking Spaces distribution")

#By the plot we can see most properties has less than 4 parking spaces. Now let us have a look at 
#boxplot to see the range of area and rooms to each quantity of parking space. 
area<- build %>% group_by(parking_spaces) %>% ggplot(aes(parking_spaces, area, group= parking_spaces)) +
  geom_boxplot()+ scale_y_log10()+theme_bw() +ggtitle("Parking Spaces vs area")
rooms<-build %>% group_by(parking_spaces) %>% ggplot(aes(parking_spaces, rooms, group= parking_spaces)) +
  geom_boxplot()+ scale_y_log10()+theme_bw() +ggtitle("Parking Spaces vs rooms")
grid.arrange(area, rooms, nrow=2)

#There are 8 houses with only one room and 4 or more parking spaces. Also there are 3 other houses with 
#less than 100m2, 2 rooms and more than 4 parking. Let us remove these outliers as well.
build %>% filter(parking_spaces>=4 & area<100 & rooms<3|rooms==1 & parking_spaces>3) %>% 
  select(city, area, rooms, bathroom,hoa,  parking_spaces, rent_amount) %>%
  arrange(desc(parking_spaces)) %>% knitr:: kable()

#removing outliers
remove_index<- which(build$parking_spaces>=4 & build$area<100 & build$rooms<3|build$rooms==1 & build$parking_spaces>3)
build<- build[-remove_index,]
dim(build)

#parking spaces vs rental price.
build %>% group_by(parking_spaces) %>% ggplot(aes(parking_spaces, rent_amount, group=parking_spaces)) +
  geom_boxplot() + scale_y_log10()+ theme_bw() + ggtitle("Parking spaces vs Rental price")

#Furniture

#We can think that prices of furnished houses are more expensive due to the investment made. 
#But let us see what data shows. Furnished properties represent about 25% of our dataset.  
hist<- build %>% group_by(furniture) %>% ggplot(aes(furniture)) + theme_bw()+ 
  geom_bar(color= "black", fill= "lightblue") 
box<- build %>% group_by(furniture) %>% ggplot(aes(furniture, rent_amount)) + 
  geom_boxplot() + scale_y_log10() + theme_bw() + ggtitle("Furniture vs Rental price")
grid.arrange(hist, box, ncol= 2)

#Animal

#Although it is a very relevant factor in the decision to choose a home for those who have a pet,
#we do not expect the price to be impacted. Let us see the plots. 
pet1<- build %>% group_by(animal) %>% ggplot(aes(animal)) + theme_bw()+ 
  geom_bar(color= "black", fill= "lightblue") + ggtitle("Animal Policy")
pet<- build %>% group_by(city) %>% ggplot(aes(animal, rent_amount)) + 
  geom_boxplot() + scale_y_log10() + theme_bw() +ggtitle("Animal Policy vs Rental price")
grid.arrange(pet1, pet, ncol=2)

#Floor

#Highest floors
build %>% arrange(desc(floor)) %>% select(city, area,floor) %>% head() %>% knitr::kable()

#It is unlikely that this information is true once the highest building in brazil has 46 floors.
#So for now we will change floors higher than 46 (301 and 51) to 31.
build<- build %>% mutate(floor= ifelse(floor==301|floor==51,31,floor))
build %>% arrange(desc(floor))%>% select(city, area,floor) %>% head()

#floor distribution and relation to price
floor<- build %>% filter(floor>0)  %>% arrange(floor) %>% ggplot(aes(floor))+
  geom_histogram(binwidth= 1, color= "black", fill= "lightblue") + theme_bw()+ 
  ggtitle("Floor distribution")
floor2<- build %>% filter(floor>0 &floor<30)  %>% arrange(floor) %>%
  ggplot(aes(floor, rent_amount, group= floor))+ geom_boxplot() + scale_y_log10() + 
  theme_bw()+ ggtitle("Floor vs Rental")
grid.arrange(floor, floor2, nrow=2)

#Property tax

#highest values:
build %>% select(city, area, property_tax,hoa,rent_amount) %>% arrange(desc(property_tax)) %>%
  head(., 8) %>% knitr::kable()

#It can be seen that or the area is very small or the tax is much higher than other properties with
#similar area and rent value at the same city. So, we will take these first 5 properties off. 
remove_index<- which(build$property_tax>9500)
build<- build[-remove_index,]
dim(build)

#There are 1.448 properties where the tax is equal to zero.
build %>% filter(property_tax==0) %>% nrow()

#plot area vs rental to property tax==0
build %>% select(city,area, property_tax,hoa,rent_amount) %>% filter(property_tax==0) %>%
  arrange(desc(area)) %>% ggplot(aes(area, rent_amount)) + geom_point() + theme_bw()+
  ggtitle("Area vs Rental (Property tax = 0)")

# area vs city property tax== 0
build %>% filter(property_tax==0) %>% select(city, area, property_tax) %>% ggplot(aes(city, area)) + 
  geom_boxplot() + theme_bw() + ggtitle("Area by city (Property tax = 0)")

#Now let us see what happens to properties where the property tax is due. 
build %>% group_by(city) %>% filter(property_tax != 0) %>% ggplot(aes(area, property_tax)) + 
  geom_point() + scale_x_log10()+scale_y_log10() +theme_bw()+ facet_wrap(~city)+theme_bw() + 
  ggtitle("Area vs property tax")

#Rental price vs property tax (>0)
build %>% group_by(city) %>% filter(property_tax != 0) %>% ggplot(aes(rent_amount,property_tax)) + 
  geom_point() + scale_x_log10()+scale_y_log10() +theme_bw()+ facet_wrap(~city)+theme_bw() +
  ggtitle("Rental price vs property tax")

#hoa
#Let's see the lowest and highest values. 

#There are 2.138 properties with hoa= 0, what can be used to define if a property is an independent house
#or whether it is an apartment or condominium house, once we do not have the field about the kind of home
build %>% filter(`hoa`==0) %>% nrow()

#Highest values 
build %>% filter(hoa>0) %>% select(city, area, property_tax,  hoa,rent_amount) %>% arrange(desc(hoa)) %>%
  head(., 8) %>% knitr::kable()

#The first 3 do not make any sense, they are too high to be paid monthly if compared with area, 
#property tax or rental price. Let's remove them.
remove_index<- which(build$hoa>30000)
build<- build[-remove_index,]
dim(build)

#create a new feature "condominium" to classify properties in two classes: those pay the fee and those 
#who do not.
build %>% mutate(condominium= ifelse(hoa==0, "no","yes")) %>% ggplot(aes(condominium, rent_amount)) +
  geom_boxplot() +theme_bw()+ scale_y_log10()+ facet_wrap(~city)+ ggtitle("Condominium vs Rental price)") 

#Hoa by city
build %>% filter(hoa<32000 & hoa>1) %>% ggplot(aes(city, hoa)) + geom_boxplot() +scale_y_log10()+
  theme_bw()+ggtitle("Hoa by city") 

#hoa summary
build %>% filter(hoa<32000 & hoa>1) %>% group_by(city) %>% 
  summarise(avg= mean(hoa), sd= sd(hoa),median= median(hoa)) %>% knitr::kable()

#relation between hoa and rental price.
build %>% filter(hoa<32000 & hoa>1) %>% ggplot(aes(hoa, rent_amount)) + geom_point() + 
  theme_bw()+ scale_y_log10() + scale_x_log10()+ggtitle("Hoa fee vs Rental price") 

#Fire Insurance

#histogram
build %>% ggplot(aes(fire_insurance)) + geom_histogram(color="black", fill="lightblue") +
  scale_y_sqrt()+ theme_bw() + ggtitle("Fire Insurance Distribution")

#Fire tax <=200, 98.7% of data
build %>% filter(fire_insurance<=200) %>% nrow()/nrow(build)

#High correlation between fire insurance and rental price. It must de a dependent variable.
build %>% ggplot(aes(fire_insurance, rent_amount)) + geom_point()+ scale_y_log10() +
  scale_x_log10()+ theme_bw() + ggtitle("Fire insurance vs Rental price")

#Total

#This feature is also a dependable variable, equal to the sum of rental price and all previous taxes and 
#fees. Let's remove it, leaving the following features remaining:
  
#removing total and fire insurance
build<- select(build,- c(total, fire_insurance))
names(build)

#Modelling

#preprocess

nzv<- nearZeroVar(build)
nzv


#Correlation between numeric features
library(ggcorrplot)
vars_cont <- build %>% select_if(is.numeric)
corr <-  cor(vars_cont, use = 'complete.obs')
options(repr.plot.width=20, repr.plot.height=20)
ggcorrplot(corr, lab = TRUE, colors = c("aquamarine", "white", "dodgerblue"), 
           show.legend = F, outline.color = "gray", type = "upper", 
           tl.cex = 12, lab_size = 3, sig.level = .1) + ggtitle("Correlation between variables")+
  labs(fill = "Correlation")

#summary correlation
vars_cont <- build %>% select_if(is.numeric)
buildCor <- cor(vars_cont)
summary(buildCor[upper.tri(buildCor)])

#summary correlation after removing correlation >0.75
highlyCor <- findCorrelation(buildCor, cutoff = .75)
vars_cont <- vars_cont[,-highlyCor]
names(vars_cont)
buildCor2 <- cor(vars_cont)
summary(buildCor2[upper.tri(buildCor2)])

#removing area feature 
build<- select(build,-area)

#Look for linear dependency
comboInfo <- findLinearCombos(vars_cont)
comboInfo

#dimension build dataset
dim(build)

#Create train and test set
set.seed(27, sample.kind = "Rounding")
test_index <- createDataPartition(y =build$rent_amount, times = 1, p = 0.1, list = FALSE)
train<- build[-test_index,]
test <- build[test_index,]


#Define cross_validation - 10 fold
control <- trainControl(method = "cv",
                        number = 10, p=0.9)

# Machine Learning algorithms

# 1- Linear Regression
ini_time<- Sys.time()
set.seed(35, sample.kind = "Rounding")
train_lm<- train(rent_amount~., method= "lm", data = train, preProcess= "scale",trControl = control)
end_time<- Sys.time()
train_lm
print(end_time-ini_time)

#Predictions and results
y_hat_lm<- predict(train_lm, test)
lm_results<- postResample(y_hat_lm, test$rent_amount)
lm_results

#Function - Coefficients
train_lm$finalModel$coefficients %>% knitr::kable()

#Plot results LM
test %>% mutate(y_hat= y_hat_lm) %>%  ggplot(aes(y_hat, rent_amount)) + geom_point() + geom_smooth(method= "lm")  +theme_bw()+ggtitle("Linear Regression Results")

#Highest errors
test %>% mutate(y_hat= y_hat_lm, error= abs(rent_amount-y_hat)) %>% 
  select(city,bathroom,rooms, property_tax, rent_amount, y_hat, error) %>% 
  arrange(desc(error)) %>% head() %>% knitr::kable()

# 2- K-Nearest Neighbors 
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_knn<- train(rent_amount~., method= "knn", data= train, preProcess= "scale",tuneGrid= data.frame(k=seq(10,75,5)), trControl= control)
end_time<- Sys.time()
train_knn
print(end_time-ini_time)

#plot best tune , k=25
ggplot(train_knn, highlight= TRUE) +theme_bw()+ ggtitle("Knn - Best Tune")

#Predictions and results
y_hat_knn<- predict(train_knn,test, type = "raw")
knn_results<- postResample(y_hat_knn, test$rent_amount)
knn_results

#There was an improvement about 5% if compared to linear regression. Let us see if we can do better.

# 3- Regression tree
library(rpart)
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_rpart<- train(rent_amount~., method= "rpart", data= train,trControl= control, preProcess= "scale" ,tuneGrid = data.frame(cp= seq(0,0.06, len=25)), minsplit= 5)
end_time<- Sys.time()
train_rpart
print(end_time-ini_time)

#plot best tune, cp= 0.0025
ggplot(train_rpart, margin= 0.1, highlight=TRUE) + theme_bw() + ggtitle("Cp - Best Tune")

#plot final model
library(rpart.plot)
rpart.plot(train_rpart$finalModel,box.palette="RdBu", shadow.col="gray", nn=TRUE, fallen.leaves=FALSE, tweak=1.3)

#Although results on training was worse we save the RMSE on the test set.

#Predictions and results
y_hat_rpart<- predict(train_rpart, test)
rpart_results<- postResample(y_hat_rpart, test$rent_amount)
rpart_results

# 4- Random Forest 
library(randomForest)
#Tuning parameters:
#mtry (#Randomly Selected Predictors)
#pre-processing RMSE=2000.782 , no pre-processing RMSE= 2000.133.
ini_time<- Sys.time()
set.seed(35, sample.kind = "Rounding")
train_rf<- train_rf<- train(rent_amount~., method= "rf", ntree=100, data= train, tuneGrid= data.frame(mtry=c(2,3,4,5,6)), trControl= control, nSample=5000)
end_time<- Sys.time()
train_rf
print(end_time-ini_time)

#We have tested some values for ntree and mtry. The best results were obtained with ntree=100,
#mtry=3 and without pre-processing features. 

#plot best tune
ggplot(train_rf, highlight= TRUE) + theme_bw() + ggtitle("Mtry - Best Tune")

#see importance of features on model
plot(varImp(train_rf))

#Predictions and results
y_hat_rf<- predict(train_rf, test)
rf_results<- postResample(y_hat_rf, test$rent_amount)
rf_results

# We had an improvement of more 4% in relation to the previous best model "Knn", or 8% in relation to
#Linear Regression. Let us plot the results.

# Plot results
test %>% mutate(y_hat= y_hat_rf) %>%  ggplot() + 
  geom_point(aes(y_hat, rent_amount)) + theme_bw()+ggtitle("Predictions Random Forest")

# Highest absolute errors
test %>% mutate(y_hat= y_hat_knn, error= abs(rent_amount-y_hat)) %>% 
  select(city, property_tax,bathroom, hoa, parking_spaces, rent_amount, y_hat, error) %>%
  arrange(desc(error)) %>% head() %>% knitr::kable()

# 5- Random Forest - Rborist 
library(Rborist)
#Tuning parameters:
#predFixed (#Randomly Selected Predictors)
#minNode (Minimal Node Size)
ini_time<- Sys.time()
set.seed(35, sample.kind = "Rounding")
train_rb <-  train(rent_amount~.,
                   method = "Rborist",
                   nTree = 100,
                   preProcess= "scale",
                   trControl = control,
                   tuneGrid = data.frame(minNode= seq(5,250,25), predFixed= 5), 
                   data= train)
end_time<- Sys.time()
train_rb
print(end_time-ini_time)

# Plot best tune
ggplot(train_rb, highlight= TRUE) + theme_bw() + ggtitle("MinNode - Best tune")

#Predictions and results
y_hat_rb<- predict(train_rb, test)
rb_results<- postResample(y_hat_rb, test$rent_amount)
rb_results

#Although the performance on the training data was very similar but inferior to the previous model, 
#on the test dataset, Rborist got better results.

#We can see one change into the features importance.
plot(varImp(train_rb))

# 6- Support Vector Machines with Radial Basis Function Kernel 
#method= "svmRadial"
#Tuning parameters:
#sigma (Sigma)
#C (Cost)
library(kernlab)
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_svm<- train(rent_amount~., method= "svmRadial", data= train, preProcess= "scale", trControl= control)
end_time<- Sys.time()
train_svm
print(end_time-ini_time)

# Plot best tune
ggplot(train_svm, highlight= TRUE) + theme_bw() + ggtitle("Cost - Best Tune")

# Predictions and results
y_hat_svm<- predict(train_svm, test)
svm_results<-postResample(y_hat_svm, test$rent_amount)
svm_results

# 7- Model Tree
#model tree cubist
#Tuning parameters:
#committees (#Committees)
#neighbors (#Instances)
library(Cubist)
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_cub<- train(rent_amount~., method= "cubist", data= train,preProcess= "scale", trControl= control)
end_time<- Sys.time()
train_cub
print(end_time-ini_time)

# Plot best tune
ggplot(train_cub, highlight= TRUE) + theme_bw() + ggtitle("Committees and neighbors - Best Tune")

# We see hoa as the most importante feature in this model
plot(varImp(train_cub))

# Predicitions and results
y_hat_cub<- predict(train_cub, test)
cub_results<-postResample(y_hat_cub, test$rent_amount)
cub_results

# 8- Principal Component Analysis
#Principal Component Analysis
#Tuning parameters:
#ncomp (#Components)
library(pls)
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_pca<- train(rent_amount~., method= "pcr", data= train, preProcess="scale", trControl= control)
end_time<- Sys.time()
train_pca
print(end_time-ini_time)

#plot best tune
ggplot(train_pca, highlight= TRUE) + theme_bw() + ggtitle("N Components - Best Tune")

# predicitons and results
y_hat_pca<- predict(train_pca, test)
pca_results<-postResample(y_hat_pca, test$rent_amount)
pca_results

# 9- Bayesian Regularized Neural Networks
#Bayesian Regularized Neural Networks - "brnn"
#Tuning parameters:
#neurons 
library(brnn)
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_brnn<- train(rent_amount~., method= "brnn", data= train,preProcess="scale", trControl= control)
end_time<- Sys.time()
train_brnn
print(end_time-ini_time)

#plot best tune
ggplot(train_brnn, highlight= TRUE) + theme_bw() + ggtitle("Neurons - Best Tune")

# Predictions and results
y_hat_brnn<- predict(train_brnn, test)
brnn_results<-postResample(y_hat_brnn, test$rent_amount)
brnn_results

# 10- Stochastic Gradient Boosting
#Stochastic Gradient Boosting
#Tuning parameters:
#n.trees (# Boosting Iterations)
#interaction.depth (Max Tree Depth)
#shrinkage (Shrinkage)
#n.minobsinnode (Min. Terminal Node Size)
library(gbm)
library(plyr)
set.seed(35, sample.kind = "Rounding")
ini_time<- Sys.time()
train_gbm<- train(rent_amount~., method= "gbm", data= train, preProcess="scale",trControl= control)
end_time<- Sys.time()
train_gbm
print(end_time-ini_time)


#Plot best tune
ggplot(train_gbm, highlight= TRUE) + theme_bw() + ggtitle("N.trees and Interation depth - Best Tune")

# Predictions and results
y_hat_gbm<- predict(train_gbm, test)
gbm_results<-postResample(y_hat_gbm, test$rent_amount)
gbm_results

#Now let us compare the results of our models on the test set.
model_results <- t(data.frame(Linear_Regression= lm_results, KNearest_Neighbors= knn_results, 
                              Regression_tree= rpart_results, Random_Forest= rf_results,
                              Random_Rborist=rb_results, SVM_Radial=svm_results, 
                              Model_tree=cub_results, Principal_Component_Analysis=pca_results,
                              Bayesian_Reg_Neural_Networks= brnn_results,
                              Stochastic_Gradient_Boosting=gbm_results))
model_results 

# 11- Ensemble Model
#Now we are going to combine the results of the  top 3 best RMSE results on the test set 
#and predict with their average.  
top_results <- sort(model_results[,1]) %>% head(.,3)
top_results 

# Predictions and Results
ensemble<- data.frame(rf= y_hat_rf, rf= y_hat_rb, brnn= y_hat_brnn)
y_hat_ens<- rowMeans(ensemble)
ens_results<- postResample(y_hat_ens, test$rent_amount)
ens_results

#We have a small improvement. So let us decide to use the ensemble model equal to the mean prediction
#of Random Forest, Random Forest Rborist and Bayesian Regularized Neural Networks as our final model.

#Let us plot our final predictions and compare to actual values.
test %>% mutate(y_hat= y_hat_ens) %>%  ggplot() + geom_point(aes(y_hat,rent_amount, color=city)) +
  theme_bw()+ggtitle("Ensemble model Predictions") 

#Although São Paulo is the city with the highest prevalence in the database, it is also the city that 
#contains the largest ranges in features, requiring additional information to improve the accuracy of 
#models.

#Plot histogram absolute errors
test %>% mutate(y_hat= y_hat_ens, error= abs(rent_amount-y_hat)) %>%
  select(city, bathroom, hoa, property_tax, rent_amount, y_hat, error) %>%
  ggplot(aes(error)) + geom_histogram(binwidth = 40, fill="lightblue", color="black") + 
  theme_bw() +ggtitle("Absolute Error Distribution")

#Highest absolute errors
test %>% mutate(y_hat= y_hat_ens, error= abs(rent_amount-y_hat)) %>% 
  select(city, bathroom, hoa, property_tax, rent_amount, y_hat, error) %>%
  arrange(desc(error)) %>% head() %>% knitr::kable()

#RMSE is sensitive to outliers. We have some very precise predictions. About 35% of predictions had an 
#absolute error smaller than R$300. We can see most absolute errors were small but the effect of each
#error on RMSE is proportional to the size of the squared error, thus larger errors have a 
#disproportionately large effect on RMSE.

#Lowest  absolute errors
#We have some very precise predictions.
test %>% mutate(y_hat= y_hat_ens, error= abs(rent_amount-y_hat)) %>%
  select(city, bathroom, hoa, property_tax, rent_amount, y_hat, error) %>%
  arrange(error) %>% head() %>% knitr::kable()

#Absolute error<300
test %>% mutate(y_hat= y_hat_ens, error= abs(rent_amount-y_hat)) %>% 
  select(city, bathroom, hoa, property_tax, rent_amount, y_hat, error) %>% 
  filter(error<=300) %>% nrow()

# RESULTS

#As result we are going to apply our best model - ensemble - on the validation set and see how precise
#our model is.

#Validation data has 1071 observations and 13 features. As we did earlier on the build set we must
#remove features which we found high correlation or dependents variables. Resulting in 10 columns.

dim(validation)
names(validation)
validation <- select(validation,-c(area, fire_insurance, total))

#Prediction and results on validation set
results_validation<- data.frame(rf= predict(train_rf, validation), rb= predict(train_rb, validation), brnn= predict(train_brnn, validation))
predict_validation <- rowMeans(results_validation)
postResample(predict_validation, validation$rent_amount)

# Predicions by city
validation %>% mutate(y_hat= predict_validation) %>% group_by(city) %>% 
  ggplot(aes(y_hat,rent_amount)) + geom_point() + geom_smooth()+facet_wrap(~city)+theme_bw()+
  ggtitle("Final result")

#We were able to get very precise results on smaller values of rent amount, and lost precision on the higher ones. 

#The model proved to be satisfactory mainly if we took into consideration the small sample and the lack of 
#relevant features like type of place (studio, apartment, house, country house, etc), 
#neighborhood characteristics as comercial/residential area, points of interest (public transportation, 
#scholls, malls, banks, or others). 

# CONCLUSION
#This project is one part of the final evaluation of the Data Science Professional Certificate 
#from Harvardx-Edx, PH125.9x:Data Science: Capstone

#The chosen problem was the prediction of renting houses with a dataset containing information 
#about 10.692 houses in 5 different cities in Brazil and 13 different features. Data was collected 
#from a rental website, on 3/20/20, and it was available at Kaggle.

#We tested 10 machine learning models and finally have chosen the ensemble model, equal to the mean prediction of Random Forest, Random Forest Rborist
#and Bayesian Regularized Neural Networks as our final model. After applying it in a unknown dataset 
#(validation) we have the results: RMSE =  2171, Rsquared =  0.6266 and MAE = 1286. 

#Among the available features, the most important were the Property Tax, Bathroom and Hoa on most of
#machine learning models. We can improve results ensuring data quality and incrementing with other important 
#variables  such as type of location (studio, apartment, house, country house, bedroom), 
#distance to nearby points of interest: supermarket, pharmacy, public transportation, hospitals, schools, 
#shopping malls, etc., existence of leisure area, gym, swimming pool, barbecue, etc. Normally this kind 
#of information is described in an open field "Description".

