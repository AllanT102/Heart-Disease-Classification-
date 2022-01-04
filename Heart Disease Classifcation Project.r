#Allan Tan
#December 9, 2021
#load all packages
library(tidymodels)
library(tidyverse)
library(repr)

#read csv file that is downloaded from UCI

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

download.file(url, "heart.csv")
heart <- read_csv("heart.csv")

#WRANGLING DATA

set.seed(1234)
# add meaningful column names to the data set, as the dataset originally had none, we named the predicted class hd as the one on the 
# website "num" had little meaning
colnames(heart) <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd")

# change chr to dbl
heart$ca <- as.numeric(heart$ca)
heart$thal <- as.numeric(heart$thal)

# since predicting the "hd" or whether or not the patient has heart disease, we want to change it into a factor
heart$hd <- as.factor(heart$hd)

# see that for the hd column, there are numbers other than 0 and 1, we want to get rid of those so that we can predict whether 
# or not the patient has heart disease or not to do this, we reassign values of 4, 3, and 2 with 1 because numbers > 1 also 
# just mean that the patient has heart disease

heart$hd[heart$hd== "4"]<- "1"
heart$hd[heart$hd== "3"]<- "1"
heart$hd[heart$hd== "2"]<- "1"

set.seed(1234)
# number of rows with missing data
narows <- sum(is.na(heart))
nrow(heart)
#remove all with na
heart <- na.omit(heart)


#Begin building classification model using KNN algorithm


set.seed(1234)
#split into training and testing sets with 0.7 as the proportion to maintain high accuracy by having sufficient training and testing data
heart_split <- initial_split(heart, prop = 0.70, strata = hd)
heart_train <- training(heart_split)
heart_test <- testing(heart_split)

set.seed(1234)
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
            set_engine("kknn") %>%
            set_mode("classification")

#heart recipe with all 3 predictors
heart_recipe_with_3 <- recipe(hd ~ thalach+chol+trestbps, data = heart_train)%>%
step_scale(all_predictors())%>%
step_center(all_predictors())

set.seed(1234)
#10 fold cross validation
heart_vfold <- vfold_cv(heart_train, v = 10, strata = hd)

#k values we want to test 1 through 10
k_vals <- tibble(neighbors = seq(1:50))

#workflow
knn_results_3 <- workflow() %>%
                 add_recipe(heart_recipe_with_3) %>%
                 add_model(knn_spec) %>%
                 tune_grid(resamples = heart_vfold, grid = k_vals) %>%
                 collect_metrics()

#filter out accuracy in metrics
accuracies_3 <- knn_results_3 %>%
                 filter(.metric == 'accuracy')

#plot accuracy vs k
cross_val_plot_3 <- ggplot(accuracies_3, aes(x = neighbors, y = mean)) +
                  geom_point() +
                  geom_line() + 
                  labs(x = 'Neighbors', y = 'Accuracy Estimate') +
                  theme(text = element_text(size = 20)) +
                  scale_y_continuous(limits = c(0.4, 0.9))

                  set.seed(1234)

#using 3 predictors made our accuracy percentage worse, therefore, we tested out
#and narrowed it down to two predictors

#best recipe

heart_recipe <- recipe(hd ~ thalach+trestbps, data = heart_train)%>%
step_scale(all_predictors())%>%
step_center(all_predictors())

#workflow
knn_results <- workflow() %>%
                 add_recipe(heart_recipe) %>%
                 add_model(knn_spec) %>%
                 tune_grid(resamples = heart_vfold, grid = k_vals) %>%
                 collect_metrics()

#filter out accuracy in metrics
accuracies <- knn_results %>%
                 filter(.metric == 'accuracy')

#plot accuracy vs k
cross_val_plot <- ggplot(accuracies, aes(x = neighbors, y = mean)) +
                  geom_point() +
                  geom_line() + 
                  labs(x = 'Neighbors', y = 'Accuracy Estimate') +
                  theme(text = element_text(size = 20)) + 
                  scale_y_continuous(limits = c(0.4, 0.9))

best_k <- accuracies %>% arrange(desc(mean))%>% slice(1)
best_k

set.seed(1234)
#new spec
best_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 25)  %>%
            set_engine("kknn") %>%
            set_mode("classification")

#new workflow with old recipe
knn_best_results <- workflow() %>%
                 add_recipe(heart_recipe) %>%
                 add_model(best_spec) %>%
                 fit(data = heart_train)

#predict 
heart_predictions <- knn_best_results %>%
                 predict(heart_test)%>%
                 bind_cols(heart_test)

heart_metrics <- heart_predictions %>%
metrics(truth = hd, estimate = .pred_class)

heart_conf_mat <- heart_predictions %>%
conf_mat(truth = hd, estimate = .pred_class)

#turn confusion matrix into dataframe
Truth <- factor(c(0, 0, 1, 1))
Predicted <- factor(c(0, 1, 0, 1))
Y      <- c(11,36,26,15)
conf_mat <- data.frame(Truth, Predicted, Y)

#plot as tiles
confmatplot <- ggplot(data =  conf_mat, mapping = aes(x = Truth, y = Predicted)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
  scale_fill_gradient(low = "yellow", high = "green") +
  theme_bw() + theme(legend.position = "none") +
  ggtitle("Confusion Matrix")

confmatplot
heart_conf_mat
heart_metrics





















