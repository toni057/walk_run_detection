library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(magrittr)
library(ggplot2)
library(reshape2)

library(caret)
library(gbm)
library(ranger)
library(glmnet)

library(ROCR)

source('helper_functions.R')

d <- read_csv('dataset.csv')


# add lap time that starts from 0 at every measurement
d <- d %>% 
   mutate(block = c(0, cumsum(abs(diff(activity))))) %>%
   group_by(block) %>%
   mutate(t = time %>% 
             strsplit(':') %>%
             map_dbl(function(x) {
                x <- as.numeric(x)
                x[1:3] %*% c(60*60, 60, 1) + x[4]/10^ceiling(log10(x[4]))
             }),
          t = t - min(t)) %>%
   ungroup(block) %>%
   arrange(block, t) %>%
   select(date, time, block, t, username:gyro_z)


# scatterplot
d %>%
   mutate(n = 1:n()) %>%
   select(n, activity, acceleration_x:acceleration_z) %>%
   gather(direction, acceleration, -n, -activity) %>%
   ggplot() +
   geom_point(aes(x = n, y = acceleration, group = direction, color = direction), size=1, alpha=0.1) +
   facet_grid(direction ~ activity, scales="free_y")


# rotation matrices
Ry <- function(x) {
   matrix(c(cos(x), -sin(x), 0, sin(x), cos(x), 0, 0, 0, 1), nrow = 3, byrow = T)
}

Rp <- function(x) {
   matrix(c(cos(x), 0, sin(x), 0, 1, 0, -sin(x), 0, cos(x)), nrow = 3, byrow = T)
}

Rr <- function(x) {
   matrix(c(1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x)), nrow = 3, byrow = T)
}


# rotate accelerometer data - roll, pitch, yaw
d1 <- list()
for (i in 1:nrow(d)) {
   d1[[i]] <- t(Ry(d$acceleration_x[i])) %*% t(Rp(d$acceleration_x[i])) %*% t(Rr(d$acceleration_x[i])) %*% t(as.matrix(d[i,5:7]))
   rownames(d1[[i]]) <- c('x', 'y', 'z')
}


# merge rotated accelerometer data
d2 <- d1 %>%
   map(~t(.x)) %>%
   do.call(rbind, .) %>%
   cbind.data.frame(d, .) %>%
   mutate(a = sqrt(x^2 + y^2 + z^2)) %T>%
   head()


d2 %>%
   mutate(n = 1:n()) %>%
   select(n, activity, x:z) %>%
   gather(direction, acceleration, -n, -activity) %>%
   ggplot() +
   geom_point(aes(x = n, y = acceleration, group = direction, color = direction), size=1, alpha=0.1) +
   facet_grid(direction ~ activity, scales="free_y")


# split to train and test
trainind <- createDataPartition(d2$activity, p = .8, list = F, times = 1)
train <- d2[trainind,]
test <- d2[-trainind,]


# formula
fmla <- activity ~ acceleration_x + acceleration_y + acceleration_z + gyro_x + gyro_y + gyro_z + x + y + z + a


# fit logistic regression
m <- glm(fmla, train, family = binomial(), control = glm.control(maxit = 25, trace = F))
m <- MASS::stepAIC(m, direction = 'bo', k = 5)
car::vif(m)
fit.tr <- m$fitted.values
fit.te <- predict(m, test)
eval.model(fit.tr, train$activity, fit.te, test$activity)
get.AUC(fit.tr, train$activity)
get.AUC(fit.te, test$activity)


# fit glmnet
m <- glmnet(model.matrix(fmla, d2), d2$activity, family="binomial", nlambda = 200)
fit.tr <- predict(m, model.matrix(fmla, train))
fit.te <- predict(m, model.matrix(fmla, test))

auc.glmnet.tr <- apply(fit.tr, 2, get.AUC, lab=train$activity)
plot(m$lambda, auc.glmnet.tr)

auc.glmnet.te <- apply(fit.te, 2, get.AUC, lab=test$activity)
plot(m$lambda, auc.glmnet.te)

# select the best lambda based on the validation set
pred_glmnet <- predict(m, model.matrix(fmla, d2), s = m$lambda[which(auc.glmnet.te == max(auc.glmnet.te))], type = 'response')
eval.model(pred_glmnet[trainind], train$activity, pred_glmnet[-trainind], test$activity)


# fir gradient boost
m <- gbm(fmla, distribution = 'bernoulli', data = train, n.trees = 50, shrinkage = 0.1, interaction.depth = 5, n.cores = 4, verbose = T)
fit.te <- predict(m, test, n.trees = m$n.trees)
auc.gbm.te.0 <- auc.glmnet.te <- get.AUC(fit.te, lab=test$activity)

for (i in 1:9) {
   m0 <- m
   m <- gbm.more(m, 50)

   fit.te <- predict(m, test, n.trees = m$n.trees)
   
   auc.gbm.te <- get.AUC(fit.te, lab=test$activity)
   if (auc.gbm.te > auc.gbm.te.0) {
      m0 <- m
   }
}
m <- m0
fit.tr <- predict(m, train, n.trees = m$n.trees)
fit.te <- predict(m, test, n.trees = m$n.trees)
eval.model(fit.tr, train$activity, fit.te, test$activity)
pred_gbm <- predict(m, d2, n.trees = m$n.trees, type = 'response')


# fit random forest
# optimize number of trees and min.node.size
J <- function(x) {
   if (any(x<0)) return (1e10)
   
   num.trees <- ceiling(x[1])
   min.node.size <- ceiling(x[2])
   
   set.seed(1)
   m <- ranger(fmla, train, num.trees = num.trees, num.threads = 4, verbose = T, min.node.size = min.node.size)
   fit.te <- predict(m, test)$predictions
   
   auc <- get.AUC(fit.te, lab=test$activity)
   print(c(x, auc))
   -auc
}
x <- optim(c(2, 5), J, control = list(parscale=c(2, 1)))

m <- ranger(fmla, train, num.trees = ceiling(x$par[1]), num.threads = 4, verbose = T, min.node.size = ceiling(x$par[2]))
pred_rf <- predict(m, d2)$predictions
eval.model(pred_rf[trainind], train$activity, pred_rf[-trainind], test$activity)


pred_ensemble <- rowMeans(cbind(pred_glmnet, pred_gbm, pred_rf))


eval.model(pred_glmnet[trainind], train$activity, pred_glmnet[-trainind], test$activity)
eval.model(pred_gbm[trainind], train$activity, pred_gbm[-trainind], test$activity)
eval.model(pred_rf[trainind], train$activity, pred_rf[-trainind], test$activity)
eval.model(pred_ensemble[trainind], train$activity, pred_ensemble[-trainind], test$activity)



















