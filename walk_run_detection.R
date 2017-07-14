library(dplyr)
library(readr)
library(purrr)
library(magrittr)
library(ggplot2)

library(caret)
library(gbm)
library(ranger)

library(ROCR)
library(FactoMineR)
library(factoextra)


d <- read_csv('dataset.csv')



Ry <- function(x) {
   matrix(c(cos(x), -sin(x), 0, sin(x), cos(x), 0, 0, 0, 1), nrow = 3, byrow = T)
}

Rp <- function(x) {
   matrix(c(cos(x), 0, sin(x), 0, 1, 0, -sin(x), 0, cos(x)), nrow = 3, byrow = T)
}

Rr <- function(x) {
   matrix(c(1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x)), nrow = 3, byrow = T)
}


d1 <- list()
for (i in 1:nrow(d)) {
   d1[[i]] <- t(Ry(d$acceleration_x[i])) %*% t(Rp(d$acceleration_x[i])) %*% t(Rr(d$acceleration_x[i])) %*% t(as.matrix(d[i,5:7]))
   rownames(d1[[i]]) <- c('x', 'y', 'z')
}


d2 <- d1 %>%
   map(~t(.x)) %>%
   do.call(rbind, .) %>%
   cbind.data.frame(d, .) %T>%
   head()


fmla <- activity ~ acceleration_x + acceleration_y + acceleration_z + gyro_x + gyro_y + gyro_z + x + y + z

m <- glm(fmla, d2, family = binomial())
fitted <- m$fitted.values


pca <- PCA(d2[,5:13], graph = F, ncp = 9)
fviz_pca(pca)
fviz_eig(pca)


trainind <- createDataPartition(d2$activity, p = .8, list = F, times = 1)
train <- d2[trainind,]
test <- d2[-trainind,]

m <- gbm(fmla, distribution = 'bernoulli', data = train, n.trees = 100, shrinkage = 0.1, interaction.depth = 5, n.cores = 4, verbose = T)
trfit <- m$fit
tefit <- predict(m, test, n.trees = m$n.trees)

m <- ranger(fmla, train, num.trees = 1000, num.threads = 4, verbose = T, min.node.size = 5)
trfit <- m$predictions
tefit <- predict(m, test)$predictions


pred_tr <- prediction(trfit, train$activity)
perf <- performance(pred_tr,"tpr","fpr")
plot(perf)

pred_te <- prediction(tefit, test$activity)
perf <- performance(pred_te,"tpr","fpr")
plot(perf)

auc_tr <- performance(pred_tr,"auc")
auc_te <- performance(pred_te,"auc")

perf1 <- performance(pred_tr, "prec", "rec")
plot(perf1)
perf1 <- performance(pred_te, "prec", "rec")
plot(perf1)




get.ROC.df <- function(fit, lab, set) {
   pred <- prediction(fit, lab)
   perf <- performance(pred, "tpr", "fpr")
   
   data.frame(x = perf@x.values[[1]],
              y = perf@y.values[[1]],
              xlab = perf@x.name,
              ylab = perf@y.name, 
              set = set)
}

rbind.data.frame(get.ROC.df(trfit, train$activity, 'train'),
                 get.ROC.df(tefit, test$activity, 'test')) %>%
   ggplot(data = ., mapping = aes(x = x, y = y, color = set, group = set)) +
   geom_line() + 
   coord_fixed() +
   labs(title='ROC')
























