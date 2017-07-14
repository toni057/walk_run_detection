# walk_run_detection

library(dplyr)
library(readr)
library(purrr)
library(magrittr)
library(FactoMineR)
library(factoextra)

d <- read_csv('dataset.csv.zip')



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
# m <- glm(activity ~ acceleration_x + acceleration_y + acceleration_z + gyro_x + gyro_y + gyro_z, d2, family = binomial())
# m <- glm(activity ~ x + y + z, d2, family = binomial())
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

pred_tr <- prediction(trfit, train$activity)
perf <- performance(pred,"tpr","fpr")
plot(perf)

pred_te <- prediction(tefit, test$activity)
perf <- performance(pred,"tpr","fpr")
plot(perf)

performance(pred_tr,"auc")
performance(pred_te,"auc")

perf1 <- performance(pred_tr, "prec", "rec")
plot(perf1)
perf1 <- performance(pred_te, "prec", "rec")
plot(perf1)



