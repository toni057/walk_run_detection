###########################################################################################################
################################################ functions ################################################
###########################################################################################################

#' get.ROC.df
#'
#' calculates ROC curve data for plotting
get.ROC.df <- function(fit, lab, set) {
   pred <- prediction(fit, lab)
   perf <- performance(pred, "tpr", "fpr")
   
   data.frame(x = perf@x.values[[1]],
              y = perf@y.values[[1]],
              xlab = perf@x.name,
              ylab = perf@y.name, 
              set = set)
}

#' get.AUC
#'
#' calculates AUC
get.AUC <- function(fit, lab, set) {
   pred <- prediction(fit, lab)
   perf <- performance(pred, "auc")
   perf@y.values[[1]]
}



#' eval.model
#' 
#' plots ROC curves for training validation sets
eval.model <- function(fit.tr, lab.tr, fit.te, lab.te) {
   auc.train <- get.AUC(fit.tr, lab.tr)
   auc.test <- get.AUC(fit.te, lab.te)
   
   rbind.data.frame(get.ROC.df(fit.tr, lab.tr, sprintf('train auc = %0.5f', auc.train)),
                    get.ROC.df(fit.te, lab.te, sprintf('test auc = %0.5f', auc.test))) %>%
      ggplot(data = ., mapping = aes(x = x, y = y, color = set, group = set)) +
      geom_line() +
      coord_fixed() + 
      facet_grid(~set)
}






