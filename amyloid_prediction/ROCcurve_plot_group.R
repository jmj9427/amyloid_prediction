setwd("C:/wd/AD/3rd_data/with_ID")
dir()

################################################################################
### A-beta Predictive model(LR) Code
################################################################################
library(dplyr)
library(tidyverse)

## a. load & exploraration for the data

df = read.csv("data_set_new_np.csv")

row.names(df) = df[, 1] ; df = df[, -1]

str(df) ; summary(df) ; sum(is.na(df))


## b. group_seg() function for grouping data by dx

dx_group = list(c("NC_old", "MCI", "AD"), # all group
                c("NC_old", "MCI"), # non demented
                c("MCI", "AD"), # CI
                "NC_old",
                "MCI")

names_group = c("all_group", "non_demented", "CI", "NC_old", "MCI")

###############   define group_seg()   ###################
group_seg = function(df, dx_group, names_group) {
  
  lst_data = list()
  n_iter = length(dx_group)
  
  for (i in 1:n_iter){
    
    lst_data[[i]] = filter(df, Group %in% dx_group[[i]]) %>%
      select(-Group)
    
    print(paste(names_group[i],"is segmented"))
    
  }
  
  names(lst_data) = names_group
  return(lst_data)
  
}
##########################################################

lst_data = group_seg(df, dx_group, names_group)


## c. stepwise feature selection with logistic regression

direction = c("both", "backward", "forward")

##################   define step_selection()   ###################
step_selection = function(lst_data, direction) {
  
  i_iter = length(lst_data) ; j_iter = length(direction)
  lst_var = list()
  
  for (i in 1:i_iter) {
    
    tmp_df = lst_data[[i]]
    
    for(j in 1:j_iter) {
      if (direction[j] != "forward") {
        
        model = glm(y ~ ., data = tmp_df, family = binomial(link = "logit"))
        model_opt = step(model, direction = direction[j])
        
      } else {
        
        model = glm(y ~ ., data = tmp_df, family = binomial(link = "logit"))
        nothing = glm(y ~ 1, data = tmp_df, family = binomial(link = "logit"))  
        model_opt = step(nothing, direction = direction[j],
                         scope = list(lower = formula(nothing),
                                      upper = formula(model)))
        
      }
      
      lst_var[[j_iter*(i-1) + j]] = attr(model_opt$terms, "term.labels")
      names(lst_var)[j_iter*(i-1) + j] = paste0(names(lst_data)[i], "_", direction[j])
      
    }
    
  }
  
  return(lst_var)
  
}
#####################################################################

lst_var = step_selection(lst_data, direction)

final_var = matrix(0, ncol = max(sapply(lst_var, length)), nrow = length(lst_var))
rownames(final_var) = names(lst_var)
for (i in 1:length(lst_var)){ final_var[i, 1:length(lst_var[[i]])] = lst_var[[i]] }

write.csv(final_var, "final_varialbes.csv")

## d. Model fitting & Summarise
library(caret)
library(ROCR)

lst_final_table = list()

n_fold = 10 ; prop_fold = .8 ; cutoff = .5

colordata=c('red', 'blue', 'purple')
##################   k-fold crossvalidation with LR   ###################
for (i in 1:n_fold) {
  
  final_table = matrix(0, nrow = length(lst_var), ncol = 8)
  
  for (j in 1:length(dx_group)) {
    
    raw_df = lst_data[[j]]
    
    for (k in 1:length(direction)) {
      
      x_set = k + length(direction) * (j - 1)
      
      df = raw_df[, c("y", lst_var[[x_set]])]
      
      #data partitioning
      set.seed(1125)
      idx_trn = createDataPartition(df$y, p = prop_fold, list = F)
      train_df = df[idx_trn, ] ; test_df = df[-idx_trn, ]
      
      #model fitting
      model_lr = glm(y ~ ., data = train_df, family = binomial(link = "logit"))
      tr_pred = predict(model_lr, newdata = train_df, type = "response")
      ts_pred = predict(model_lr, newdata = test_df, type = "response")
      
      #summarise the outcome
      #train
      tr_prlab = factor(ifelse(tr_pred >= cutoff, "pos", "neg"))
      tr_ctable = confusionMatrix(tr_prlab, train_df$y, positive = "pos")
      
      sense = tr_ctable$byClass[1]
      spec = tr_ctable$byClass[2]
      acc = tr_ctable$overall["Accuracy"]
      
      tr_predtn = prediction(tr_pred, train_df$y)
      auc = as.numeric(performance(tr_predtn, "auc")@y.values)
      
      tr_score = c(Sen = sense, Spe = spec, Acc = acc, AUC = auc)
      #test
      ts_prlab = factor(ifelse(ts_pred >= cutoff, "pos", "neg"))
      ts_ctable = confusionMatrix(ts_prlab, test_df$y, positive = "pos")
      
      sense = ts_ctable$byClass[1]
      spec = ts_ctable$byClass[2]
      acc = ts_ctable$overall["Accuracy"]
      
      ts_predtn = prediction(ts_pred, test_df$y)
      auc = as.numeric(performance(ts_predtn, "auc")@y.values)
      prf<-performance(ts_predtn, 'tpr', 'fpr')
      
      ts_score = c(Sen = sense, Spe = spec, Acc = acc, AUC = auc)
      
      #save
      final_table[x_set, 1:4] = tr_score
      final_table[x_set, 5:8] = ts_score
      
      ## j == 1 -> all group
      ## j == 2 -> non_demented
      ## j == 3 -> CI
      ## j == 4 -> NC_old
      ## j == 5 => MCI
      
      if(j == 5){
        plot(prf, col=colordata[k], lty=k,
             xlab="1 - specificity (False positivie rate)", ylab="Sensitivity (True positive rate)",main="ROC curve of MCI")
        abline(a=0,b=1)
        par(new=T)
      }
    }
    
  }
  
  lst_final_table[[i]] = final_table
  
}
#########################################################################

lst_final_table

######## summarise lst_final_table ############
x = matrix(0, nrow = length(lst_var), ncol = 8)


for (i in 1:10) {
  
  x = x + lst_final_table[[i]]
  
}

x = x / 10


z = matrix(0, nrow = length(lst_var), ncol = 10)

for (i in 1:10) {
  
  z[, i] = lst_final_table[[i]][, 8] 
  
}

final = cbind(x, apply(z, 1, max))
final
#################################################

rownames(final) = names(lst_var)
colnames(final) = c("tr_sen", "tr_spe", "tr_acc", "tr_AUC", "ts_sen", "ts_spe", "ts_acc", "ts_AUC", "max_AUC")

## 
## j==1 -> final_table[#,8] # = 1,2,3 _ all group
## j==2 -> final_table[#,8] # = 4,5,6 _ non_demented
## j==3 -> final_table[#,8] # = 7,8,9 _ CI
## j==4 -> final_table[#,8] # = 10,11,12 _ NC_old
## j==5 -> final_table[#,8] # = 13,14,15 _ MCI

legend('bottomright', legend = c(paste(direction[1], round(final_table[13,8],4)), paste(direction[2], round(final_table[14,8],4)),
                                 paste(direction[3], round(final_table[15,8],4))), 
       col=c('red', 'blue', 'purple'), lty = 1:3, lwd=2)

write.csv(final, "final_table.csv")
