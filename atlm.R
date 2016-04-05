coc81 = read.csv("coc81.csv")
col_names = colnames(coc81)
factors = col_names[seq(1,22)]
local_factor <- function(arg1) {
  factorized = factor(arg1, c(seq(0:6)))
  return(factorized)
}
#coc81[, factors] = lapply(coc81[, factors], factor)
#head(coc81)
rows = coc81[sample(nrow(coc81)),]
folds = cut(seq(1,nrow(rows)),breaks=10,labels=FALSE)

pass <- function(arg1) {
  return(arg1)
}

square <- function(arg1) {
  sq = arg1^2
  return(sq)
}

transforms = c(pass, log, sqrt)
inverse_transforms = c(pass, exp, square)

iteration <- function(dTrain, dTest) {
  fns = c()
  ifns = c()
  col_names = colnames(coc81)[seq(1,24)]
  model_formula = paste(col_names[24], "~")
  isFirst = TRUE
  for (col_num in 1:23) {
    if ((col_num <= 23) && (length(unique(trainData[[col_num]])) > 1)) {
      if (isFirst) {
        model_formula = paste(model_formula, col_names[col_num])  
        isFirst = FALSE
      } else {
        model_formula = paste(model_formula, "+", col_names[col_num]) 
      }
    }
    if (col_num <= 22) {
      # Factor
      fns = c(fns, pass)
      ifns = c(ifns, pass)
      ids = which(!(dTest[[col_num]] %in% dTrain[[col_num]]))
      dTest[[col_num]][ids] = NA
      dTrain[[col_num]] = as.factor(dTrain[[col_num]])
      dTest[[col_num]] = as.factor(dTest[[col_num]])
    } else {
      # Continuous
      column = dTrain[,col_num]
      lowestSkew = 1000000
      index = 0
      for (i in 1:3) {
        skew = skewness(sapply(column, transforms[[i]]))
        if (skew < lowestSkew) {
          lowestSkew = skew
          index = i
        }
      }
      fns = c(fns, transforms[[index]])
      ifns = c(ifns, inverse_transforms[[index]])
    }
    dTrain[[col_num]] = sapply(dTrain[[col_num]], fns[[col_num]])
    dTest[[col_num]] = sapply(dTest[[col_num]], fns[[col_num]])
  }
  lm_model = lm(model_formula, trainData)
  predicted = predict(lm_model, testData)
  #predicted = sapply(predicted, ifns[[24]])
  #actuals = sapply(testData[[24]], ifns[[24]])
  actuals = testData[[24]]
  total_mre = 0
  #print("predicted", predicted)
  #print("actuals", actuals)
  for (j in 1:length(predicted)) {
    mre = abs(actuals[j] - predicted[j])/actuals[j]
    total_mre = total_mre + mre
  }
  mmre = total_mre / length(predicted)
  print(mmre[1])
  return(mmre)
}

errors = c(0, length=10)
for (i in 1:10) {
  testIndices = which(folds==i, arr.ind = TRUE)
  testData = rows[testIndices,]
  trainData = rows[-testIndices,]
  #testData[, factors] = lapply(testData[, factors], factor)
  #trainData[, factors] = lapply(trainData[, factors], factor)
  error = iteration(trainData, testData)
  errors[i] = error
}
print(errors)
print(paste("Mean =", mean(errors)))
print(paste("STD =", sd(errors)))