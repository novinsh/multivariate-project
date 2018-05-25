rm(list=ls())
library(reticulate) # for loading npy objects
library(MASS) # for LDA and QDA
library(ramify) # for argmax

np <- import("numpy")
X <- np$load("dataset/X.npy")
y <- np$load("dataset/Y.npy")
y <- argmax(y)-1
unique(y)

# the labels loaded from the dataset are not corresponding to the value shown in the picture
# map the labels to the values shown in the picture values:
idx9 <- which(y==0)
idx0 <- which(y==1)
idx7 <- which(y==2)
idx6 <- which(y==3)
idx1 <- which(y==4)
idx8 <- which(y==5)
idx4 <- which(y==6)
idx3 <- which(y==7)
idx2 <- which(y==8)
idx5 <- which(y==9)

y[idx0] <- 0
y[idx1] <- 1
y[idx2] <- 2
y[idx3] <- 3
y[idx4] <- 4
y[idx5] <- 5
y[idx6] <- 6
y[idx7] <- 7
y[idx8] <- 8
y[idx9] <- 9

#correct the rotation
rotate <- function(x) t(apply(x, 2, rev))
for (i in c(1:nrow(X))) {
  X[i,,] <- rotate(X[i,,])
}

PCA <- function(X) {
  pca <- prcomp(X, scale=T)
  
  cumulativeVariance <- cumsum(pca$sdev^2)/sum(pca$sdev^2)
  #### pick components ####
  # criteria 1:
  nrFeatures1 <- which(cumulativeVariance < 0.95)
  nrFeatures1 <- nrFeatures1[end(nrFeatures1)][1]
  nrFeatures2 <- which(cumulativeVariance < 0.90)
  nrFeatures2 <- nrFeatures2[end(nrFeatures2)][1]
  nrFeatures3 <- which(cumulativeVariance < 0.80)
  nrFeatures3 <- nrFeatures3[end(nrFeatures3)][1]
  # criteria 2:
  nrFeatures4 <- which(pca$sdev^2 > mean(pca$sdev^2))
  nrFeatures4 <- nrFeatures4[end(nrFeatures4)][1]
  #cumulativeVariance[181] # how much using above criteria
  
  return (list("pcs" = pca$x, "nr_cuts" = c(nrFeatures1, nrFeatures2, nrFeatures3, nrFeatures4)))
}

# make the image flat to perform pca
dim(X) <- c(nrow(X), 64*64)

# plot to make sure
nr_rows = 5
par(mfrow = c(nr_rows, nr_rows), mar=c(1,1,1,1))
indices2plot <- as.integer(runif(nr_rows, 1, nrow(X)))
for( i in indices2plot) {
  m = matrix(X[i,], 64, 64)
  image(m, useRaster=TRUE, axes=FALSE)
  title(paste0(y[i]), font.main=2)
}

# shuffle the data
shuffled <- sample(c(1:nrow(X)))
X <- X[shuffled,]
y <- y[shuffled]

# perform PC and obtain different cuts that explains the original data
pcaResult <- PCA(X)
pcaX <- pcaResult$pcs
pcaCuts <- pcaResult$nr_cuts

# split into train and test
ratio = 0.75
trainSize <- ceiling(ratio*nrow(pcaX))
X_train <- pcaX[1:trainSize,]
y_train <- y[1:trainSize]
X_test <- pcaX[(trainSize+1):end(pcaX)[1],]
y_test <- y[(trainSize+1):end(pcaX)[1]]

# make sure the nr samples are fine
stopifnot(nrow(X) == nrow(X_train) + nrow(X_test))
stopifnot(nrow(X) == length(y_train) + length(y_test))

best_acc <- 0
best_nr_pc <- -1

grid <- expand.grid(models=c("lda", "qda"), methods=c("moment","mle"), pcs=pcaCuts)
errors = c()
for (r in c(1:nrow(grid))) {
  model = grid$models[r]
  method = grid$methods[r]
  nr_features = grid$pcs[r]
  
  if (nr_features > 130 & model == "qda") {
    nr_features = 50
    grid$pcs[r] = nr_features
  }
  train <- data.frame(X_train[,1:nr_features], y_train)
  cat("\n")
  print(paste("model=", model, ", method=", method, ", nr_features=", nr_features))
  if (model == "lda") {
    theModel = lda(y_train~., data=train, CV=T, method=as.character(method))
  } else {
    theModel = qda(y_train~., data=train, CV=T, method=as.character(method))
  }
  acc <- mean(theModel$class==y_train)
  aper <- 1 - mean(theModel$class==y_train)
  print(paste("acc: ", acc, " -- aper: ", aper))
  errors <- c(errors, aper)
}

best_model_idx = which.min(errors)
best_model_name = grid$models[best_model_idx]
best_method = grid$methods[best_model_idx]
best_nr_features = grid$pcs[best_model_idx]

train <- data.frame(X_train[,1:best_nr_features], y_train)
test <- data.frame(X_test[,1:best_nr_features], y_test)
if (best_model_name == "lda") {
  best_model =lda(y_train~., data=train, method=as.character(method))
} else {
  best_model =qda(y_train~., data=train, method=as.character(method))
}
yhat <- predict(best_model, test)
table(y_test, yhat$class)
mean(yhat$class==y_test)
aper = 1 - mean(yhat$class==y_test)
aper

misclass <- which(yhat$class!=y_test)

# display the misclassifications
nr_rows = ceiling(sqrt(length(misclass)))
par(mfrow = c(nr_rows, nr_rows))

for( i in c(1:length(misclass))) {
  m = matrix(X[misclass[i],], 64, 64)
  # plot.new()
  # plot.window(xlim=c(0, 1), ylim=c(0, 1), asp=1)
  image(m, useRaster=TRUE, axes=FALSE)
  title(paste0(yhat$class[misclass[i]],' > ', y_test[misclass[i]]), font.main=2)
}

misclass