rm(list=ls())
library(reticulate) # for loading npy objects
library(MASS) # for LDA and QDA
library(ramify) # for argmax

set.seed(0) # make results reproducible

# load dataset
np <- import("numpy")
X <- np$load("dataset/X.npy")
y <- np$load("dataset/Y.npy")
y <- argmax(y)-1
unique(y)

# the labels loaded from the dataset are not corresponding to the value shown in 
# the image map the labels to the values shown in the picture values:
idx9 <- which(y==0); idx0 <- which(y==1); idx7 <- which(y==2)
idx6 <- which(y==3); idx1 <- which(y==4); idx8 <- which(y==5)
idx4 <- which(y==6); idx3 <- which(y==7); idx2 <- which(y==8)
idx5 <- which(y==9)

y[idx0] <- 0; y[idx1] <- 1; y[idx2] <- 2; y[idx3] <- 3; y[idx4] <- 4
y[idx5] <- 5; y[idx6] <- 6; y[idx7] <- 7; y[idx8] <- 8; y[idx9] <- 9

#correct the rotation
rotate <- function(x) t(apply(x, 2, rev))
for (i in c(1:nrow(X))) {
  X[i,,] <- rotate(X[i,,])
}

# make the image flat to perform pca
dim(X) <- c(nrow(X), 64*64)

# shuffle the data for identical distribution among different splits
shuffled <- sample(c(1:nrow(X)))
X <- X[shuffled,]
y <- y[shuffled]

# plot to make sure of the performed operations
nr_rows = 5
par(mfrow = c(nr_rows, nr_rows), mar=c(1,1,1,1))
indices2plot <- as.integer(runif(nr_rows*nr_rows, 1, nrow(X)))
for( i in indices2plot) {
  m = matrix(X[i,], 64, 64)
  image(m, useRaster=TRUE, axes=FALSE)
  title(paste0(y[i]), font.main=2)
}

# perform PCA and pick a number of PCs as features
pca <- prcomp(X, scale=T)
pcCuts <- c(70, 50, 30, 10) # different number of pcs
cumulativeVariance <- cumsum(pca$sdev^2)/sum(pca$sdev^2) # cumulative variance

# split into train and test
ratio = 0.75
trainSize <- ceiling(ratio*nrow(pca$x))
X_train <- pca$x[1:trainSize,]
y_train <- y[1:trainSize]
X_test <- pca$x[(trainSize+1):end(pca$x)[1],]
y_test <- y[(trainSize+1):end(pca$x)[1]]

# make sure the nr samples are fine
stopifnot(nrow(X) == nrow(X_train) + nrow(X_test))
stopifnot(nrow(X) == length(y_train) + length(y_test))

# crossvalidation and finding the best number of features
for (nr_features in pcCuts) {
  train <- data.frame(X_train[,1:nr_features], y_train)
  varExplained <- sum(cumulativeVariance[1:nr_features])
  cat("variance explained by first ", nr_features, " components: ", varExplained, "\n")
  qdaModel = qda(y_train~., data=train, CV=T, method="mle")
  acc <- mean(qdaModel$class==y_train)
  aper <- 1 - mean(qdaModel$class==y_train)
  print(paste("nr_features: ", nr_features, " -- acc: ", round(acc, 4), " -- aper: ", round(aper, 4)))
}

# prediction with QDA
best_nr_features = 50
train <- data.frame(X_train[,1:best_nr_features], y_train)
test <- data.frame(X_test[,1:best_nr_features], y_test)
qdaModel = qda(y_train~., data=train, method="mle")
yhat <- predict(qdaModel, test)

table(y_test, yhat$class)
mean(yhat$class==y_test)
aper = 1 - mean(yhat$class==y_test)
aper

misclass <- which(yhat$class!=y_test)
misclass <- which(yhat$class!=y_test)

# display the misclassifications
nr_rows = 5#ceiling(sqrt(length(misclass)))
par(mfrow = c(5, 8))

for( i in c(1:40)) {
  if (i > length(misclass)) {
    break
  }
  m = matrix(X[misclass[i],], 64, 64)
  # plot.new()
  # plot.window(xlim=c(0, 1), ylim=c(0, 1), asp=1)
  image(m, useRaster=TRUE, axes=FALSE)
  title(paste0(yhat$class[misclass[i]],' > ', y_test[misclass[i]]), font.main=2)
}