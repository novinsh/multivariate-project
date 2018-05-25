##
# Since QDA was the best model based on experiment on discriminantAnalysis with PCA
# Fitted QDA on a the whole dataset

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

# perform PC and obtain different cuts that explains the original data
pcaResult <- PCA(X)
pcaX <- pcaResult$pcs
pcaCuts <- pcaResult$nr_cuts

data <- data.frame(pcaX[,1:140], y)
model = qda(y~., data=data, method="moment")
yhat <- predict(model, data)
mean(yhat$class==y)
sum(yhat$class!=y)
