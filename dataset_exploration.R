rm(list=ls())
library(reticulate) # for loading npy objects
library(MASS) # for LDA and QDA
library(ramify) # for argmax

set.seed(0)

np <- import("numpy")
X <- np$load("dataset/X.npy")
y <- np$load("dataset/Y.npy")
y <- argmax(y)-1 # turn the labels from one-hot vector to scalar

dim(X)
length(y)

options(max.print=1000000)
summary(X) # to see the scale of the varibales
table(y) # distribution of labels

#correct the rotation
rotate <- function(x) t(apply(x, 2, rev))
for (i in c(1:nrow(X))) {
  X[i,,] <- rotate(X[i,,])
}

# make the image flat to perform pca
dim(X) <- c(nrow(X), 64*64)

# plot to make sure of the performed operations
nr_rows = 5
par(mfrow = c(nr_rows, nr_rows), mar=c(1,1,1,1))
indices2plot <- as.integer(runif(nr_rows*nr_rows, 1, nrow(X)))
for( i in indices2plot) {
  m = matrix(X[i,], 64, 64)
  image(m, useRaster=TRUE, axes=FALSE)
  title(paste0(y[i]), font.main=2)
} 
# TODO: as a result of this plot we can see that the labels and the digit 
# demonstrated in the image are not matching - need to map to the correct digits!

# distribution of 9 random pixels in the image
size=3
par(mfrow=c(size, size), mar=c(2,2,2,2))
randomPixels <- sample(c(1:ncol(X)), size=size*size)
pixels <- c(100, 300, 500, 1400, 1600, 1800, 2800, 3000, 3200)
for (i in pixels) {
  title <- paste("pixel ", as.character(i))
  hist(X[,i], main=title, probability=TRUE, col="gray", border="white")
  d <- density(X[,i])
  lines(d, col="red")
  rug(X[,i], col="blue")
}

# example
par(mfrow = c(2, 5), mar=c(1,1,1,1))
#indices2plot <- as.integer(runif(nr_rows*nr_rows, 1, nrow(X)))
for( i in seq(0,9)) {
  idx <- which(y == i)[1]
  m = matrix(X[idx,], 64, 64)
  image(m, useRaster=TRUE, axes=FALSE)
  title(paste0(y[idx]), font.main=2)
}