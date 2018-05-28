rm(list=ls())
library(reticulate) # for loading npy objects
library(MASS) # for LDA and QDA
library(ramify) # for argmax

set.seed(0)

np <- import("numpy")
X <- np$load("dataset/X.npy")
y <- np$load("dataset/Y.npy")
y <- argmax(y)-1

#correct the rotation
rotate <- function(x) t(apply(x, 2, rev))
for (i in c(1:nrow(X))) {
  X[i,,] <- rotate(X[i,,])
}

# make the image flat to perform pca
dim(X) <- c(nrow(X), 64*64)

# Let's visualize a result of PCA
pca_res <- prcomp(X, scale=T)
# Calculate variance
var <- pca_res$sdev ^ 2
# Calculate proportion of explained variance
prop_var <- var/sum(var)

par(mfrow=c(2,2))
plot(prop_var,xlab="Principal component", ylab="Proportion of variance explained", ylim=c(0,1), type='line')
plot(cumsum(prop_var),xlab="Principal component", ylab="Cumulative Proportion of variance explained", ylim=c(0,1), type='line')
screeplot(pca_res)
screeplot(pca_res,type="l")
par(mfrow=c(1,1))


# Density plot
size=3
par(mfrow=c(size, size), mar=c(2,2,2,2))
colnames <- colnames(pca_res$x)
for (i in 1:(size*size)) {
  d <- density(pca_res$x[,i])
  plot(d, type="n", main=colnames[i])
  polygon(d, col="red", border="gray")
  rug(pca_res$x[,i], col="blue")
}

# Histograms and density lines
size=3
par(mfrow=c(size, size))
colnames <- colnames(pca_res$x)
for (i in 1:(size*size)) {
  hist(pca_res$x[,i], main=colnames[i], probability=TRUE, col="gray", border="white")
  d <- density(pca_res$x[,i])
  lines(d, col="red")
  rug(pca_res$x[,i], col="blue")
}

