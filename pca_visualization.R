rm(list=ls())
library(reticulate) # for loading npy objects
library(MASS) # for LDA and QDA
library(ramify) # for argmax

set.seed(0)

np <- import("numpy")
X <- np$load("dataset/X.npy")

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

