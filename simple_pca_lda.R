rm(list=ls())
library(reticulate)
np <- import("numpy")
X <- np$load("dataset/X.npy")
y <- np$load("dataset/Y.npy")

dim(X)
dim(y)

# correct the rotation
rotate <- function(x) t(apply(x, 2, rev))
for (i in c(1:nrow(X))) {
  X[i,,] <- rotate(X[i,,])
}

# make the image flat to perform pca
dim(X) <- c(nrow(X), 64*64)

# display image for debug
# m = matrix(X[1,],64,64)
# par(mar=c(0, 0, 0, 0))
# image(m, useRaster=TRUE, axes=FALSE)

# range of values
summary(X[1,])

dim(X)

meanvec<-colMeans(X); sdvec<-apply(X,2,sd)
length(meanvec); min(meanvec); max(meanvec)
length(sdvec); min(sdvec); max(sdvec)

pca <- prcomp(X, scale=T)
summary(pca)
dim(pca$x)
sum(pca$sdev^2)

# plot(c(1:30),(pca$sdev^2)[1:30], main="Variances of PCs") # Screeplot with the first 30 eigenvalues
# screeplot(pca, npcs=30, type="lines") # screeplot with R-function 'screeplot'

mean(diag(pca$rotation))
diag(pca$rotation)

par(mar = c(0, 0, 0, 0))
cumulativeVariance <- cumsum(pca$sdev^2)/sum(pca$sdev^2)
plot(cumulativeVarance)

#### pick components ####

# criteria 1:
feature2Use <- which(cumulativeVariance < 0.95)
feature2Use[end(feature2Use)]


# criteria 2:
feature2Use <- which(pca$sdev^2 > mean(pca$sdev^2))
feature2Use[end(feature2Use)]

cumulativeVariance[181] # how much using above criteria

# seq.int(3, round(nrow(X) - 10)

library(MASS)

unique(max.col(y))
y <- max.col(y)

data <- data.frame(pca$x[,1:302], y)

lsc1=lda(y~., data=data)
lsc1   # study the summary of the LDA fit

yhat <- predict(lsc1,data)
# table(yhat$class, y)

mean(y == yhat$class)