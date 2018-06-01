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
par(mfrow=c(1,1), mar=c(1,1,1,1))
hist(y)
