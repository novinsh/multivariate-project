rm(list=ls())
data <- read.csv("usermodeling.csv", header = T)
fix(data)
nrow(data)
library(purrr)
library(tidyr)
library(ggplot2)

data %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(bins = 20)

unique(data$UNS)

plot(data$STG, data$SCG)