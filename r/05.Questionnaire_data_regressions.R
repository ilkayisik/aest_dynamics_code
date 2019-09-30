rm(list=ls()) # clear working directory
# set working directory
setwd ("/Users/ilkay.isik/aesthetic_dynamics/")
#################### LOAD LIBRARIES ####################
library(readr)
require(ggplot2)
library(xtable)
require(reshape2)
library(MASS)
library(QuantPsyc)
library(car)
library(tidyr)
library(lm.beta)
#################### LOAD THE DATA ####################
ques_data <- read_delim("data/questionnaire_data.csv", delim=";", skip=0)
mean_rmsd <- read_delim("data/mean_rmsd.csv", delim=",", skip=0)
mean_rmsd$group = NULL
mean_odata <- read_delim("data/mean_odata.csv",  delim=";",skip=0)
mean_odata$group = NULL
# Put all data together
mydata <- cbind(ques_data, mean_rmsd, mean_odata)
# Plot the histograms for the questionnaire data
ques_vars <- c("panas_pos", "panas_neg", "shaps", "stai_s", "stai_t", "area_tot", 
               "odata_mean")
df_ques <- mydata[ques_vars]

ggplot(gather(df_ques), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

# zsore the ques data
df_ques_z <- df_ques
df_ques_z [ques_vars] <- scale(df_ques_z[ques_vars]) 
ggplot(gather(df_ques_z), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

# transform some variables to be normally distributed
lambdaList <- boxcox(panas_neg~odata_mean, data=df_ques)
(lambda <- lambdaList$x[which.max(lambdaList$y)])
df_ques$panas_neg_tr <- 1/(df_ques$panas_neg^2)

lambdaList <- boxcox(shaps~odata_mean, data=df_ques)
(lambda <- lambdaList$x[which.max(lambdaList$y)])
df_ques$shaps_tr <- df_ques$shaps^2

ggplot(gather(df_ques), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

# plot the relationship bw OVERALL DATA and Questionnaire Responses
vars1 <- c("area_app", "area_exp", "area_eng", "area_tot" , 
           "panas_neg", "panas_pos", "shaps", "stai_s", "stai_t",
           "odata_mean")

df_mean <- mydata[vars1]
# melt the data for the geom jitter plot
df_mean_ = melt(df_mean, id.vars='odata_mean')

ggplot(df_mean_) +
  geom_jitter(aes(value, odata_mean, colour=variable)) + geom_smooth(aes(value, odata_mean, colour=variable), 
                                                                      method=lm, se=FALSE) + theme_bw() +
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Questionnaire Scores", y = "Mean Overall Rating")

# plot the relationship bw Mean RMSD and Questionnaire Responses: retest for both groups
vars2 <- c("area_app", "area_exp", "area_eng",  "area_tot" , 
           "panas_neg", "panas_pos", "shaps", "stai_s", "stai_t", 
           "mean_rmsd_rt")
df_rmsd_retest <- mydata[vars2]
df_rmsd_retest_ = melt(df_rmsd_retest, id.vars="mean_rmsd_rt")
ggplot(df_rmsd_retest_) +
  geom_jitter(aes(value, mean_rmsd_rt, colour=variable)) + geom_smooth(aes(value, mean_rmsd_rt, colour=variable), 
                                                                       method=lm, se=FALSE) + 
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Questionnaire Scores", y = "Mean RMSD - Retest")


################################################################################################################################
# MULTIPLE REGRESSIONS  between overall data and questionnaire scores
odata.mr <- lm(odata_mean ~ panas_pos + panas_neg + shaps +
                 stai_s + stai_t + area_tot, data=df_mean)
summary(odata.mr)
# print out the coefficients 
# standardized betas and Ci's were added to that table using excel
# with the calculations below
coefs<-xtable(coef(summary(odata.mr)),digits=c(3,3,3,3,3))
print.xtable(coefs, type="html", file="output/tables/S3.Table_mreg_results_questionnaire-meanoveralldata.html")

# standardized betas
# reporting this in the manuscript:
lm.beta(odata.mr)
# Confident intervals
confint(odata.mr)
cinfs <- xtable(confint(odata.mr),digits=c(2,2,2))
# print.xtable(cinfs, type="html", file="output/MultipleRegressionResults_ConfInts_QuesVsMeanOverallData.html")

# MULTIPLE REGRESSION  between rmsd and questionnaire scores
rmsd.mr <- lm(mean_rmsd_rt ~  panas_pos + panas_neg + shaps +
                stai_s + stai_t + area_tot, data=df_rmsd_retest)

coefs2<-xtable(coef(summary(rmsd.mr)),digits=c(3,3,3,3,3))
# print out the coefficients
# standardized betas and Ci's were added to that table using excel
print.xtable(coefs2, type="html", file="output/tables/S4.Table_mreg_results_questionnaire-meanrmsd.html")
summary(rmsd.mr)
lm.beta(rmsd.mr)
# cinf2 <- xtable(confint(rmsd.mr),digits=c(2,2,2))
# print.xtable(cinf2, type="html", file="MultipleRegressionResults_ConfInts_QuesVsMeanRMSD.html")

##########################################################################################
# Relationship between observers??? self-reported category preferences and overall ratings
##### LSCP ##### 
vars_lsp <- c("lsp_like", "lsp_int", "odata_mean_lsp",  "mean_rmsd_lsp_rt")
df_lsp <- mydata[vars_lsp]
# mean overall data
df_lsp_ = melt(df_lsp, id.vars='odata_mean_lsp')
ggplot(df_lsp_) +
  geom_jitter(aes(value, odata_mean_lsp, colour=variable)) + geom_smooth(aes(value, odata_mean_lsp, colour=variable), 
                                                                         method=lm, se=FALSE) + 
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Lsp Liking", y = "Mean Overall Rating for Landscape Videos")
# mean rmsd
df_lsp_ = melt(df_lsp, id.vars='mean_rmsd_lsp_rt')
ggplot(df_lsp_) +
  geom_jitter(aes(value, mean_rmsd_lsp_rt, colour=variable)) + geom_smooth(aes(value, mean_rmsd_lsp_rt, colour=variable), 
                                                                           method=lm, se=FALSE) + 
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Lsp Liking", y = "Mean rmsd Rating for lsp Videos")

# linear regression
# build linear regression model on full data
summary(lsp_odata_md <- lm(odata_mean_lsp ~ lsp_like, data=df_lsp)) # sig
lm.beta(lsp_odata_md)

# mean rmsd
summary(lsp_rmsd_md <- lm(mean_rmsd_lsp_rt ~ lsp_like, data=df_lsp)) # sig
lm.beta(lsp_rmsd_md)
##### DANCE ##### 
vars_dnc<- c("dnc_like", "dnc_int", "odata_mean_dnc",  "mean_rmsd_dnc_rt")
df_dnc <- mydata[vars_dnc]
# mean overall data
df_dnc_ = melt(df_dnc, id.vars='odata_mean_dnc')
ggplot(df_dnc_) +
  geom_jitter(aes(value, odata_mean_dnc, colour=variable)) + geom_smooth(aes(value, odata_mean_dnc, colour=variable), 
                                                                         method=lm, se=FALSE) + 
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Dnc Liking", y = "Mean Overall Rating for Dance Videos")
# mean rmsd
df_dnc_ = melt(df_dnc, id.vars='mean_rmsd_dnc_rt')
ggplot(df_dnc_) +
  geom_jitter(aes(value, mean_rmsd_dnc_rt, colour=variable)) + geom_smooth(aes(value, mean_rmsd_dnc_rt, colour=variable), 
                                                                           method=lm, se=FALSE) + 
  facet_wrap(~variable, scales="free_x") +
  labs(x = "Dnc Interaction", y = "Mean rmsd Rating for dance videos")
# linear regression
# build linear regression model on full data
summary(dnc_odata_md <- lm(odata_mean_dnc ~ dnc_like, data=df_dnc)) # sig
lm.beta(dnc_odata_md)

# mean rmsd
summary(dnc_rmsd_md <- lm(mean_rmsd_dnc_rt ~ dnc_like, data=df_dnc)) #nonsig
lm.beta(dnc_rmsd_md)
