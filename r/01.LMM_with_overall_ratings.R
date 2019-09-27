# Linear Mixed Effects analysis with overall ratings
rm(list=ls()) # clear working directory
# set working directory
setwd ("/Users/ilkay.isik/aesthetic_dynamics/")
# load libararies
library(readr)
library(MASS)
library(lme4)
library(RePsychLing)
library(lmerTest)
library(xtable)
library(sjPlot)
library(lsmeans)

############### STEP 1: Load the data and organize the data frame ############### 
# load data 
overall_data <- read_delim("data/df_overall.csv", delim=",") 
# Change the name of session values test -> Test..
overall_data$session[overall_data$session == "test"] <- "Test"
overall_data$session[overall_data$session == "retest"] <- "Retest"
# Change the name of category values D -> Dance..
overall_data$category[overall_data$category == "D"] <- "Dance"
overall_data$category[overall_data$category == "L"] <- "Landscape"

# make some cols factors
cols <- c("category", "movName", "subject", "sub_code", "session", "group")
overall_data[cols] <- lapply(overall_data[cols], factor)
summary(overall_data)
str(overall_data)

#################### STEP 2: Data Transformation  ####################
# determine lambda for oData [https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/]
lambdaList <- boxcox(oData+2~category, data=overall_data)
(lambda <- lambdaList$x[which.max(lambdaList$y)]) # = no transformation necessary
# no transformation is necessary
################### STEP 3: LINEAR MIXED MODEL ######################
# Preps
# relevel the session column to Test, Retest
overall_data$session <- factor(overall_data$session , levels = c("Test", "Retest"))
levels(overall_data$session)
# overall_data$category <- factor(overall_data$category , levels = c("Landscape", "Dance"))
levels(overall_data$category)
levels(overall_data$group)

# Set up sum contrasts
# . w/ sum contrasts; intercept will be GM of conditions
(contrasts(overall_data$category) <- contr.sum(2))
(contrasts(overall_data$session) <- contr.sum(2))
(contrasts(overall_data$group) <- contr.sum(2))

# Set up different models, to compare
# simplest model
summary(M1 <- lmer(oData ~ 1 + session * category * group + 
                  (1|sub_code) + 
                  (1|movName), 
                   overall_data))

# category in the subjects random effects structure
summary(M2 <- lmer(oData ~ 1 + session * category * group + 
                     (1+category|sub_code) + 
                     (1|movName), 
                   overall_data))
coefficients(M2)
anova(M1, M2) # pick the 2nd model over 1st
summary(rePCA(M2)) # model is not degenerate

# category and session in the subjects random effects structure: FULL MODEL
summary(M3  <- lmer(oData ~ 1 + session * category * group + 
                           (1 + category+session|sub_code) + 
                           (1 | movName), 
                         overall_data))
coefficients(M3)
anova(M2, M3) # they are not different

summary(rePCA(M3)) # model is degenerate (3rd variable-session is not explaning any variance)
# go on with the second model

# looks like we go on with M2
# Check the residuals to see if tehy are normally distributed
qqnorm(residuals(M2))
hist(residuals(M2)) 

# Going on with M2
# to write out the results as html table
tab_model(M2, show.se = TRUE, show.stat = TRUE, show.icc=TRUE, show.obs = FALSE,
string.stat = "t", rm.terms = TRUE, digits = 3
, file="output/tables/Table1.odata_lmm_result_table.html"
)
# keep the coefs as a table
coefs<-xtable(coef(summary(M2)))
print.xtable(coefs, type="html", file="output/tables/odata_lmm_coefs.html")
# Breakdown the category main effect 
lsmeans(M2, pairwise~category, adjust="tukey")

tukey <- as.data.frame(summary(lsmeans(M2, pairwise~category, adjust="tukey"))$contrasts)
tukey$df <- NULL
tukey<-xtable(tukey,digits=c(3,3,3,3,3,4))
print.xtable(tukey, type="html", file="tukey_Category_MainEffect.html", include.rownames = FALSE)
