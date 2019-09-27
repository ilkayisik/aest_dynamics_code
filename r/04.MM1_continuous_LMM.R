rm(list=ls()) # clear working directory
# set working directory
setwd ("/Users/ilkay.isik/Desktop/aesthetic_dynamics/")
library(readr)
library(ggplot2)
library(MASS)
library(lme4)
library(lmerTest)
library(lsmeans)
library(remef)
library(RePsychLing)
library(sjPlot)

# load data
cont_mm1 <- read_delim("data/mm1_continuous.csv", delim=",",
                     skip=1,
                     col_names=c("mm1_corr", "ztrans_mm1", "movName", "category",
                                 "session", "sub", "group"))

# Change the name of session values test -> Test..
cont_mm1$session[cont_mm1$session == "test"] <- "Test"
cont_mm1$session[cont_mm1$session == "retest"] <- "Retest"
# make some cols factors
cols <- c("movName", "category", "sub", "group", "session")
cont_mm1[cols] <- lapply(cont_mm1[cols], factor)

# Check the distributions
rate = cont_mm1[cont_mm1$group == 'Rate',]
view = cont_mm1[cont_mm1$group == 'View',]

# Histograms with rate and view values: ztransformed
rateplot <- ggplot(rate, aes(ztrans_mm1)) + geom_histogram(binwidth = 0.5) +
  facet_wrap(category~session)
print(rateplot)
viewplot <- ggplot(view, aes(ztrans_mm1)) + geom_histogram(binwidth = 0.5) +
  facet_wrap(category~session)
print(viewplot)

############################ LMM with cont MM1 values ############################
# relevel the session column to Test, Retest
cont_mm1$session <- factor(cont_mm1$session , levels = c("Test", "Retest"))
levels(cont_mm1$session)
levels(cont_mm1$category)

# determine lambda for oData
lambdaList <- boxcox(ztrans_mm1 + 2 ~category, data=cont_mm1)
(lambda <- lambdaList$x[which.max(lambdaList$y)]) # = no transformation necessary
# https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/

# . w/ sum contrasts; intercept will be GM of conditions
(contrasts(cont_mm1$category) <- contr.sum(2))
(contrasts(cont_mm1$session) <- contr.sum(2))
(contrasts(cont_mm1$group) <- contr.sum(2))

# category in the subjects random effects structure (overall's M2)
summary(M1<- lmer(ztrans_mm1 ~ session * category * group +
                    (1|sub) +
                    (1|movName),
                  cont_mm1))

# category in the subjects random effects structure (overall's M2)
summary(M2<- lmer(ztrans_mm1 ~ session * category * group +
                     (1+category|sub) +
                     (1|movName),
                   cont_mm1))

summary(M3<- lmer(ztrans_mm1 ~ session * category * group +
                    (1+category+session|sub) +
                    (1|movName),
                  cont_mm1))

anova(M0, M1, M3) # M2 is a better model
# main effects of:
# Session (higher agreement in the Test session)
# Category (lscp agreement is higher than dance agreement)

################## S2 Table ###############
# keep the coefs as a table
tab_model(M2, show.se = TRUE, show.stat = TRUE, show.obs = FALSE,
          string.stat = "t", digits = 3
          ,file="output/tables/S2.Table_Continous_MM1_lmm_result_table_tabmodel.html"
)

# coefs<-xtable(coef(summary(M2)))
# print.xtable(coefs, type="html", file="output/ContData_MM1_LMM_coefs.html")



############ NOT USED IN THE MANUSCRIPT ############
# Create a combined plot
# aggregate data to get the mean rating
agg_all <- aggregate(ztrans_mm1 ~ sub + category + group + session, mean,
                     data=cont_mm1)

# pdf('ContMM1_Submeans_R_BoxPlot_Combined.pdf')

# create box plot
x <- qplot(data=agg_all, x=session, y=ztrans_mm1, fill=category) +
  geom_jitter(width = 0.1) +
  geom_boxplot(outlier.shape = 1, notch=FALSE, width=0.5) +
  ggtitle("MM1 subject means")  + theme_bw() +
  ylab("MM1 Cont") + xlab("Session") + guides(fill=FALSE) +
  facet_grid(group~category) +
  scale_fill_manual(values = c("deepskyblue","firebrick2"))+
  theme_set(theme_gray(base_size = 24)) +
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),
        legend.title = element_blank())
print(x)
# dev.off()

# LMM ONLY FOR RATE
# : w/ sum contrasts; intercept will be the GM of conditions
rate$session <- factor(rate$session , levels = c("Test", "Retest"))
levels(rate$session)
(contrasts(rate$category) <- contr.sum(2))
(contrasts(rate$session) <- contr.sum(2))

# category in the subjects random effects structure (overall's M2)
summary(M2 <- lmer(ztrans_mm1 ~ session * category +
                    (1+category|sub) +
                    (1|movName),
                  rate))

# keep the coefs as a table
coefs<-xtable(coef(summary(M2)))
print.xtable(coefs, type="html", file="output/ContData_MM1_LMM_coefs_Rate.html")


agg_rate <- aggregate(ztrans_mm1 ~ sub + category + session, mean,
                     data=rate)

x <- qplot(data=agg_rate, x=session, y=ztrans_mm1, fill=category) +
  geom_jitter(width = 0.1) +
  geom_boxplot(outlier.shape = 1, notch=TRUE, width=0.5) +
  ggtitle("MM1 subject means - Rate")  + theme_bw() +
  ylab("MM1 Cont") + xlab("Session") + guides(fill=FALSE) +
  facet_grid(category) +
  scale_fill_manual(values = c("deepskyblue","firebrick2"))+
  theme_set(theme_gray(base_size = 24)) +
  theme(plot.title = element_text(hjust = 0.5),
        panel.background = element_blank(),
        legend.title = element_blank())
print(x)

