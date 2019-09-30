rm(list=ls()) # clear working directory
# set working directory
setwd ("/Users/ilkay.isik/Desktop/aesthetic_dynamics/")
# setwd("C:/Users/ilkay/MPI-Documents/fcContent/BehavioralExperiments/BP03-BP04/")
library(readr)
library(ggplot2)
library(MASS)
library(lme4)
library(lmerTest)
library(RePsychLing)
library(xtable)
library(sjPlot)
library(lsmeans)
library(ez)
library(lsr)
library(apaTables)
library(remef)

########################## STEP 1: Load data and organize the df ##########################
# load data 
ov_mm1 <- read_delim("data/mm1_overall.csv", delim=",",
                           skip=1,
                           col_names=c("subCode","mm1_corr", "ztrans_mm1",
                                       "category",  "session", "sub", "group"))
# Change the name of session values test -> Test..
ov_mm1$session[ov_mm1$session == "test"] <- "Test"
ov_mm1$session[ov_mm1$session == "retest"] <- "Retest"
# make some cols factors
cols <- c("subCode", "category", "sub", "group", "session")
ov_mm1[cols] <- lapply(ov_mm1[cols], factor)
# relevel the session column to Test, Retest
ov_mm1$session <- factor(ov_mm1$session , levels = c("Test", "Retest"))
levels(ov_mm1$session)

rate = ov_mm1[ov_mm1$group == 'Rate',]
view = ov_mm1[ov_mm1$group == 'View',]

summary(ov_mm1)
str(ov_mm1)

# Histograms with rate and view values: ztransformed
rateplot <- ggplot(rate, aes(ztrans_mm1)) + geom_histogram(binwidth = 0.5) +
  facet_wrap(category~session)
print(rateplot)
viewplot <- ggplot(view, aes(ztrans_mm1)) + geom_histogram(binwidth = 0.5) +
  facet_wrap(category~session)
print(viewplot)

########################## LMM ############################
levels(ov_mm1$session)
levels(ov_mm1$category)

# determine lambda for oData
lambdaList <- boxcox(ztrans_mm1 + 2 ~category, data=ov_mm1)
(lambda <- lambdaList$x[which.max(lambdaList$y)]) # = no transformation necessary
# https://www.statisticshowto.datasciencecentral.com/box-cox-transformation/

# . w/ sum contrasts; intercept will be GM of conditions
(contrasts(ov_mm1$category) <- contr.sum(2))
(contrasts(ov_mm1$session) <- contr.sum(2))
(contrasts(ov_mm1$group) <- contr.sum(2))
# category in the subjects random effects structure (overall's M2)
summary(M1<- lmer(ztrans_mm1 ~ session * category * group + 
                    (1+category|sub),
                    ov_mm1))

# model is not degenerate
summary(rePCA(M1)) # model is not degenerate

summary(M2<- lmer(ztrans_mm1 ~ session * category * group + 
                 (1+category+session|sub),
                  ov_mm1))

# model is not degenerate
summary(rePCA(M2)) # model is not degenerate
anova(M1, M2)

# M2 is a better model pick that
# export coef table as html
# coefs<-xtable(coef(summary(M2)),digits=c(3,3,3,3,3,3))
# print.xtable(coefs, type="html", file="output/LMM_MM1Overall_Coefs_M2.html")
########################  S1 table ########################  
tab_model(M2, show.se = TRUE, show.stat = TRUE, show.obs = FALSE,
          string.stat = "t", digits = 2
          ,file="output/tables/S1.Table_LMM_MM1Overall_Coefs_M2_tabmodel.html"
)
# break down significant interaction
lsmeans(M2, pairwise~session*group, adjust="tukey")  # test rate vs retest rate: different

# export tukey results as html
tukey <- as.data.frame(summary(lsmeans(M2, pairwise~session*group, adjust="tukey"))$contrasts)
tukey$df <- NULL
tukey<-xtable(tukey,digits=c(2,2,2,2,2,3))
print.xtable(tukey, type="html", file="output/tables/S1.Table_Interaction_LMM_MM1Overall_TukeyComparisons_M2.html", include.rownames = FALSE)









##########################  OTHER EXPLORATIONS ##########################  
# check the descriptive stats of the data
ezStats(data=ov_mm1,
        dv = .(ztrans_mm1), 
        wid = .(sub), 
        within=.(category, session), 
        between=.(group),
        type=3)

# Run ANOVA on ztransformed r values: ALL
mixed_aov <- ezANOVA(data=ov_mm1,
        dv = .(ztrans_mm1), 
        wid = .(sub), 
        within=.(category, session), 
        between=.(group),
        type=3
        #detailed = 1,
        #return_aov = 1
        )
# export coef table as html
effects <- ezANOVA(data=ov_mm1,
                   dv = .(ztrans_mm1), 
                   wid = .(sub), 
                   within=.(category, session), 
                   between=.(group))$ANOVA
effects <- xtable(effects, digits=c(2,2,2,2,2,2,2,2))
# print.xtable(effects, type="html", file="output/mm1_overallrating_ANOVA.html", include.rownames = FALSE)


# Print out an APA style ANOVA table
apa.ezANOVA.table(mixed_aov, correction = "GG", 
table.title = "Table_Mixed ANOVA with Overall rating MM1 values",
filename='output/Table_Mixed ANOVA with Overall rating MM1 values.rtf', table.number = 1)

# Plot the interaction with ezPlot
group_by_session = ezPlot(
  data = ov_mm1,
  dv = .(ztrans_mm1),
  wid = .(sub),
  within = .(session, category),
  between = .(group),
  x = .(category),
  split = .(session),
  col = .(group),
  x_lab = 'Category',
  y_lab = 'MM1',
  split_lab = 'Session') + theme_bw() + theme(legend.position = c(0.15, 0.8)) +
  theme(legend.title=element_blank()) + ylab("z-transformed MM1 scores")

fname = "output/MixedANOVAwithOverallRatingsMM1_Results.tiff"
tiff(fname, units="in", width=3.5, height=3, res=300)
# print(group_by_session)
dev.off()

# group by session interaction ?
# Posthoc comparisons
# agreegate by session 
rate_ses <- aggregate(ztrans_mm1~sub+session,mean, data=rate)
view_ses <- aggregate(ztrans_mm1~sub+session,mean, data=view)

t_test_rate_session <- t.test(rate_ses$ztrans_mm1[rate_ses$session=='Test'], 
                              rate_ses$ztrans_mm1[rate_ses$session=='Retest'], 
                              paired=TRUE, 
                              conf.level=0.95)

t_test_rate_session <- t.test(view_ses$ztrans_mm1[view_ses$session=='Test'], 
                              view_ses$ztrans_mm1[view_ses$session=='Retest'], 
                              paired=TRUE, 
                              conf.level=0.95)

dnc = rate[rate$category == "Dance",]
lscp = rate[rate$category == "Landscape",]

t_test_rate_dnc <- t.test(dnc$ztrans_mm1[dnc$session=='Test'], 
                          dnc$ztrans_mm1[dnc$session=='Retest'], 
                          paired=TRUE, 
                          conf.level=0.95)

# Run ANOVA on ztransformed r values: RATE
ezANOVA(data=rate,
        dv = .(ztrans_mm1), 
        wid = .(sub), 
        within=.(category, session))
effects <- ezANOVA(data=rate,
                   dv = .(ztrans_mm1), 
                   wid = .(sub), 
                   within=.(category, session))$ANOVA
effects <- xtable(effects,  digits=c(2,2,2,2,2,2,2,3))

# print.xtable(effects, type="html", file="output/mm1_overallrating_ANOVA_Rate.html", include.rownames = FALSE)

# Run ANOVA on ztransformed r values: VIEW
ezANOVA(data=view,
        dv = .(ztrans_mm1), 
        wid = .(subCode), 
        within=.(category, session))
effects <- ezANOVA(data=view,
                   dv = .(ztrans_mm1), 
                   wid = .(subCode), 
                   within=.(category, session))$ANOVA
effects <- xtable(effects,  digits=c(2,2,2,2,2,2,2,3))
#print.xtable(effects, type="html", file="mm1_overallrating_ANOVA_View.html", include.rownames = FALSE)
agg <- aggregate(mm1_corr~category+group+session,mean, data=ov_mm1)
