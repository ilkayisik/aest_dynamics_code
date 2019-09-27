rm(list=ls()) # clear working directory
# set working directory
setwd ("/Users/ilkay.isik/Desktop/aesthetic_dynamics/")
library(readr)
library(MASS)
library(lme4)
library(lmerTest)
library(RePsychLing)
library(sjPlot)
library(MuMIn) # for the rsquared
library(lsmeans)
library(xtable)
library(ggplot2)
library(lattice)

############### STEP 1: Load the data and organize the data frame ###############
df_rmsd <- read_delim("data/df_rmsd_values.csv", delim=",")
# Change the name of session values "D" -> "Dance"
df_rmsd$Category[df_rmsd$Category == "D"] <- "Dance"
df_rmsd$Category[df_rmsd$Category == "L"] <- "Landscape"
# make some cols factors
cols <- c("Movie", "Category", "Subject", "Group", "Session")
df_rmsd[cols] <- lapply(df_rmsd[cols], factor)
summary(df_rmsd)
str(df_rmsd)

################### Check the distributions ####################
hist(df_rmsd$rmsd) 
qqnorm(df_rmsd$rmsd)
shapiro.test(df_rmsd$rmsd)  # It is not normal, very much skewed and requires a trans.

#################### STEP 2: Data Transformation  ####################
# Do you need data transformation?
# cannot deal with negative values so shifting the data
lambdaList1 <- boxcox(rmsd + 0.00001 ~ Category, data=df_rmsd) 
# http://www.statisticshowto.com/box-cox-transformation/
(lambda <- lambdaList1$x[which.max(lambdaList1$y)]) # log is advised

df_rmsd$log_rmsd <- log(df_rmsd$rmsd)
hist(df_rmsd$log_rmsd, 40)
# change infs to nan
df_rmsd$log_rmsd[which(!is.finite(df_rmsd$log_rmsd))] <- NaN

# Do LMM first for the rate group if no session effect then do a retest LMM and report it
############### STEP 2: Rate LMM  ###############
rate = df_rmsd[df_rmsd$Group == 'Rate',]
rate['Group'] <- NULL
cols <- c("Movie", "Category", "Subject", "Session")
rate[cols] <- lapply(rate[cols], factor)
summary(rate)
str(rate)
# Set sum contrasts
(contrasts(rate$Category) <- contr.sum(2))
(contrasts(rate$Session) <- contr.sum(2))
# check the levels set correctly
levels(rate$Session)
levels(rate$Category)

# category in the subjects random effects structure (overall's M2)
summary(rate.M1<- lmer(log_rmsd ~ Session * Category +
                    (1|Subject) +
                    (1|Movie),
                    rate))
# p = n of fixed effects; q = n of random effects; nth = n of variance component parameters
getME(rate.M1, name="devcomp") 

summary(rate.M2 <- lmer(log_rmsd ~ Session * Category +
                    (1+Category|Subject) +
                    (1|Movie),
                  rate))
anova(rate.M1, rate.M2) # 2nd model is better
summary(rePCA(rate.M2)) # not degenerate

summary(rate.M3 <- lmer(log_rmsd ~ Session * Category +
                          (1+Category+Session|Subject) +
                          (1|Movie), na.action = na.omit,
                        rate))

anova(rate.M2, rate.M3) # third model is better
summary(rePCA(rate.M3)) # model is not degenerate

qqnorm(residuals(rate.M3))
hist(residuals(rate.M3)) 
# pick 3rd
tab_model(rate.M3)

###### No session effect found in RATE LMM ########
#### USE THE FOLLOWING LMM IN THE MANUSCRIPT ######
#############ONLY WITH RETEST SESSIONS#############
retest = df_rmsd[df_rmsd$Session == 'Retest',]
cols <- c("Movie", "Category", "Subject", "Session", "Group")
retest[cols] <- lapply(retest[cols], factor)
retest["Session"] <- NULL
retest$Group <- factor(retest$Group , levels = c("Rate", "View"))
levels(retest$Group)
retest$Category <- factor(retest$Category ,levels = c( "Dance", "Landscape"))
levels(retest$Category)
str(retest)
summary(retest)
(contrasts(retest$Category) <- contr.sum(2))
(contrasts(retest$Group) <- contr.sum(2))

summary(retest.M1<- lmer(log_rmsd ~ Category * Group +
                         (1|Subject) +
                         (1|Movie),
                       retest))

summary(retest.M2<- lmer(log_rmsd ~ 1 + Category*Group +
                         (1+Category|Subject) +
                         (1|Movie),
                       retest))
anova(retest.M1, retest.M2) # 2nd model is better
summary(rePCA(retest.M2)) # model is not degenerate
# Check if the residuals are normally distributed
qqnorm(residuals(retest.M2))
hist(residuals(retest.M2)) 
tab_model(retest.M2, show.se = TRUE, show.stat = TRUE, show.icc=TRUE, show.obs = FALSE,
          string.stat = "t", rm.terms = TRUE, digits = 3
          ,file="output/tables/Table2.rmsd_retest_lmm_result_table_withtabmodel.html"
          )
tukey <- as.data.frame(summary(lsmeans(retest.M2, pairwise~Group*Category, adjust="tukey"))$contrasts)
tukey$df <- NULL
tukey<-xtable(tukey,digits=c(3,3,3,3,3,4))
print.xtable(tukey, type="html", file="output/tables/rmsd_retest_lmm_tukey_table.html", include.rownames = FALSE)
difflsmeans(retest.M2, test.effs="Category:Group")





# trying to make sense of this interaction
# interaction plot
library(ez)
retest$log_rmsd[is.na(retest$log_rmsd)] <- 0
ezANOVA(data=retest, 
        dv = .(log_rmsd),
        wid = .(Subject), 
        within=.(Category)
)
# Plot the interaction with ezPlot
int_plot = ezPlot(
  data = retest,
  dv = .(log_rmsd),
  wid = .(Subject),
  within = .(Category),
  between = .(Group),
  x = .(Category),
  split = .(Group),
  # col = .(version),
  x_lab = 'Category',
  y_lab = 'rmsd',
  split_lab = 'Group') + theme_bw() + theme(legend.position = c(0.15, 0.8)) +
  theme(legend.title=element_blank()) + ylab("log-transformed rmsd scores")

fname = "output/figures/rmsd_retest_interaction_plot_results.tiff"
tiff(fname, units="in", width=3.5, height=3, res=300)
print(int_plot)
dev.off()

######################## RANEF ########################
# Plot for the random effects
retest.M2.ranef <- ranef(retest.M2, condVar = TRUE)
names(retest.M2.ranef[[1]])[1:2] <- c("Subject", "Category1")
names(retest.M2.ranef[[2]]) <- c("Movie")
# plot(retest.M2.ranef, aspect = 1, type = c("g", "p"))[[1]]   # plot.mer
### 
# .conditional modes
trellis.device(color = FALSE) # set to black and white
dotplot(retest.M2.ranef, layout=c(3, 1), scales = list(x = list(relation = 'free', rot=0), 
                                                       y=list()), strip = TRUE)[[1]][c(1,2)]



trellis.device(color = FALSE) # set to black and white
dotplot(retest.M2.ranef, layout=c(3, 1), scales = list(x = list(relation = 'same', rot=0), 
                                                      y=list()), strip = TRUE)[[2]][c(1)]
