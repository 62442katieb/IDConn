library(psych)
library(rstatix)
library(lme4)

sink("/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output/anovas/iq-timeXsexXclass-anova_output.txt")

dat <- read.csv(file='/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/rescored/all_iq_sex_class-long.csv', stringsAsFactors=TRUE)

dat$Subject <- factor(dat$Subject)
dat$Time <- factor(dat$Time)

VCIdat <- dat[ which(dat$Measure=='VCI'),]
PRIdat <- dat[ which(dat$Measure=='PRI'),]
PSIdat <- dat[ which(dat$Measure=='PSI'),]
WMIdat <- dat[ which(dat$Measure=='WMI'),]
FSIQdat <- dat[ which(dat$Measure=='FSIQ'),]

print('************************** FSIQ ************************')
print('Shapiro test for notmality:')
FSIQdat %>% group_by(Class.Type, Sex, Time) %>% shapiro_test(IQ)
print('Outlier check:')
FSIQdat %>% group_by(Class.Type, Sex, Time) %>% identify_outliers(IQ)
print('Levene test for homogeneity of variance:')
FSIQdat %>% group_by(Time) %>% levene_test(IQ ~ Sex*Class.Type)

mixed_aov <- aov(IQ ~ Sex*Class.Type*Time + Error(Subject/(Time)) + Sex*Class.Type, data=FSIQdat)
summary(mixed_aov)

print('************************** PRI *************************')
print('Shapiro test for notmality:')
PRIdat %>% group_by(Class.Type, Sex, Time) %>% shapiro_test(IQ)
print('Outlier check:')
PRIdat %>% group_by(Class.Type, Sex, Time) %>% identify_outliers(IQ)
print('Levene test for homogeneity of variance:')
PRIdat %>% group_by(Time) %>% levene_test(IQ ~ Sex*Class.Type)

mixed_aov <- aov(IQ ~ Sex*Class.Type*Time + Error(Subject/(Time)) + Sex*Class.Type, data= PRIdat)
summary(mixed_aov)

print('************************** PSI *************************')
print('Shapiro test for notmality:')
PSIdat %>% group_by(Class.Type, Sex, Time) %>% shapiro_test(IQ)
print('Outlier check:')
PSIdat %>% group_by(Class.Type, Sex, Time) %>% identify_outliers(IQ)
print('Levene test for homogeneity of variance:')
PSIdat %>% group_by(Time) %>% levene_test(IQ ~ Sex*Class.Type)

mixed_aov <- aov(IQ ~ Sex*Class.Type*Time + Error(Subject/(Time)) + Sex*Class.Type, data= PSIdat)
summary(mixed_aov)

print('************************** VCI *************************')
print('Shapiro test for notmality:')
VCIdat %>% group_by(Class.Type, Sex, Time) %>% shapiro_test(IQ)
print('Outlier check:')
VCIdat %>% group_by(Class.Type, Sex, Time) %>% identify_outliers(IQ)
print('Levene test for homogeneity of variance:')
VCIdat %>% group_by(Time) %>% levene_test(IQ ~ Sex*Class.Type)

mixed_aov <- aov(IQ ~ Sex*Class.Type*Time + Error(Subject/(Time)) + Sex*Class.Type, data= VCIdat)
summary(mixed_aov)

print('************************** WMI *************************')
print('Shapiro test for notmality:')
WMIdat %>% group_by(Class.Type, Sex, Time) %>% shapiro_test(IQ)
print('Outlier check:')
WMIdat %>% group_by(Class.Type, Sex, Time) %>% identify_outliers(IQ)
print('Levene test for homogeneity of variance:')
WMIdat %>% group_by(Time) %>% levene_test(IQ ~ Sex*Class.Type)

mixed_aov <- aov(IQ ~ Sex*Class.Type*Time + Error(Subject/(Time)) + Sex*Class.Type, data= WMIdat)
summary(mixed_aov)

sink()