library(AER)
library(sandwich)

dat <- read.csv('/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/rescored/physics_learning-nonbrain_OLS+fd+eff-BayesianImpute.csv', header=TRUE)
dat$Class.Type[dat$Class.Type == ""] <- NA
dat$Sex[dat$Sex == "" ] <- NA
dat = subset(dat, select = -c(Mod,F,Strt.Level) )

dat$Class.Type = as.factor(dat$Class.Type)
dat$Sex = as.factor(dat$Sex)

dat$deltaPRI = dat$PRI2 - dat$PRI1
dat$deltaWMI = dat$WMI2 - dat$WMI1
dat$deltaVCI = dat$VCI2 - dat$VCI1
dat$deltaPSI = dat$PSI2 - dat$PSI1
dat$deltaFSIQ = dat$FSIQ2 - dat$FSIQ1

iqs <- c('FSIQ2', 'PRI2', 'VCI2', 'PSI2', 'WMI2', 'deltaFSIQ', 'deltaPRI', 'deltaVCI', 'deltaPSI', 'deltaWMI')
outcomes <- c('efficiency_1_fci_high.level_craddock2012',
              'efficiency_1_fci_high.level_shen2015',
              'efficiency_1_retr_high.level_craddock2012',
              'efficiency_1_retr_high.level_shen2015')

col.names <- c('F', 'p(F)', 'BIC', 'IQ', 'IQ*Sex', 'IQ*Class', 'IQ*Sex*Class', 
               'Sex', 'Class', 'Sex*Class', 'Age', 'StrtLvl',
               'p(IQ)', 'p(IQ*Sex)', 'p(IQ*Class)', 'p(IQ*Sex*Class)', 
               'p(Sex)', 'p(Class)', 'p(Sex*Class)', 'p(Age)', 'p(StrtLvl)', 'ANOVA vs. Model')

row.names <- c()
for (iq in iqs) {for (outcome in outcomes) {
        row.names <- c(row.names, paste(outcome, iq))
    }
}

df <- as.data.frame(matrix(ncol=length(col.names), nrow=length(row.names)), row.names = row.names)

for (i in c(1:length(col.names))) {
    names(df)[names(df) == names(df)[i]] <- col.names[i]
}

col.names <- c('F', 'p(F)', 'BIC', 'IQ', 
               'Sex', 'Class', 'Age', 'StrtLvl',
               'p(IQ)', 
               'p(Sex)', 'p(Class)', 'p(Age)', 'p(StrtLvl)')

df.simple <- as.data.frame(matrix(ncol=length(col.names), nrow=length(row.names)), row.names = row.names)

for (i in c(1:length(col.names))) {
    names(df.simple)[names(df.simple) == names(df.simple)[i]] <- col.names[i]
}


for (iq in iqs) {
    for (outcome in outcomes) {
        intrxnFormula <- paste(outcome, '~', iq, ' + ', iq, '*Sex + ', iq, '*Class.Type + ', iq, 
                               '*Sex*Class.Type + Sex + Sex*Class.Type + Class.Type + StrtLvl + Age')
        simpleFormula <- paste(outcome, '~', iq, ' + Sex + Sex*Class.Type + Class.Type + StrtLvl + Age')
        linearModI <- lm(intrxnFormula, data=dat)
        linearModS <- lm(simpleFormula, data=dat)
        sum <- summary(linearModI)
        comparison <- anova(linearModI, linearModS)

        vcv <- vcovHC(linearModI)
        coeffs <- coeftest(linearModI, vcv)

        f <- sum$fstatistic
        p <- pf(f[1], f[2], f[3], lower.tail=F)

        regression = paste(outcome, iq)
        
        df[regression,'F'] = f['value']
        df[regression,'p(F)'] = p
        df[regression,'BIC'] = BIC(linearModI)
        
        df[regression,'IQ'] = coeffs[iq, 'Estimate']
        df[regression,'IQ*Sex'] = coeffs[paste(iq, ':SexM', sep=''), 'Estimate']
        df[regression,'IQ*Class'] = coeffs[paste(iq, ':Class.TypeMod', sep=''), 'Estimate']
        df[regression,'IQ*Sex*Class'] = coeffs[paste(iq, ':SexM:Class.TypeMod', sep=''), 'Estimate']
        df[regression,'Sex'] = coeffs['SexM', 'Estimate']
        df[regression,'Class'] = coeffs['Class.TypeMod', 'Estimate']
        df[regression,'Sex*Class'] = coeffs['SexM:Class.TypeMod', 'Estimate']
        df[regression,'Age'] = coeffs['Age', 'Estimate']
        df[regression,'StrtLvl'] = coeffs['StrtLvl', 'Estimate']

        df[regression,'p(IQ)'] = coeffs[iq, 'Pr(>|t|)']
        df[regression,'p(IQ*Sex)'] = coeffs[paste(iq, ':SexM', sep=''), 'Pr(>|t|)']
        df[regression,'p(IQ*Class)'] = coeffs[paste(iq, ':Class.TypeMod', sep=''), 'Pr(>|t|)']
        df[regression,'p(IQ*Sex*Class)'] = coeffs[paste(iq, ':SexM:Class.TypeMod', sep=''), 'Pr(>|t|)']
        df[regression,'p(Sex*Class)'] = coeffs['SexM:Class.TypeMod', 'Pr(>|t|)']
        df[regression,'p(Sex)'] = coeffs['SexM', 'Pr(>|t|)']
        df[regression,'p(Class)'] = coeffs['Class.TypeMod', 'Pr(>|t|)']
        df[regression,'p(Age)'] = coeffs['Age', 'Pr(>|t|)']
        df[regression,'p(StrtLvl)'] = coeffs['StrtLvl', 'Pr(>|t|)']

        df[regression,'ANOVA vs. Simple'] = comparison[2,'Pr(>F)']
        write.csv(df, '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output/efficiency~iq-ols_robust-knn_imp-interaction-R.csv')

        sum <- summary(linearModS)
        vcv <- vcovHC(linearModS)
        coeffs <- coeftest(linearModS, vcv)

        f <- sum$fstatistic
        p <- pf(f[1], f[2], f[3], lower.tail=F)
        
        df.simple[regression,'F'] = f['value']
        df.simple[regression,'p(F)'] = p
        df.simple[regression,'BIC'] = BIC(linearModI)
        
        df.simple[regression,'IQ'] = coeffs[iq, 'Estimate']
        df.simple[regression,'Sex'] = coeffs['SexM', 'Estimate']
        df.simple[regression,'Class'] = coeffs['Class.TypeMod', 'Estimate']
        df.simple[regression,'Age'] = coeffs['Age', 'Estimate']
        df.simple[regression,'StrtLvl'] = coeffs['StrtLvl', 'Estimate']

        df.simple[regression,'p(IQ)'] = coeffs[iq, 'Pr(>|t|)']
        df.simple[regression,'p(Sex)'] = coeffs['SexM', 'Pr(>|t|)']
        df.simple[regression,'p(Class)'] = coeffs['Class.TypeMod', 'Pr(>|t|)']
        df.simple[regression,'p(Age)'] = coeffs['Age', 'Pr(>|t|)']
        df.simple[regression,'p(StrtLvl)'] = coeffs['StrtLvl', 'Pr(>|t|)']

        write.csv(df.simple, '/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output/efficiency~iq-ols_robust-knn_imp-simple-R.csv')

    }
}

