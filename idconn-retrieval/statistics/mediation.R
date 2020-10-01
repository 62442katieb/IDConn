dat <- read.csv(file='/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output/acc+conn~iq-mediation_edges-KNN.csv', header = T, sep="\t")
cols <- names(dat)
library(lavaan)
smallDat <- dat[,c("Mod", "F", "FSIQ2", "Age", "Strt_Level", "post_phys_fci_fd", "FCIPhyAcc2")]
smallDat$F = smallDat$F
smallDat$Mod = smallDat$Mod
smallDat$FSIQ2 = scale(smallDat$FSIQ2)
smallDat$FSIQ2XClass = smallDat$FSIQ2 * smallDat$Mod
smallDat$FSIQ2XSex = smallDat$FSIQ2 * smallDat$F
smallDat$FSIQ2XClassXSex = smallDat$FSIQ2 * smallDat$Mod * smallDat$F
smallDat$SexXClass = smallDat$Mod * smallDat$F
smallDat$fd = scale(smallDat$post_phys_fci_fd)
FIQMediationSyntax <- "
  #M ~ a*X regression (connectivity predicts IQ)
  edge ~ a1*FSIQ2 + a2*FSIQ2XSex + a3*FSIQ2XClass + a4*FSIQ2XClassXSex + a5*Age + a6*Strt_Level + a8*SexXClass + a9*F + a10*Mod + a7*fd
  #Y ~ cPrime*X + b*M regression
  FCIPhyAcc2 ~ cPrime1*FSIQ2 + cPrime2*FSIQ2XSex + cPrime3*FSIQ2XClass + cPrime4*FSIQ2XClassXSex + cPrime5*Age + cPrime6*Strt_Level + cPrime8*SexXClass + cPrime9*F + cPrime10*Mod + b*edge

  #Residual variances
  FCIPhyAcc2 ~~ dy*FCIPhyAcc2
  edge ~~ dm*edge
  FSIQ2 ~~ FSIQ2

  #Covariances
  


  #Intercepts
  FCIPhyAcc2 ~ b0Acc*1
  
  #simple slopes
  iqSSf := a1 + a2*2
  iqSSm := a1 + a2*1

  iqSSa := a1 + a3*2
  iqSSl := a1 + a3*1

  iqSSaf := a1 + a2*2 + a3*2
  iqSSlf := a1 + a2*2 + a3*1
  iqSSam := a1 + a2*1 + a3*2
  iqSSlm := a1 + a2*1 + a3*1

  #conditional indirects
  iqVconnf := iqSSf*b
  iqVconnm := iqSSm*b

  iqVconna := iqSSa*b
  iqVconnl := iqSSl*b

  iqVconnaf := iqSSaf*b
  iqVconnlf := iqSSlf*b
  iqVconnlm := iqSSlm*b
  iqVconnam := iqSSam*b

  #indices of moderated mediations
  modMedSex := a2*b
  modMedClass := a3*b
  modMedSexClass := a4*b

  #mediation
  edgeMedIQAcc := a1*b
"
for (edge in grep("craddock*", colnames(dat))){
  
  smallDat["edge"] <- dat[,c(edge)]
  sink(sprintf("/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output/mediations/%s_mediation_output.txt", colnames(dat)[edge]))
  modFit = sem(model = FIQMediationSyntax, smallDat, missing = "ml", estimator="mlr")
  modSum = summary(modFit, fit.measures = T, rsquare = T, standardized = T)
  modEst = parameterEstimates(modFit, standardized = T)
  sink()
}

for (edge in grep("shen*", colnames(dat))){
  
  smallDat["edge"] <- dat[,c(edge)]
  sink(sprintf("/Users/kbottenh/Dropbox/Projects/physics-retrieval/data/output/mediations/%s_mediation_output.txt", colnames(dat)[edge]))
  modFit = sem(model = FIQMediationSyntax, smallDat, missing = "ml", estimator="mlr")
  modSum = summary(modFit, fit.measures = T, rsquare = T, standardized = T)
  modEst = parameterEstimates(modFit, standardized = T)
  #print(modSum)
  #print(modESt)
  sink()
}