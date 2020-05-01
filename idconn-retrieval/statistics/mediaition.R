dat <- read.csv(file="/home/kbott006/physics-retrieval/mediation_edges.csv", header = T, sep="\t")
cols <- names(dat)
library(lavaan)
smallDat <- dat[,c("Mod", "F", "deltaPRI", "Age", "Strt_Level", "post_phys_fci_fd", "FCIPhyAcc2")]
smallDat$F = smallDat$F
smallDat$Mod = smallDat$Mod
smallDat$deltaPRIXClass = smallDat$deltaPRI * smallDat$Mod
smallDat$deltaPRIXSex = smallDat$deltaPRI * smallDat$F
smallDat$deltaPRIXClassXSex = smallDat$deltaPRI * smallDat$Mod * smallDat$F
smallDat$SexXClass = smallDat$Mod * smallDat$F
smallDat$fd = scale(smallDat$post_phys_fci_fd)
FIQMediationSyntax <- "
  #M ~ a*X regression (connectivity predicts IQ)
  edge ~ a1*deltaPRI + a2*deltaPRIXSex + a3*deltaPRIXClass + a4*deltaPRIXClassXSex + a5*Age + a6*Strt_Level + a8*SexXClass + a9*F + a10*Mod + a7*fd
  #Y ~ cPrime*X + b*M regression
  FCIPhyAcc2 ~ cPrime1*deltaPRI + cPrime2*deltaPRIXSex + cPrime3*deltaPRIXClass + cPrime4*deltaPRIXClassXSex + cPrime5*Age + cPrime6*Strt_Level + cPrime8*SexXClass + cPrime9*F + cPrime10*Mod + b*edge

  #Residual variances
  FCIPhyAcc2 ~~ dy*FCIPhyAcc2
  edge ~~ dm*edge
  deltaPRI ~~ deltaPRI

  #Covariances
  deltaPRI ~~ deltaPRIXSex
  deltaPRI ~~ deltaPRIXClass
  deltaPRI ~~ deltaPRIXClassXSex
  deltaPRI ~~ Age
  deltaPRI ~~ Mod
  deltaPRI ~~ F
  deltaPRI ~~ Strt_Level
  deltaPRIXSex ~~ Age
  deltaPRIXSex ~~ Mod
  deltaPRIXSex ~~ F
  deltaPRIXSex ~~ Strt_Level
  deltaPRIXClass ~~ Age
  deltaPRIXClass ~~ Mod
  deltaPRIXClass ~~ F
  deltaPRIXClass ~~ Strt_Level
  deltaPRIXClassXSex ~~ Age
  deltaPRIXClassXSex ~~ Mod
  deltaPRIXClassXSex ~~ F
  deltaPRIXClassXSex ~~ Strt_Level


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
for (edge in grep("*edge", colnames(dat))){
  
  smallDat["edge"] <- dat[,c(edge)]
  sink(sprintf("mediations/%s_mediation_output.txt", colnames(dat)[edge]))
  modFit = sem(model = FIQMediationSyntax, smallDat, missing = "fiml", estimator="mlr", fixed.x = F)
  modSum = summary(modFit, fit.measures = T, rsquare = T, standardized = T)
  modEst = parameterEstimates(modFit, standardized = T)
  sink()
}
