retr_data <- read.csv(file="/Users/Katie/Dropbox/Projects/physics-retrieval/data/iq+brain+demo.csv", header = T)

maleRetrData <- read.csv(file="/Users/Katie/Dropbox/Projects/physics-retrieval/data/male_df.csv", header = T)
femaleRetrData <- read.csv(file="/Users/Katie/Dropbox/Projects/physics-retrieval/data/female_df.csv", header = T)

library(lavaan)

#first we test the female LCEN-RCEN-gen connection & grade mediation by IQs
FLcenRcenFullCov <- cov(femaleRetrData[,c("Phy48Grade", "fc.left.central.executive.right.central.executive.gen", "Full.Scale.IQ_2")])
FLcenRcenVerbCov <- cov(femaleRetrData[,c("Phy48Grade", "fc.left.central.executive.right.central.executive.gen", "Verbal.Comprehension.Sum_2")])
FLcenRcenPercCov <- cov(femaleRetrData[,c("Phy48Grade", "fc.left.central.executive.right.central.executive.gen", "Perceptual.Reasoning.Sum_2")])

n <- nrow(femaleRetrData)

#syntax for the model wherein IQ mediates the relationship between connectivity and grade
FIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Full.Scale.IQ_2 ~ a*fc.left.central.executive.right.central.executive.gen
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.left.central.executive.right.central.executive.gen + b*Full.Scale.IQ_2

	#Residual variances
	Full.Scale.IQ_2 ~~ Full.Scale.IQ_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.left.central.executive.right.central.executive.gen ~~ fc.left.central.executive.right.central.executive.gen

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

PIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Perceptual.Reasoning.Sum_2 ~ a*fc.left.central.executive.right.central.executive.gen
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.left.central.executive.right.central.executive.gen + b*Perceptual.Reasoning.Sum_2

	#Residual variances
	Perceptual.Reasoning.Sum_2 ~~ Perceptual.Reasoning.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.left.central.executive.right.central.executive.gen ~~ fc.left.central.executive.right.central.executive.gen

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

VIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Verbal.Comprehension.Sum_2 ~ a*fc.left.central.executive.right.central.executive.gen
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.left.central.executive.right.central.executive.gen + b*Verbal.Comprehension.Sum_2

	#Residual variances
	Verbal.Comprehension.Sum_2 ~~ Verbal.Comprehension.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.left.central.executive.right.central.executive.gen ~~ fc.left.central.executive.right.central.executive.gen

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

FIQMediationFit <- lavaan(model = FIQMediationSyntax, sample.cov = FLcenRcenFullCov, sample.nobs = n)
summary(FIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(FIQMediationFit, standardized = T)

VIQMediationFit <- lavaan(model = VIQMediationSyntax, sample.cov = FLcenRcenVerbCov, sample.nobs = n)
summary(VIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(VIQMediationFit, standardized = T)

PIQMediationFit <- lavaan(model = PIQMediationSyntax, sample.cov = FLcenRcenPercCov, sample.nobs = n)
summary(PIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(PIQMediationFit, standardized = T)

#then we do some math for male conn-grade med by iq
MHipLcenFullCov <- cov(maleRetrData[,c("Phy48Grade", "fc.hippo.left.central.executive.phy", "Full.Scale.IQ_2")])
MHipLcenVerbCov <- cov(maleRetrData[,c("Phy48Grade", "fc.hippo.left.central.executive.phy", "Verbal.Comprehension.Sum_2")])
MHipLcenPercCov <- cov(maleRetrData[,c("Phy48Grade", "fc.hippo.left.central.executive.phy", "Perceptual.Reasoning.Sum_2")])

n <- nrow(maleRetrData)

FIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Full.Scale.IQ_2 ~ a*fc.hippo.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.left.central.executive.phy + b*Full.Scale.IQ_2

	#Residual variances
	Full.Scale.IQ_2 ~~ Full.Scale.IQ_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.left.central.executive.phy ~~ fc.hippo.left.central.executive.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

PIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Perceptual.Reasoning.Sum_2 ~ a*fc.hippo.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.left.central.executive.phy + b*Perceptual.Reasoning.Sum_2

	#Residual variances
	Perceptual.Reasoning.Sum_2 ~~ Perceptual.Reasoning.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.left.central.executive.phy ~~ fc.hippo.left.central.executive.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

VIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Verbal.Comprehension.Sum_2 ~ a*fc.hippo.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.left.central.executive.phy + b*Verbal.Comprehension.Sum_2

	#Residual variances
	Verbal.Comprehension.Sum_2 ~~ Verbal.Comprehension.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.left.central.executive.phy ~~ fc.hippo.left.central.executive.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

FIQMediationFit <- lavaan(model = FIQMediationSyntax, sample.cov = MHipLcenFullCov, sample.nobs = n)
summary(FIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(FIQMediationFit, standardized = T)

VIQMediationFit <- lavaan(model = VIQMediationSyntax, sample.cov = MHipLcenVerbCov, sample.nobs = n)
summary(VIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(VIQMediationFit, standardized = T)

PIQMediationFit <- lavaan(model = PIQMediationSyntax, sample.cov = MHipLcenPercCov, sample.nobs = n)
summary(PIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(PIQMediationFit, standardized = T)

#then we do some math for male conn-grade med by iq
MGlEffFullCov <- cov(maleRetrData[,c("Phy48Grade", "global.efficiency.phy", "Full.Scale.IQ_2")])
MGlEffVerbCov <- cov(maleRetrData[,c("Phy48Grade", "global.efficiency.phy", "Verbal.Comprehension.Sum_2")])
MGlEffPercCov <- cov(maleRetrData[,c("Phy48Grade", "global.efficiency.phy", "Perceptual.Reasoning.Sum_2")])

n <- nrow(maleRetrData)

FIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Full.Scale.IQ_2 ~ a*global.efficiency.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*global.efficiency.phy + b*Full.Scale.IQ_2

	#Residual variances
	Full.Scale.IQ_2 ~~ Full.Scale.IQ_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  global.efficiency.phy ~~ global.efficiency.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

PIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Perceptual.Reasoning.Sum_2 ~ a*global.efficiency.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*global.efficiency.phy + b*Perceptual.Reasoning.Sum_2

	#Residual variances
	Perceptual.Reasoning.Sum_2 ~~ Perceptual.Reasoning.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  global.efficiency.phy ~~ global.efficiency.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

VIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Verbal.Comprehension.Sum_2 ~ a*global.efficiency.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*global.efficiency.phy + b*Verbal.Comprehension.Sum_2

	#Residual variances
	Verbal.Comprehension.Sum_2 ~~ Verbal.Comprehension.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  global.efficiency.phy ~~ global.efficiency.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

FIQMediationFit <- lavaan(model = FIQMediationSyntax, sample.cov = MGlEffFullCov, sample.nobs = n)
summary(FIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(FIQMediationFit, standardized = T)

VIQMediationFit <- lavaan(model = VIQMediationSyntax, sample.cov = MGlEffVerbCov, sample.nobs = n)
summary(VIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(VIQMediationFit, standardized = T)

PIQMediationFit <- lavaan(model = PIQMediationSyntax, sample.cov = MGlEffPercCov, sample.nobs = n)
summary(PIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(PIQMediationFit, standardized = T)

MLEffLCENFullCov <- cov(maleRetrData[,c("Phy48Grade", "le.left.central.executive.phy", "Full.Scale.IQ_2")])
MLEffLCENVerbCov <- cov(maleRetrData[,c("Phy48Grade", "le.left.central.executive.phy", "Verbal.Comprehension.Sum_2")])
MLEffLCENPercCov <- cov(maleRetrData[,c("Phy48Grade", "le.left.central.executive.phy", "Perceptual.Reasoning.Sum_2")])

n <- nrow(maleRetrData)

FIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Full.Scale.IQ_2 ~ a*le.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*le.left.central.executive.phy + b*Full.Scale.IQ_2

	#Residual variances
	Full.Scale.IQ_2 ~~ Full.Scale.IQ_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  le.left.central.executive.phy ~~ le.left.central.executive.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

PIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Perceptual.Reasoning.Sum_2 ~ a*le.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*le.left.central.executive.phy + b*Perceptual.Reasoning.Sum_2

	#Residual variances
	Perceptual.Reasoning.Sum_2 ~~ Perceptual.Reasoning.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  le.left.central.executive.phy ~~ le.left.central.executive.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

VIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Verbal.Comprehension.Sum_2 ~ a*le.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*le.left.central.executive.phy + b*Verbal.Comprehension.Sum_2

	#Residual variances
	Verbal.Comprehension.Sum_2 ~~ Verbal.Comprehension.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  le.left.central.executive.phy ~~ le.left.central.executive.phy

    ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
    c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

FIQMediationFit <- lavaan(model = FIQMediationSyntax, sample.cov = MLEffLCENFullCov, sample.nobs = n)
summary(FIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(FIQMediationFit, standardized = T)

VIQMediationFit <- lavaan(model = VIQMediationSyntax, sample.cov = MLEffLCENVerbCov, sample.nobs = n)
summary(VIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(VIQMediationFit, standardized = T)

PIQMediationFit <- lavaan(model = PIQMediationSyntax, sample.cov = MLEffLCENPercCov, sample.nobs = n)
summary(PIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(PIQMediationFit, standardized = T)

#now we collapse across sexes and test whole-sample whole-sample mediations
HipDMNFullCov <- cov(retr_data[,c("Phy48Grade", "fc.hippo.default.mode.phy", "Full.Scale.IQ_2")])
HipDMNVerbCov <- cov(retr_data[,c("Phy48Grade", "fc.hippo.default.mode.phy", "Verbal.Comprehension.Sum_2")])
HipDMNPercCov <- cov(retr_data[,c("Phy48Grade", "fc.hippo.default.mode.phy", "Perceptual.Reasoning.Sum_2")])

n <- nrow(retr_data)

FIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Full.Scale.IQ_2 ~ a*fc.hippo.default.mode.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.default.mode.phy + b*Full.Scale.IQ_2

	#Residual variances
	Full.Scale.IQ_2 ~~ Full.Scale.IQ_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.default.mode.phy ~~ fc.hippo.default.mode.phy

  ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
  c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

PIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Perceptual.Reasoning.Sum_2 ~ a*fc.hippo.default.mode.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.default.mode.phy + b*Perceptual.Reasoning.Sum_2

	#Residual variances
	Perceptual.Reasoning.Sum_2 ~~ Perceptual.Reasoning.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.default.mode.phy ~~ fc.hippo.default.mode.phy

  ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
  c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

VIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Verbal.Comprehension.Sum_2 ~ a*fc.hippo.default.mode.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.default.mode.phy + b*Verbal.Comprehension.Sum_2

	#Residual variances
	Verbal.Comprehension.Sum_2 ~~ Verbal.Comprehension.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.default.mode.phy ~~ fc.hippo.default.mode.phy

  ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
  c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

FIQMediationFit <- lavaan(model = FIQMediationSyntax, sample.cov = HipDMNFullCov, sample.nobs = n)
summary(FIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(FIQMediationFit, standardized = T)

VIQMediationFit <- lavaan(model = VIQMediationSyntax, sample.cov = HipDMNVerbCov, sample.nobs = n)
summary(VIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(VIQMediationFit, standardized = T)

PIQMediationFit <- lavaan(model = PIQMediationSyntax, sample.cov = HipDMNPercCov, sample.nobs = n)
summary(PIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(PIQMediationFit, standardized = T)

#and now we test the hippocampus-lcen connection
HipLCENFullCov <- cov(retr_data[,c("Phy48Grade", "fc.hippo.left.central.executive.phy", "Full.Scale.IQ_2")])
HipLCENVerbCov <- cov(retr_data[,c("Phy48Grade", "fc.hippo.left.central.executive.phy", "Verbal.Comprehension.Sum_2")])
HipLCENPercCov <- cov(retr_data[,c("Phy48Grade", "fc.hippo.left.central.executive.phy", "Perceptual.Reasoning.Sum_2")])

n <- nrow(retr_data)

FIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Full.Scale.IQ_2 ~ a*fc.hippo.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.left.central.executive.phy + b*Full.Scale.IQ_2

	#Residual variances
	Full.Scale.IQ_2 ~~ Full.Scale.IQ_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.left.central.executive.phy ~~ fc.hippo.left.central.executive.phy

  ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
  c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

PIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Perceptual.Reasoning.Sum_2 ~ a*fc.hippo.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.left.central.executive.phy + b*Perceptual.Reasoning.Sum_2

	#Residual variances
	Perceptual.Reasoning.Sum_2 ~~ Perceptual.Reasoning.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.left.central.executive.phy ~~ fc.hippo.left.central.executive.phy

  ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
  c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

VIQMediationSyntax <- "
	#M ~ a*X regression (connectivity predicts IQ)
	Verbal.Comprehension.Sum_2 ~ a*fc.hippo.left.central.executive.phy
  #Y ~ cPrime*X + b*M regression
  #Label the direct effect (cPrime) of CEN connectivity and direct effect of IQ (b) in the Grade regression.
  Phy48Grade ~ cPrime*fc.hippo.left.central.executive.phy + b*Verbal.Comprehension.Sum_2

	#Residual variances
	Verbal.Comprehension.Sum_2 ~~ Verbal.Comprehension.Sum_2          # ~~ indicates a two-headed arrow (variance or covariance)
	Phy48Grade ~~ Phy48Grade    # These lines say that IQ and Grade both have residual variances
  fc.hippo.left.central.executive.phy ~~ fc.hippo.left.central.executive.phy

  ab := a*b                   #indirect effect of CEN connectivity on grade (mediated by iq)
  c := cPrime + ab            #total relationship of CEN connectivity on Grade
"

FIQMediationFit <- lavaan(model = FIQMediationSyntax, sample.cov = HipLCENFullCov, sample.nobs = n)
summary(FIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(FIQMediationFit, standardized = T)

VIQMediationFit <- lavaan(model = VIQMediationSyntax, sample.cov = HipLCENVerbCov, sample.nobs = n)
summary(VIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(VIQMediationFit, standardized = T)

PIQMediationFit <- lavaan(model = PIQMediationSyntax, sample.cov = HipLCENPercCov, sample.nobs = n)
summary(PIQMediationFit, fit.measures = T, rsquare = T, ci = T)
parameterEstimates(PIQMediationFit, standardized = T)
