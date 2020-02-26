dat = read.csv(file='/Users/Katie/Dropbox/Projects/physics-retrieval/data/rescored/non-brain-data.csv')
small_dat = subset(dat, select=c("deltaPRI", "F", "GID.Post", "FCIPhyAcc2", "Age", "Mod", "Strt.Level", "SexXClass", "deltaPRIXClass", "deltaPRIXSex", "deltaPRIXClassXSex"))
library(lavaan)

mediationSyntax <- "
GID.Post ~ a*deltaPRI + d*deltaPRIXClass + f*deltaPRIXSex + h*deltaPRIXClassXSex + k*Strt.Level + Age + Mod + SexXClass + F
FCIPhyAcc2 ~ c_prime*deltaPRI + b*GID.Post + e_prime*deltaPRIXClass + g_prime*deltaPRIXSex + i_prime*deltaPRIXClassXSex + j*F + Age + Mod + Strt.Level + SexXClass
GID.Post ~~ GID.Post
FCIPhyAcc2 ~~ FCIPhyAcc2
ab := a*b
db := d*b
fb := f*b
hb := h*b
c := c_prime + ab
e := e_prime + db
g := g_prime + fb
i := i_prime + hb
"
modelFit = lavaan(model = mediationSyntax, data = small_dat)
summary(modelFit, fit.measures = TRUE)
