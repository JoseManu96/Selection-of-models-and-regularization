
# Incluimos las librerías:
library(MASS)
library(glmnet)
library(leaps)
library(pls)
library(psych)
library(caret)
library(mlbench)
library(car) 

# Fijamos la semilla:
set.seed(180)

# Analizamos los datos:
attach(Boston)
str(Boston)
Boston$rad=as.numeric(Boston$rad)
Boston$chas=as.factor(Boston$chas)

# Veamos las correlaciones:
pairs.panels(Boston[,-4],cex=2)

# Ajustamos nuestro primer modelo con todas las variables:
m0=lm(crim~.,data=Boston)

# Comprobamos significancia y el resumen del modelo:
drop1(m0, test="F") 
summary(m0) # R-squared_Adj:  0.44 

# Graficamos:
par(mfrow=c(2,2))
plot(m0)


# I
#
# Aplicamos BoxCox:
bc=boxcox(lm(crim~.,data=Boston))

#Revisamos el valor exacto de lambda:
best.lam<-bc$x[which(bc$y==max(bc$y))]
best.lam # Sugiere que hagamos log.

# Ajustamos un segundo modelo teniendo en cuenta las transformación:
m1=lm(log(crim)~.,data=Boston)

# Comprobamos significancia y el resumen del modelo:
drop1(m1, test="F") 
summary(m1) # R-squared_Adj:  0.872 

# Graficamos:
par(mfrow=c(2,2))
plot(m1)  # Mejora la linealidad respecto al anterior.


# II
#
# Modelo con los efectos principales más productos de variables 2 a 2:
m2=lm(log(crim)~.^2,data=Boston)

# Comprobamos significancia y el resumen del modelo:
drop1(m2, test="F") 
summary(m2) # R-squared_Adj:  0.924

# MSE:
mean(m2$residuals^2)


# Graficamos:
par(mfrow=c(2,2))
plot(m2)  # Mejora la linealidad respecto al anterior.


# III
#
# Ajustamos el modelo empleando usando busqueda exhaustiva (Se aplica al 
#resultado el steo BIC, por problema de capacidad de cómputo):
m3= regsubsets(log(crim) ~ zn + indus + chas + nox + rm + age + 
                 dis + rad + tax + ptratio + black + lstat + medv + zn:ptratio + 
                 zn:lstat + zn:medv + indus:chas + indus:age + indus:dis + 
                 indus:rad + indus:lstat + chas:rad + chas:tax + nox:rad + 
                 rm:age + rm:black + age:medv + dis:rad + dis:tax + dis:ptratio + 
                 rad:tax + rad:ptratio + rad:black + tax:black + black:lstat + 
                 lstat:medv, data=Boston,nvmax=38, nbest = 1, method="exhaustive",really.big=F)

# Veamos el summary:
summary(m3)

# RSS:
plot(m3$rss, pch=19, col="red", xlab="Tamaño del subconjunto (K)",
     ylab="RSS")

# Veamos los coeficientes que emplea el 3er modelo:
coef(m3,27)

# Variables del modelo:
names(coef(m3,27))

# Modelo escogido por el best subset:
m3f=lm(log(crim) ~ zn + nox + rm  + 
         dis + rad + tax  + black + lstat + medv + zn:ptratio + 
         zn:lstat + zn:medv + 
         indus:rad + indus:lstat  + nox:rad + 
         rm:age + rm:black + age:medv + dis:rad + dis:tax + dis:ptratio + 
         rad:tax + rad:ptratio + rad:black + tax:black + black:lstat + 
         lstat:medv, data=Boston)

# Summary:
summary(m3f)

# Empleamos la función step para hacer selección de variables y ajustar (k=2):
m4= step(lm(log(crim) ~.^2, data=Boston), trace = 0, k=2)

# Vemos el summary y la significancia de las variables:
drop1(m4, test="F")
summary(m4)

# MSE:
mean(m4$residuals^2)

# Empleamos la función step para hacer selección de variables y ajustar (k=log(506)):
m5= step(lm(log(crim) ~.^2, data=Boston), trace=0, k=log(506))

# Vemos el summary y la significancia de las variables:
drop1(m5,test= "F")
summary(m5)

# MSE:
mean(m5$residuals^2)

# IV
#
# Ridge:

set.seed(180)
# Definimos x e y como una matriz y un vector respectivamente:
y= log(Boston$crim)
x = model.matrix(log(crim) ~ .^2, data = Boston)

# MSE para difetentes valores de log(lambda):
cv.out = cv.glmnet(x, y,type.measure="mse",nfolds=10,alpha=0)
plot(cv.out)

# Buscamos el mejor lambda:
bestlam = cv.out$lambda.1se

# Regresión de Ridge para el modelo entero:
ridge = glmnet(x, y, alpha = 0)

# Calculamos el MSE:
y_hat = predict(ridge, s = bestlam, newx = x)
MSE <- mean((log(Boston$crim) - y_hat)^2)
print(sprintf("Ridge regression test MSE= %10.3f", MSE))

# Calculamos los errores:
assess.glmnet(ridge,s=cv.out$lambda.min,newx = x, newy = y)$mse # Error aparente
cv.out$cvm[which(cv.out$lambda==cv.out$lambda.min)] # Error de prueba por CV
cv.out$cvsd[which(cv.out$lambda==cv.out$lambda.min)] # Desviación estándar
cv.out$nzero[which(cv.out$lambda==cv.out$lambda.min)]+1 # Betas diferentes de cero
cv.out$lambda.min # Lambda empleado

# Graficamos el resultado de la regresión:
plot(ridge)

# Betas:
dim(ridge$beta)

# Lasso:

# MSE para difetentes valores de log(lambda):
set.seed(180)
cv.out = cv.glmnet(x, y, type.measure="mse", nfolds=10, alpha=1)
plot(cv.out)

# Buscamos el mejor lambda:
bestlam = cv.out$lambda.1se

# Regresión de Ridge para el modelo entero:
lasso = glmnet(x, y, alpha = 1)

# Calculamos el MSE:
y_hat = predict(lasso, s = bestlam, newx = x)
MSE <- mean((log(Boston$crim) - y_hat)^2)
print(sprintf("Lasso regression test MSE= %10.3f", MSE))

# Calculamos los errores:
assess.glmnet(lasso,s=cv.out$lambda.min,newx = x, newy = y)$mse # Error aparente
cv.out$cvm[which(cv.out$lambda==cv.out$lambda.min)] # Error de prueba por CV
cv.out$cvsd[which(cv.out$lambda==cv.out$lambda.min)] # Desviación estándar
cv.out$nzero[which(cv.out$lambda==cv.out$lambda.min)]+1 # Betas diferentes de cero
cv.out$lambda.min # Lambda empleado

# Graficamos el resultado de la regresión:
plot(lasso)

# Betas:
dim(lasso$beta)


# Elastic Net
set.seed(180)
# MSE para difetentes valores de log(lambda):
cv.out = cv.glmnet(x, y,type.measure="mse",nfolds=10,alpha=0.5)
plot(cv.out)

# Buscamos el mejor lambda:
bestlam = cv.out$lambda.1se

# Regresión de Ridge para el modelo entero:
elastic_net = glmnet(x, y, alpha = 0.5)

# Calculamos el MSE:
y_hat = predict(elastic_net, s = bestlam, newx = x)
MSE = mean((log(Boston$crim) - y_hat)^2)
print(sprintf("Elastic net regression test MSE= %10.3f", MSE))

# Calculamos los errores:
assess.glmnet(elastic_net,s=cv.out$lambda.min,newx = x, newy = y)$mse # Error aparente
cv.out$cvm[which(cv.out$lambda==cv.out$lambda.min)] # Error de prueba por CV
cv.out$cvsd[which(cv.out$lambda==cv.out$lambda.min)] # Desviación estándar
cv.out$nzero[which(cv.out$lambda==cv.out$lambda.min)]+1 # Betas diferentes de cero
cv.out$lambda.min # Lambda empleado

# Graficamos el resultado de la regresión:
plot(elastic_net)

# Betas:
dim(elastic_net$beta)


# Errores aparentes:
set.seed(180)
ntrain=createDataPartition(Boston$crim, p=0.7)$Resample1 # Índices de entrenamiento
train=Boston[ntrain,] # Muestra de entrenamiento

err_noreg=matrix(NA,nrow=4, ncol=5)
colnames(err_noreg)=c("App","Test","Rank","Deviance","R^2_adj")
rownames(err_noreg)=c("m2","m3","m4","m5")
mx=list(1,2,3,4,5,6,7) # Lista de modelos

mx[[1]]=lm(formula(m2),data=train)
mx[[2]]=lm(formula(m3f),data=train)
mx[[3]]=lm(formula(m4),data=train)
mx[[4]]=lm(formula(m5),data=train)

for (Nmod in c(1,2,3,4)){
  err_noreg[Nmod,1]=mean(mx[[Nmod]]$residuals^2)
  err_noreg[Nmod,3]=mx[[Nmod]]$rank
}

# Errores de predicción:

set.seed(180)
#Modelos no regularizados (Repeated Training Test):
N=500 # Número de repeticiones
ntrain=createDataPartition(Boston$crim, p=0.7)$Resample1 # Índices de entrenamiento
train=Boston[ntrain,] # Muestra de entrenamiento
test=Boston[-ntrain,] # Muestra de prueba
aux=matrix(NA,nrow=N, ncol=7)
colnames(aux)=c("Test_m2","Test_m3","Test_m4", "Test_m5","Test_Ridge","Test_Lasso","Test_Elast_net")
Maux=list(1,2,3,4,5,6,7)

t1=proc.time()
for(Nrep in 1:N){
  ntrain=createDataPartition(Boston$crim, p=0.7)$Resample1 # Índices de entrenamiento
  train=Boston[ntrain,] # Muestra de entrenamiento
  test=Boston[-ntrain,] 
  Maux[[1]]=lm(formula(m2),data=train)
  Maux[[2]]=lm(formula(m3f),data=train)
  Maux[[3]]=lm(formula(m4),data=train)
  Maux[[4]]=lm(formula(m5),data=train)
  aux[Nrep,1]=mean((predict(Maux[[1]],newdata=test) -log(crim)[-ntrain])^2)
  aux[Nrep,2]=mean((predict(Maux[[2]],newdata=test) -log(crim)[-ntrain])^2)
  aux[Nrep,3]=mean((predict(Maux[[3]],newdata=test) -log(crim)[-ntrain])^2)
  aux[Nrep,4]=mean((predict(Maux[[4]],newdata=test) -log(crim)[-ntrain])^2)
}
(proc.time()-t1)

for (Nmod in c(1,2,3,4)){
  err_noreg[Nmod,2]=mean(aux[,Nmod])
}

err_noreg[c(1,3,4),4]=c(147.0498,151.4137,172.8271)
err_noreg[c(1,3,4),5]=c(92.4,92.75,92.12)
err_noreg # Lista de los errores para los modelos no regularizados



#Modelos regularizados
set.seed(180)
N=500
err_reg=matrix(NA,nrow=3, ncol=2)
colnames(err_reg)=c("App","Test")
rownames(err_reg)=c("Ridge","Lasso","Elastic")
Mcvaux=list(1,2,3,4,5,6,7,8)

t1=proc.time()
alp=c(0,1,.5)  #1:lasso, 0:ridge, .5:elastic net.
set.seed(3)
for(Nrep in 1:N){
    ntrain=createDataPartition(Boston$crim, p=0.7)$Resample1 # Índices de entrenamiento
    Maux[[1]] = glmnet(x[ntrain,],y[ntrain], alpha=0)
    Maux[[2]] = glmnet(x[ntrain,],y[ntrain], alpha=1)
    Maux[[3]] = glmnet(x[ntrain,],y[ntrain], alpha=0.5)
    Mcvaux[[1]]=cv.glmnet(x[ntrain,], y[ntrain],type.measure="mse",nfolds=10,alpha=0)
    Mcvaux[[2]]=cv.glmnet(x[ntrain,], y[ntrain],type.measure="mse",nfolds=10,alpha=1)
    Mcvaux[[3]]=cv.glmnet(x[ntrain,], y[ntrain],type.measure="mse",nfolds=10,alpha=0.5)
    aux[Nrep,5]=assess.glmnet(Maux[[1]],s=Mcvaux[[1]]$lambda.min ,newx = x[-ntrain,], newy = y[-ntrain])$mse
    aux[Nrep,6]=assess.glmnet(Maux[[2]],s=Mcvaux[[2]]$lambda.min ,newx = x[-ntrain,], newy = y[-ntrain])$mse
    aux[Nrep,7]=assess.glmnet(Maux[[3]],s=Mcvaux[[3]]$lambda.min ,newx = x[-ntrain,], newy = y[-ntrain])$mse
}
(proc.time()-t1)

for (Nmod in c(1:3)){
  err_reg[Nmod,2]=mean(aux[1:N,(Nmod+4)])
}

err_reg[c(1:3),1]=c(0.4962495 ,0.3447591   ,0.3516535 )
err_reg # Lista de los errores para los modelos regularizados

# R-cuadrado:

# Definimos la función evaluadora:
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Lasso:
set.seed(180)
cv.out = cv.glmnet(x, y, type.measure="mse", nfolds=10, alpha=1)
bestlam = cv.out$lambda.1se
predictions_train <- predict(lasso, s = bestlam, newx = x)
eval_results(y, predictions_train, x)

# Ridge:
cv.out = cv.glmnet(x, y, type.measure="mse", nfolds=10, alpha=0)
bestlam = cv.out$lambda.1se
predictions_train <- predict(ridge, s = bestlam, newx = x)
eval_results(y, predictions_train, x)

# Elastic net:
cv.out = cv.glmnet(x, y, type.measure="mse", nfolds=10, alpha=0.5)
bestlam = cv.out$lambda.1se
predictions_train <- predict(elastic_net, s = bestlam, newx = x)
eval_results(y, predictions_train, x)


# Coeficientes:

set.seed(180)
# Imprimimos los coeficientes del modelo 2:
round(cbind("LSBetas"=m2$coefficients,
            "sd(LSbetas)"=sqrt(diag(vcov(m2))),
            m2$coefficients/sqrt(diag(vcov(m2))) ),4)

# Imprimimos los coeficientes del modelo 4:
round(cbind("LSBetas"=m4$coefficients,
            "sd(LSbetas)"=sqrt(diag(vcov(m4))),
            m4$coefficients/sqrt(diag(vcov(m4))) ),4)

# Imprimimos los coeficientes del modelo 5:
round(cbind("LSBetas"=m5$coefficients,
            "sd(LSbetas)"=sqrt(diag(vcov(m5))),
            m5$coefficients/sqrt(diag(vcov(m5))) ),4)

Mcv=list(3) # Guaradamos los cv de cada modelo regularizado

set.seed(180)
# Imprimimos los coeficientes del Ridge:
(Mcv[[1]]=cv.glmnet(x, y,type.measure="mse",nfolds=10,alpha=0))
round(predict.glmnet(ridge, s=Mcv[[1]]$lambda.min, type = "coefficients"),4)
(betasR=as.matrix(predict.glmnet(ridge, s=Mcv[[1]]$lambda.min, type = "coefficients")))

# Imprimimos los coeficientes del Lasso:
(Mcv[[2]]=cv.glmnet(x, y,type.measure="mse",nfolds=10,alpha=1))
round(predict.glmnet(lasso, s=Mcv[[2]]$lambda.min, type = "coefficients"),4)
(betasL=as.matrix(predict.glmnet(lasso, s=Mcv[[2]]$lambda.min, type = "coefficients")))

# Imprimimos los coeficientes del Elastic net:
(Mcv[[3]]=cv.glmnet(x, y,type.measure="mse",nfolds=10,alpha=0.5))
round(predict.glmnet(elastic_net, s=Mcv[[3]]$lambda.min, type = "coefficients"),4)
(betasE=as.matrix(predict.glmnet(elastic_net, s=Mcv[[3]]$lambda.min, type = "coefficients")))

# Filas distintas de cero:
rownames(betasR)[betasR!=0] #92 # Ridge
rownames(betasL)[betasL!=0] #72 # Lasso
rownames(betasE)[betasE!=0] #82 # Elastic net

#LS
par(mfrow=c(1,3), mar=c(4,4,6,2))
plot(Mcv[[1]], main="Ridge", cex.main=1.3)
plot(Mcv[[2]], main="Lasso", cex.main=1.3)
plot(Mcv[[3]], main="Elastic Net", cex.main=1.3)

set.seed(180)
Mcv[[1]]$lambda.min; Mcv[[1]]$lambda.1se # Ridge
Mcv[[2]]$lambda.min; Mcv[[2]]$lambda.1se # Lasso
Mcv[[3]]$lambda.min; Mcv[[3]]$lambda.1se # Elastic net

# Graficamos los modelos regularizados:
par(mfrow=c(1,1), mar=c(4,4,6,2))
plot(ridge, main="Ridge",label = T,cex.main=1)
plot(elastic_net, main="Lasso",label = T, cex.main=1)
plot(lasso, main="Elastic Net",label = T, cex.main=1)

# Guardamos los resultados:
write.csv(err_reg,'ModelReg')
write.csv(err_noreg,'ModelNReg')

# Comparamos los valores:
compareCoefs(m2,m4,m5,zvals=TRUE, pvals=TRUE)
compareCoefs(m2,m4,m5,zvals=TRUE)
compareCoefs(elastic_net,lasso,ridge,zvals=TRUE, pvals=TRUE) #no applicable!

library(xtable)

coefi=matrix(NA,nrow=92,ncol = 8)
coefi[1:92,2]=round(m2$coefficients,4)
coefi[1:28,3]=round(m3f$coefficients,4)
coefi[1:59,4]=round(m4$coefficients,4)
coefi[1:37,5]=round(m5$coefficients,4)
coefi[,6]=round(as.numeric(betasR[c(1,3:93)]),4)
coefi[,7]=round(as.numeric(betasL[c(1,3:93)]),4)
coefi[,8]=round(as.numeric(betasE[c(1,3:93)]),4)
coefi[,1]=names(coef(m2))
coefi[1:37,]
print(xtable(coefi), include.rownames = FALSE)

# Desviaciones estandar:
sd(aux[,1])
sd(aux[,2])
sd(aux[,3])
sd(aux[,4])
sd(aux[,5])
sd(aux[,6])
sd(aux[,7])


