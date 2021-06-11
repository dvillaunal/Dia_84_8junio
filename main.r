## ---- eval=FALSE, include=TRUE-------------------------------------------
## "Protocolo:
## 
##  1. Daniel Felipe Villa Rengifo
## 
##  2. Lenguaje: R
## 
##  3. Tema: Regresión logistica binaria: Modelos [glm] y Curvas ROC &amp; AUC
## 
##  4. Fuentes:
##     https://dlegorreta.wordpress.com/tag/e1071/"


## ------------------------------------------------------------------------
# Cargamos la base de datos:
Titanic_data <- read.csv(file = "Titanic_dataset.csv")

# echamos un vistazo a los datos
head(Titanic_data)

# Observamos las frequencias de la variable [Survived]
#table(Titanic_data$Survived)

write.table(table(Titanic_data$Survived),
            file = "FreqSurvived.txt", row.names = F)

# creamos una partición para crear un conjunto de test y otro de entrenamiento
#install.packages("caret")
library(caret)

# Creamos una semilla:
set.seed(123)

# creamos un vector de particion sobre la variable Survived
# el tamaño de muestra será de 70%

particion <- createDataPartition(Titanic_data$Survived, p=0.70)

# Eliminamos a "particion":
rm(particion)

# particion solo contiene un vector sobre la variable Survived
trainIndex <- particion$Resample1

# definimos dos conjuntos de muestra:

## conjunto entrenamiento
d_titanic_train <- Titanic_data[trainIndex, ]

## conjunto de test
d_titanic_test <- Titanic_data[-trainIndex, ]


"Una vez tenemos los conjuntos de test y de aprendizaje creamos el modelo, usando la misma simbología que en el caso de los modelos de naive_bayes., la diferencia con [gml]  es que tenemos que identificar un umbral de probabilidad a partir del que consideramos el pronostico 0 o 1."

# Construimos el modelo de predicción con la función glm
m_glm <- glm(Survived ~ Class+Sex, data = d_titanic_train, family = "binomial")

# resumen del modelo
summary(m_glm)

# vemos las predicciones en el conjunto de test
d_titanic_test$pred<-predict(m_glm, d_titanic_test, type= "response")

# Graficamos el resultado para una mejor comprensión:
# Exporamos el grafico

png(filename = "ModeloGMLtitanic.png")

hist(100*d_titanic_test$pred, col="skyblue",
     main=" resultados modelo glm() sobre datos Titanic test",
     xlab="Probabilidad en % de supervivencia",
     ylab="Frecuencia")

# Marcamos un umbral en el que consideramos el pronostico como donación
# este umbral lo ponemos en un valor del 60%
abline(v= 60,col= "navy", lwd=3)  # marcamos el umbral de supervivencia

dev.off()

# Convertimos los datos en 1 or 0 (si es mayor o menor que 0.6)
d_titanic_test$pred_final_60 <- ifelse(d_titanic_test$pred > 0.6, 1, 0)

# Ahora uan tabla de frequencias para los resultados
# resumen de resultados
write.table(table(d_titanic_test$pred_final_60),
            file = "FreqPred60.txt", row.names = F)

# vamos a cambiar los levels de survived No=0, Yes=1
table(d_titanic_test$Survived) # vemos cual es el primero ---> No

# podemos calcular el ajuste respecto a los casos reales con esta sencilla formula
# en vez de NO & YES por 1 or 0:
levels(d_titanic_test$Survived) <- c(0,1)
mean(d_titanic_test$pred_final_60 == d_titanic_test$Survived)
"Como vemos una vez realizado el pronostico podríamos probar diferentes umbrales y ver cual es el que da un mejor resultado con esta metodología."


## ------------------------------------------------------------------------
# Cargamos la libraría de graficos ROC
#install.packages("pROC")
library(pROC)

"Utilizaremos el modelo anterior para graficar ROC"
# Creamos una curva ROC basada en el modelo glm anterior
# Con la función roc() utlizamos la varible Survived y la predicción
ROC_glm60 <- roc(d_titanic_test$Survived, d_titanic_test$pred_final_60)

# Graficamos la curva ROC
#Exportmaos el grafico
png(filename = "ROC_gml60.png")

plot(ROC_glm60, col = "blue")

dev.off()

# Calculamos el area bajo la ROC(AUC)

auc(ROC_glm60) ## Resultado: "Area under the curve: 0.6671"

"Ahora porbaremos con un umbral del 40%, para ver si el modelo es mucho mejor"

# Convertimos los datos en 1 or 0 (si es mayor o menor que 0.4)
d_titanic_test$pred_final_40 <- ifelse(d_titanic_test$pred > 0.4, 1, 0)

# Creamos la curva ROC basada con en modelo de predicción al 40%
ROC_glm40 <-roc(d_titanic_test$Survived, d_titanic_test$pred_final_40)

# Graficamos la curva ROC
# Exportamos el resultado

png(filename = "ROC_glm40.png")

plot(ROC_glm40, col = "red")

dev.off()

# Calculamos el area bajo la ROC(AUC)
auc(ROC_glm40) ## Resultado: "Area under the curve: 0.723"


"Vistos los resultados, el seleccionar un umbral de 40, mejora la predicción de casos positivos de supervivencia."
