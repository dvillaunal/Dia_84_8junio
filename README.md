```{r, eval=FALSE, include=TRUE}
"Protocolo:
 
 1. Daniel Felipe Villa Rengifo
 
 2. Lenguaje: R
 
 3. Tema: Regresión logistica binaria: Modelos [glm] y Curvas ROC & AUC
 
 4. Fuentes:  
    https://dlegorreta.wordpress.com/tag/e1071/"
```


> Nota: Todos estos temas los trataremos gracias a que proximamente abordaremos el machine learning

# Regresión logistica binaria

Se trata de un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica (aquella que puede adoptar un número limitado de categorías) en función de las variables predictoras. Este modelo se enmarca dentro de los modelos denominados de predicción lineal generalizados o `glm` como son conocidos por sus siglas en inglés.

Con el adjetivo binario nos referimos a las predicciones sobre variables binarias o dicotómicas que simplemente tratan de decir si algo es [1 o 0, SI o NO].

Este modelo de pronóstico se usa mucho en variables que se distribuyen en forma de binomial.

+ Citado: _La denominación de logística se debe precisamente a la forma de la propia función de distribución de probabilidad binomial que presenta un crecimiento exponencial y que se parece a una_ `S` _y que toma el nombre matemático de función logística_ `(1+(e^-t))^-1`

la anterior curva es una aproximación continua a la función discreta binaria, pues el cambio de 0 a 1 se produce en corto espacio y muy pronunciado.

__Si usáramos otras funciones como la lineal para la regresión de datos binarios funcionaría muy mal__, pues el ajuste lineal no capta bien la forma de los datos.

lo que buscamos es dos agrupaciones que buscamos separar o clasificar.

Los modelos de regresión logísticos se generan con la función `glm()` del paquete base R `stats`.

Una vez dada esta introducción, daremos inicio al tema:

# Crear modelos `gml`

Con la base de datos (usada en replits anteriores) del Titanic, vamos a crear un modelo logístico que pronostique la variable Survived.

Al igual que todos los modelos de aprendizaje, el modelo se compone de una fórmula, y luego se pronostica con la función `predict()`.

> Nota: En los modelos `glm()`, los únicos argumentos de `predict()` son `response` y `terms`.

El primer caso da directamente la probabilidad de la respuesta y el segundo argumento proporciona los coeficientes de cada término en la fórmula, en caso de obtener un valor de predicción usaremos `type = "response"`.

```{r}
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
```

__Ahora trabajaremos las curvas ROC y AUC:__

# curvas ROC y AUC

Estas curvas nos ayudan a controlar el acierto o no de los modelos cuando uno de los eventos es muy raro. Esto implica que predecir el evento opuesto conlleva un gran porcentaje de aciertos, y en cierta forma falsea la utilidad real de la predicción lo que hay que vigilar y entender.

> Nota: En estos casos es mejor sacrificar los aciertos generales en favor de concentrarlos sobre uno de los resultados, el más raro, el que buscamos distinguir.

Por lo tanto la exactitud de la predicción general es una medida engañosa en el rendimiento de lo que realmente nos interesa.
Este es un caso muy común en predicciones binomiales pues un caso, el de éxito puede tener una probabilidad general mucho menor que el de fracaso, y un porcentaje de acierto elevado, puede no tener importancia, pues __lo que nos interesa no es acertar los fracasos sino los éxitos__.

Las __curvas ROC__ son buenas para evaluar este problema en conjuntos de datos desequilibrados.

## Una manera de entender lo que vamos a relizar:

Al hacer una gráfica ROC se representa mejor la compensación entre un modelo que es demasiado agresivo y uno que es demasiado pasivo. Lo que interesa es que el área de la curva sea máxima, cercana a 1, por lo que cuanto más se eleve respecto de la linea media mejor.

Estas gráficas se pintan con la libraría `pROC`. Usaremos dos funciones una para pintar la gráfica y otra que calcula el __AUC__ o área bajo la curva.

```{r}
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
```
