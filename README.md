# 🌟 Neural Network Keras 🌟

**👤 Autor:** *Alejandro Garcia*\
**📅 Fecha:** *7 de marzo*\
**📷 Instagram:** **[@ale\_garcia454](https://www.instagram.com/ale_garcia454/)**

🚀 Este proyecto implementa una **red neuronal para clasificación de dígitos manuscritos** utilizando *Python* y librerías como **Keras, TensorFlow y Matplotlib**. La red neuronal se entrena en el conjunto de datos **MNIST**.

![MNIST Neural Network](https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif)

---

## 🎯 Características

👉 Implementación de una **red neuronal multicapa** con *Keras*.\
👉 Entrenamiento en el **conjunto de datos MNIST**.\
👉 **Normalización de imágenes** y conversión de etiquetas a *one-hot encoding*.\
👉 Visualización de **ejemplo de imagen** del dataset.\
👉 Evaluación del modelo con **precisión en datos de prueba**.

![Gato en computadora](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)

---

## 🔧 Requisitos

Asegúrate de tener instaladas las siguientes librerías antes de ejecutar el código:

```bash
pip install numpy keras tensorflow matplotlib
```

---

## 🚀 Uso

Para ejecutar el código, simplemente corre el siguiente comando en tu terminal:

```bash
python main.py
```

Esto **entrenará la red neuronal con el dataset MNIST** y mostrará una imagen de ejemplo.


---


## 📌 Estructura del Código

📚 **train\_and\_evaluate()**: Función principal que entrena y evalúa la red.\
📚 **mnist.load\_data()**: Carga el conjunto de datos de imágenes manuscritas.\
📚 **Sequential()**: Define la estructura de la red neuronal.\
📚 **model.fit()**: Realiza el entrenamiento.\
📚 **model.evaluate()**: Evalúa la precisión en datos de prueba.
---

## 🏅 main.py

Este es el script principal que ejecuta el entrenamiento de la red neuronal:

```python
# Importar la función de entrenamiento y evaluación desde el módulo neural_network_keras
from src.neural_network_keras import train_and_evaluate

# Verifica si el script se ejecuta directamente
if __name__ == "__main__":
    train_and_evaluate()  # Llama a la función para entrenar y evaluar la red neuronal
```

---

## 📚 Estructura del Proyecto

La estructura del proyecto es la siguiente:

```
📂 main/
👉📂 src/
👉   📂 __pycache__/
👉   📄 neural_network_keras.py
👉📄 .gitignore
👉📄 README.md
👉📄 main.py
👉📄 requirements.txt
```

---

##  Autor

📌 Desarrollado por **Alejandro Garcia**.

