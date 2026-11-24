# 🕵️ El Detective de Señales

Clasificación automática de modulaciones (**ASK, FSK, PSK, QAM**) con Deep Learning (CNN 1D en Keras/TensorFlow), incluso en presencia de ruido.

## Estructura del proyecto
```
detective_senales/
├── El_Detective_de_Senales.ipynb   # Notebook completo (3 fases)
├── src/signals.py                  # Generador de señales moduladas
├── results/                        # Gráficas y modelo entrenado
├── requirements.txt
└── README.md
```

## Fases
1. **Fase 1 – Dataset:** generación de señales I/Q (ASK, FSK, PSK, QAM) con ruido AWGN (SNR de -4 a 20 dB).
2. **Fase 2 – Modelo:** CNN 1D con 4 bloques Conv1D + BatchNorm + MaxPool.
3. **Fase 3 – Evaluación:** matriz de confusión, precisión vs. SNR y conclusiones.

## Cómo ejecutarlo
**Opción A (Google Colab):** sube el notebook a [colab.research.google.com](https://colab.research.google.com) → *Entorno de ejecución > Ejecutar todo*.

**Opción B (local):**
```bash
pip install -r requirements.txt
jupyter notebook El_Detective_de_Senales.ipynb
```

## Resultados
*(Pega aquí tus gráficas y tu precisión final, por ejemplo:)*

![Precisión vs SNR](results/04_precision_vs_snr.png)

## Herramientas
Python, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras.
