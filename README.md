# 🧠 Detector de Tumores Cerebrales con YOLOv8
 
[![Hugging Face Spaces](https://img.shields.io/badge/🤗_Hugging_Face-Spaces-yellow)](https://huggingface.co/spaces/Jhrodriguezlo/detector-tumores-cerebrales)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
 
Sistema de **detección y clasificación automática de tumores cerebrales** en imágenes de resonancia magnética (MRI) usando la arquitectura YOLOv8. Implementado como proyecto final del curso de **Procesamiento Digital de Imágenes** en la Universidad Nacional de Colombia - Sede Manizales.
 
---
 
## 🎯 Demo en Vivo
 
🚀 **Prueba el modelo aquí**: [https://huggingface.co/spaces/Jhrodriguezlo/detector-tumores-cerebrales](https://huggingface.co/spaces/Jhrodriguezlo/detector-tumores-cerebrales)
 
El demo está desplegado en **Hugging Face Spaces** y permite subir imágenes MRI para obtener detecciones en tiempo real.
 
---
 
## 📋 Tabla de Contenidos
 
- [Contexto del Problema](#-contexto-del-problema)
- [Tarea de Visión por Computador](#-tarea-de-visión-por-computador)
- [Dataset](#-dataset)
- [Arquitectura del Modelo](#-arquitectura-del-modelo)
- [Resultados](#-resultados)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Cómo Reproducir el Experimento](#-cómo-reproducir-el-experimento)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Limitaciones y Trabajo Futuro](#-limitaciones-y-trabajo-futuro)
- [Referencias](#-referencias)
---
 
## 🩺 Contexto del Problema
 
Los **tumores cerebrales** representan una de las patologías más complejas y de alta mortalidad en oncología. Según el Instituto Nacional de Cancerología (INC), en Colombia se diagnostican aproximadamente **2.000 nuevos casos anuales**, y la **detección temprana puede aumentar la supervivencia hasta en un 60%**.
 
Sin embargo, la interpretación de imágenes de resonancia magnética por parte de radiólogos enfrenta retos importantes:
 
- ⏱️ **Tiempo limitado**: cada análisis puede tardar entre 5 y 15 minutos
- 😩 **Fatiga humana**: estudios muestran que la fatiga aumenta los falsos negativos hasta en un 30%
- 🏥 **Acceso desigual**: hospitales rurales y pequeños no siempre cuentan con especialistas dedicados
- 📈 **Alto volumen**: la carga diaria de imágenes es alta y creciente
Este proyecto propone un **sistema de apoyo al diagnóstico basado en Inteligencia Artificial** que asiste al radiólogo detectando y clasificando tumores cerebrales en segundos, con alta precisión.
 
> ⚠️ **Importante**: Este sistema es una herramienta de **APOYO** al diagnóstico. El diagnóstico final siempre debe ser realizado por un profesional médico cualificado.
 
---
 
## 🎯 Tarea de Visión por Computador
 
**Tipo de tarea**: Detección de objetos (Object Detection)
 
A diferencia de la clasificación (que solo identifica QUÉ hay en una imagen) o la segmentación (que identifica PÍXEL a PÍXEL los objetos), la **detección de objetos**:
 
- ✅ Identifica **QUÉ** tipo de tumor está presente
- ✅ Identifica **DÓNDE** está ubicado en la imagen (bounding box)
- ✅ Provee un **score de confianza** de la predicción
- ✅ Es lo suficientemente rápida para uso clínico en tiempo real
### Clases detectadas (4):
 
| # | Clase | Descripción |
|---|---|---|
| 0 | 🔴 **Glioma** | Tumor de células gliales |
| 1 | 🔵 **Meningioma** | Tumor de las meninges |
| 2 | 🟢 **No Tumor** | Imagen MRI sin presencia tumoral |
| 3 | 🟠 **Pituitary** | Tumor pituitario / hipofisario |
 
---
 
## 📊 Dataset
 
### Identificación del Dataset (REQUISITO OBLIGATORIO)
 
- **Nombre**: Labeled MRI Brain Tumor Dataset
- **Autor**: Ali Rostami
- **Plataforma**: Roboflow Universe
- **🔗 URL del Dataset**: [https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset)
- **Licencia**: CC BY 4.0 (Creative Commons Attribution 4.0)
- **Tipo de imagen**: Resonancia Magnética (MRI) cerebral
- **Total de imágenes**: 2.443
### División del dataset (Train / Val / Test)
 
| Split | % | Imágenes | Propósito |
|---|---|---|---|
| **Train** | 69% | 1.695 | Entrenamiento del modelo |
| **Validation** | 20% | 502 | Evaluación durante entrenamiento |
| **Test** | 10% | 246 | Evaluación final (nunca vistas) |
 
### Distribución por clase
 
| Clase | Train | Val | Test | Total | % |
|---|---|---|---|---|---|
| Glioma | 583 | 163 | 84 | 830 | 33.6% |
| Meningioma | 359 | 125 | 63 | 547 | 22.1% |
| No Tumor | 335 | 99 | 49 | 483 | 19.5% |
| Pituitary | 440 | 118 | 54 | 612 | 24.8% |
| **TOTAL** | **1.717** | **505** | **250** | **2.472** | **100%** |
 
**Ratio de desbalance**: 1.72× (moderado, aceptable)
 
---
 
## 🏗️ Arquitectura del Modelo
 
### YOLOv8n (You Only Look Once - versión 8 nano)
 
**YOLOv8** es un algoritmo de detección de objetos de tipo **single-stage**, lo que significa que procesa la imagen en una sola pasada (a diferencia de Faster R-CNN que la procesa en dos etapas). Esto lo hace ideal para aplicaciones en tiempo real.
 
### Especificaciones técnicas:
 
| Aspecto | Valor |
|---|---|
| **Variante** | YOLOv8n (nano) |
| **Parámetros** | 3.011.628 |
| **GFLOPs** | 8.2 |
| **Capas** | 130 |
| **Resolución de entrada** | 640 × 640 píxeles |
| **Tamaño del modelo** | 6.0 MB |
 
### Justificación de la elección:
 
1. **Eficiencia computacional**: la variante "nano" permite entrenamiento en GPU gratuita (Google Colab T4)
2. **Despliegue ligero**: 6 MB es ideal para producción en la nube y dispositivos edge
3. **Velocidad**: inferencia en menos de 1 segundo por imagen
4. **Suficiencia técnica**: para 4 clases y 2.443 imágenes, variantes mayores serían sobre-dimensionadas
5. **Estado del arte**: YOLOv8 es la versión más reciente y optimizada de la familia YOLO
### Transfer Learning
 
Se aplicó **transfer learning** desde pesos pre-entrenados en el dataset COCO (80 clases, millones de imágenes). De las 355 capas del modelo, **319 se inicializaron con pesos pre-entrenados**, manteniendo el conocimiento universal de detección de objetos y solo especializando las capas finales para las 4 clases médicas.
 
### Función de pérdida (Loss Function)
 
YOLOv8 utiliza una **función de pérdida compuesta** con 3 componentes:
 
| Componente | Función |
|---|---|
| **box_loss** | Error en la posición del bounding box (CIoU loss) |
| **cls_loss** | Error en la clasificación (BCE - Binary Cross Entropy) |
| **dfl_loss** | Distribution Focal Loss para precisión espacial fina |
 
### Hiperparámetros del entrenamiento
 
| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Epochs | 50 | Balance entre tiempo y convergencia |
| Batch size | 16 | Óptimo para GPU T4 (15 GB VRAM) |
| Image size | 640×640 | Estándar de YOLOv8 |
| Optimizer | AdamW | Seleccionado automáticamente por Ultralytics |
| Learning rate | 0.00125 | Ajustado automáticamente |
| Patience | 20 | Early stopping para evitar overfitting |
| Pretrained | True | Transfer learning desde COCO |
| Seed | 42 | Reproducibilidad |
 
---
 
## 📈 Resultados
 
### Métricas globales (conjunto de TEST - 246 imágenes)
 
| Métrica | Valor | Interpretación |
|---|---|---|
| **mAP@0.5** | **95.86%** | 🏆 Sobresaliente |
| **Precision** | **93.56%** | 🤩 Excelente |
| **Recall** | **91.56%** | 🤩 Excelente |
| **mAP@0.5:0.95** | **60.92%** | 🙂 Bueno (métrica estricta) |
 
### Métricas por clase
 
| Clase | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Glioma | 95.27% | 95.89% | 96.86% | 66.50% |
| Meningioma | 96.36% | 92.06% | 96.11% | 56.20% |
| No Tumor | 94.31% | 85.71% | 96.59% | 71.17% |
| Pituitary | 88.32% | 92.59% | 93.87% | 49.81% |
 
### Comparación con la literatura científica
 
Nuestro resultado **supera** publicaciones recientes sobre el mismo dataset:
 
| Estudio | Arquitectura | mAP@0.5 |
|---|---|---|
| **Este trabajo** | **YOLOv8n** | **95.86%** ⭐ |
| Frontiers in Oncology (2025) | YOLOv7 | 87.9% |
 
---
 
## 📁 Estructura del Repositorio
 
```
deteccion-tumores-cerebrales-yolov8/
│
├── 📁 notebooks/
│   ├── Deteccion_Tumores_Cerebrales.ipynb       # Notebook de entrenamiento
│   └── Exportacion_ExecuTorch_Tumores...ipynb   # Notebook de exportación
│
├── 📁 huggingface_space/
│   ├── app.py                                    # Aplicación Gradio
│   ├── best.pt                                   # Modelo entrenado (6 MB)
│   ├── requirements.txt                          # Dependencias Python
│   └── README.md                                 # Configuración del Space
│
├── LICENSE                                       # Licencia MIT
└── README.md                                     # Este archivo
```
 
---
 
## 🔄 Cómo Reproducir el Experimento
 
### Opción 1: Probar el modelo entrenado (más rápido)
 
Simplemente visita el demo desplegado en Hugging Face Spaces:
 
🔗 **[https://huggingface.co/spaces/Jhrodriguezlo/detector-tumores-cerebrales](https://huggingface.co/spaces/Jhrodriguezlo/detector-tumores-cerebrales)**
 
Sube cualquier imagen MRI cerebral y obtén la detección en tiempo real.
 
### Opción 2: Reentrenar el modelo desde cero
 
**Requisitos previos**:
- Cuenta en Google Colab (gratis)
- Cuenta en Roboflow (gratis)
- API Key de Roboflow
**Pasos**:
 
1. Clona este repositorio:
```bash
   git clone https://github.com/SebastianRodriguez05/deteccion-tumores-cerebrales-yolov8.git
   cd deteccion-tumores-cerebrales-yolov8
```
 
2. Abre el notebook de entrenamiento en Google Colab:
   - `notebooks/Deteccion_Tumores_Cerebrales.ipynb`
   - Activa GPU: `Runtime → Change runtime type → T4 GPU`
3. Configura tu API Key de Roboflow en la celda correspondiente
4. Ejecuta las celdas en orden (el entrenamiento toma ~26 minutos en T4)
5. El modelo entrenado se guardará en tu Google Drive
### Opción 3: Re-exportar el modelo
 
Para exportar el modelo a otros formatos (TorchScript, ONNX):
 
1. Abre `notebooks/Exportacion_ExecuTorch_Tumores_Cerebrales.ipynb` en Colab
2. Ejecuta las celdas para generar los formatos optimizados
### Opción 4: Desplegar tu propio Space
 
1. Crea una cuenta en [Hugging Face](https://huggingface.co)
2. Crea un nuevo Space con SDK Gradio
3. Sube los archivos de `huggingface_space/` al Space
4. El despliegue es automático
---
 
## 🛠️ Tecnologías Utilizadas
 
| Categoría | Herramienta |
|---|---|
| **Deep Learning** | PyTorch 2.x, Ultralytics YOLOv8 |
| **Visión por Computador** | OpenCV, Pillow |
| **Datos** | Roboflow, NumPy |
| **Entrenamiento** | Google Colab (Tesla T4 GPU) |
| **Interfaz** | Gradio 5.x |
| **Despliegue** | Hugging Face Spaces |
| **Exportación** | TorchScript, ONNX |
| **Versionado** | Git, GitHub |
 
---
 
## 🚨 Limitaciones y Trabajo Futuro
 
### Limitaciones reconocidas
 
1. **Dataset único**: el modelo fue entrenado con imágenes de una sola fuente, lo que puede afectar la generalización a equipos de MRI de diferentes marcas
2. **Tamaño del dataset**: 2.443 imágenes es modesto para aplicaciones médicas críticas
3. **Sin validación clínica**: el sistema requeriría validación con radiólogos certificados y aprobación regulatoria (FDA, INVIMA) antes de uso clínico real
4. **Solo 4 clases**: existen otros tipos de patologías cerebrales no incluidas (metástasis, schwannomas, etc.)
5. **Detección 2D**: el modelo procesa cortes 2D individuales, no volúmenes 3D completos
### Trabajo futuro
 
- 🔬 Validación con datasets multi-institucionales
- 🏥 Co-diseño con equipos de radiología clínica
- 🌍 Extensión a más tipos de patologías cerebrales
- 📦 Optimización para despliegue en dispositivos edge (móviles, embebidos)
- 🧠 Exploración de modelos 3D para análisis volumétrico
- 🎯 Implementación de estimación de incertidumbre
---
 
## 📚 Referencias
 
1. **Dataset**: Rostami, A. (s.f.). Labeled MRI Brain Tumor Dataset. *Roboflow Universe*. [https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset)
2. **YOLOv8**: Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. *Ultralytics*. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. **Comparación científica**: Frontiers in Oncology (2025). YOLOv7 para detección de tumores cerebrales.
4. **PyTorch**: Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
5. **Transfer Learning**: Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. *IEEE Transactions on Knowledge and Data Engineering*.
---
 
## 👨‍🎓 Información Académica
 
- **Estudiante**: Jhon Sebastian Rodriguez Lopez
- **Programa**: Universidad Nacional de Colombia - Sede Manizales
- **Materia**: Procesamiento Digital de Imágenes
- **Periodo**: 2026-I
- **Repositorio**: [https://github.com/SebastianRodriguez05/deteccion-tumores-cerebrales-yolov8](https://github.com/SebastianRodriguez05/deteccion-tumores-cerebrales-yolov8)
---
 
## 📜 Licencia
 
Este proyecto está licenciado bajo la **Licencia MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.
 
El dataset utilizado tiene licencia **CC BY 4.0** (Creative Commons Attribution 4.0).
 
---
 
## ⚠️ Aviso Médico
 
Este sistema es un proyecto **académico de demostración** y **NO debe utilizarse para diagnóstico médico real**. La detección y clasificación de tumores cerebrales debe ser realizada exclusivamente por profesionales médicos cualificados (neurólogos, radiólogos, oncólogos). Cualquier decisión clínica debe basarse en evaluación profesional completa.
 
---
 
<div align="center">
**🧠 Detector de Tumores Cerebrales con YOLOv8**
 
Hecho con 💙 en la Universidad Nacional de Colombia - Sede Manizales
 
[🚀 Demo en Vivo](https://huggingface.co/spaces/Jhrodriguezlo/detector-tumores-cerebrales) · [📊 Dataset](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset) · [📓 Notebooks](./notebooks/)
 
</div>
