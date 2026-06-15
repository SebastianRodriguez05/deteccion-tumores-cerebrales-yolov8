---
title: Detector Tumores Cerebrales
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# 🧠 Detector de Tumores Cerebrales con YOLOv8

Sistema de apoyo al diagnóstico radiológico mediante Deep Learning.

## Descripción

Este sistema utiliza una red neuronal **YOLOv8** entrenada sobre **2.443 imágenes de resonancia magnética cerebral** 
para detectar y clasificar 4 categorías:

- 🔴 **Glioma** - Tumor de células gliales
- 🔵 **Meningioma** - Tumor de las meninges  
- 🟢 **No Tumor** - Imagen sin presencia tumoral
- 🟠 **Pituitary** - Tumor pituitario / hipofisario

## Métricas (Test set - 246 imágenes)

| Métrica | Valor |
|---|---|
| mAP@0.5 | 95.86% |
| Precision | 93.56% |
| Recall | 91.56% |

## Dataset

- **Nombre**: Labeled MRI Brain Tumor Dataset
- **Autor**: Ali Rostami
- **Fuente**: Roboflow Universe
- **Licencia**: CC BY 4.0

## Aviso médico

⚠️ Este sistema es una herramienta de **APOYO** al diagnóstico. El diagnóstico final siempre debe ser realizado por un profesional médico.

## Universidad

Universidad Nacional de Colombia · Sede Manizales  
Materia: Procesamiento Digital de Imágenes
