# ============================================================
# 🧠 Detector de Tumores Cerebrales con YOLOv8
# Sistema de apoyo al diagnóstico radiológico mediante Deep Learning
# 
# Aplicación de demostración para Hugging Face Spaces
# Universidad Nacional de Colombia - Sede Manizales
# Procesamiento Digital de Imágenes
# ============================================================

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import os
from ultralytics import YOLO


# ============================================================
# CARGAR EL MODELO ENTRENADO
# ============================================================
MODEL_PATH = "best.pt"

print(f"📂 Cargando modelo desde: {MODEL_PATH}")
trained_model = YOLO(MODEL_PATH)
print("✅ Modelo cargado correctamente\n")


# ============================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================
def predecir_tumor(imagen_input, umbral_confianza):
    """
    Recibe una imagen y un umbral, retorna:
    - Imagen con bboxes dibujados
    - Texto con resumen de detecciones
    """
    if imagen_input is None:
        return None, "⚠️ Por favor sube una imagen"
    
    # Convertir a array NumPy
    img_array = np.array(imagen_input)
    
    # Hacer la predicción
    results = trained_model.predict(
        source=img_array,
        conf=umbral_confianza,
        iou=0.45,
        verbose=False
    )
    
    result = results[0]
    
    # Colores BGR para OpenCV
    colores = {
        0: (231, 76, 60),    # Glioma - rojo
        1: (52, 152, 219),   # Meningioma - azul
        2: (46, 204, 113),   # No Tumor - verde
        3: (243, 156, 18)    # Pituitary - naranja
    }
    
    # Copia para dibujar
    img_anotada = img_array.copy()
    if len(img_anotada.shape) == 2:
        img_anotada = cv2.cvtColor(img_anotada, cv2.COLOR_GRAY2RGB)
    elif img_anotada.shape[2] == 4:  # RGBA → RGB
        img_anotada = cv2.cvtColor(img_anotada, cv2.COLOR_RGBA2RGB)
    
    # Construir resumen
    texto_resumen = "🔍 **RESULTADOS DEL ANÁLISIS**\n\n"
    
    if len(result.boxes) == 0:
        texto_resumen += "⚠️ **No se detectaron tumores** en esta imagen.\n\n"
        texto_resumen += f"Umbral usado: {umbral_confianza:.2f}\n"
        texto_resumen += "Sugerencia: prueba bajar el umbral de confianza."
    else:
        texto_resumen += f"✅ **{len(result.boxes)} detección(es) encontrada(s)**\n\n"
        
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            # Dibujar bbox
            color = colores[cls_id]
            cv2.rectangle(img_anotada,
                         (xyxy[0], xyxy[1]),
                         (xyxy[2], xyxy[3]),
                         color, 3)
            
            # Etiqueta
            label = f"{cls_name} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_anotada,
                         (xyxy[0], xyxy[1] - text_h - 10),
                         (xyxy[0] + text_w + 4, xyxy[1]),
                         color, -1)
            cv2.putText(img_anotada, label,
                       (xyxy[0] + 2, xyxy[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Agregar al resumen
            texto_resumen += f"**Detección #{i+1}:**\n"
            texto_resumen += f"  - 🏷️ Clase: **{cls_name}**\n"
            texto_resumen += f"  - 📊 Confianza: **{conf*100:.2f}%**\n"
            texto_resumen += f"  - 📐 Ubicación: ({xyxy[0]}, {xyxy[1]}) - ({xyxy[2]}, {xyxy[3]})\n\n"
        
        texto_resumen += "---\n"
        texto_resumen += "⚕️ **Nota clínica**: Este resultado es generado por IA y debe ser revisado por un profesional médico. NO constituye un diagnóstico definitivo."
    
    return img_anotada, texto_resumen


# ============================================================
# INTERFAZ GRADIO
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Detector de Tumores Cerebrales") as demo:
    
    gr.Markdown("""
    # 🧠 Detector de Tumores Cerebrales con YOLOv8
    
    ### Sistema de apoyo al diagnóstico radiológico mediante Deep Learning
    
    Este sistema utiliza una red neuronal **YOLOv8** entrenada sobre **2.443 imágenes de resonancia magnética cerebral** 
    para detectar y clasificar 4 categorías:
    
    - 🔴 **Glioma** - Tumor de células gliales
    - 🔵 **Meningioma** - Tumor de las meninges  
    - 🟢 **No Tumor** - Imagen sin presencia tumoral
    - 🟠 **Pituitary** - Tumor pituitario / hipofisario
    
    **Instrucciones**: Sube una imagen de MRI cerebral y ajusta el umbral de confianza si lo deseas.
    
    ⚠️ *Este sistema es una herramienta de APOYO. El diagnóstico final siempre debe ser realizado por un profesional médico.*
    """)
    
    with gr.Row():
        # Columna izquierda: inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Entrada")
            input_image = gr.Image(
                type="pil",
                label="Sube una imagen MRI",
                height=400
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="🎯 Umbral de confianza",
                info="Valores bajos detectan más (incluso dudosos). Valores altos solo detectan los muy seguros."
            )
            btn_predict = gr.Button("🔍 Analizar imagen", variant="primary", size="lg")
        
        # Columna derecha: outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Resultados")
            output_image = gr.Image(
                label="Imagen con detecciones",
                height=400
            )
            output_text = gr.Markdown("*Sube una imagen y presiona 'Analizar' para ver los resultados.*")
    
    # Conectar el botón con la función
    btn_predict.click(
        fn=predecir_tumor,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_text]
    )
    
    # Información al final
    gr.Markdown("""
    ---
    ### 📊 Métricas del modelo (en conjunto de test - 246 imágenes):
    
    | Métrica | Valor |
    |---|---|
    | **mAP@0.5** | 95.86% |
    | **Precision** | 93.56% |
    | **Recall** | 91.56% |
    
    ---
    
    **Tecnologías**: YOLOv8n · PyTorch · Roboflow · Hugging Face Spaces  
    **Dataset**: Labeled MRI Brain Tumor Dataset (Ali Rostami, CC BY 4.0)  
    **Universidad Nacional de Colombia** · Sede Manizales · Procesamiento Digital de Imágenes
    """)


# ============================================================
# LANZAR LA APP
# ============================================================
if __name__ == "__main__":
    demo.launch()
