
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, Request, HTTPException

app = FastAPI(title="Detector de Placas IA - PictoBlox")

RUTA_MODELO = os.path.join(
    os.path.dirname(__file__),
    "modelo_tensorflow"
)

UMBRAL_CONFIANZA = 0.30

# Cargar modelo TensorFlow una sola vez
modelo = tf.saved_model.load(RUTA_MODELO)
detector = modelo.signatures["serving_default"]


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "modelo": "DetectorPlacas_V2",
        "mensaje": "API de detección con PictoBlox"
    }


@app.post("/detectar")
async def detectar_placa(request: Request):
    imagen_bytes = await request.body()

    if not imagen_bytes:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ninguna imagen"
        )

    if len(imagen_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="La imagen supera los 10 MB"
        )

    try:
        imagen = Image.open(BytesIO(imagen_bytes))
        imagen = imagen.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="El archivo no contiene una imagen válida"
        )

    entrada = tf.convert_to_tensor(
        np.array(imagen, dtype=np.uint8)[None, ...]
    )

    resultados = detector(input_tensor=entrada)

    confianzas = resultados["detection_scores"].numpy()[0]
    cajas = resultados["detection_boxes"].numpy()[0]

    cantidad = int(
        resultados["num_detections"].numpy()[0]
    )

    if cantidad == 0:
        return {
            "detectada": False,
            "confianza": 0,
            "coordenadas": None
        }

    indice = int(np.argmax(confianzas[:cantidad]))
    confianza = float(confianzas[indice])

    if confianza < UMBRAL_CONFIANZA:
        return {
            "detectada": False,
            "confianza": round(confianza, 4),
            "coordenadas": None
        }

    ymin, xmin, ymax, xmax = cajas[indice]
    ancho, alto = imagen.size

    return {
        "detectada": True,
        "modelo": "DetectorPlacas_V2",
        "clase": "Placa",
        "confianza": round(confianza, 4),
        "coordenadas": {
            "xmin": int(xmin * ancho),
            "ymin": int(ymin * alto),
            "xmax": int(xmax * ancho),
            "ymax": int(ymax * alto)
        }
    }
