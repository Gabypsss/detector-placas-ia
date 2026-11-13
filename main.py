
import os
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, Request, HTTPException
from tflite_runtime.interpreter import Interpreter

app = FastAPI(title="Detector de Placas IA - PictoBlox")

RUTA_MODELO = os.path.join(
    os.path.dirname(__file__),
    "DetectorPlacas_Fijo.tflite"
)

UMBRAL_CONFIANZA = 0.30

# Cargar el modelo optimizado
interprete = Interpreter(
    model_path=RUTA_MODELO,
    num_threads=1
)

interprete.allocate_tensors()

entrada = interprete.get_input_details()[0]
salidas = interprete.get_output_details()


@app.get("/")
def inicio():
    return {
        "estado": "activo",
        "modelo": "DetectorPlacas_V2",
        "formato": "TensorFlow Lite",
        "mensaje": "API de detección de placas"
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
        imagen = Image.open(
            BytesIO(imagen_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="La imagen no es válida"
        )

    ancho_original, alto_original = imagen.size

    # Preparar entrada de 320 x 320
    imagen_redimensionada = imagen.resize((320, 320))

    datos = np.array(
        imagen_redimensionada,
        dtype=np.uint8
    )

    interprete.set_tensor(
        entrada["index"],
        datos[None, ...]
    )

    # Ejecutar inferencia
    interprete.invoke()

    # Obtener resultados utilizando los nombres de salida
    resultados = {
        salida["name"]: interprete.get_tensor(
            salida["index"]
        )
        for salida in salidas
    }

    confianzas = resultados[
        "StatefulPartitionedCall:4"
    ][0]

    cajas = resultados[
        "StatefulPartitionedCall:1"
    ][0]

    cantidad = int(
        resultados["StatefulPartitionedCall:5"][0]
    )

    if cantidad <= 0:
        return {
            "detectada": False,
            "mensaje": "No se detectó ninguna placa",
            "confianza": 0,
            "coordenadas": None
        }

    indice = int(
        np.argmax(confianzas[:cantidad])
    )

    confianza = float(confianzas[indice])

    if confianza < UMBRAL_CONFIANZA:
        return {
            "detectada": False,
            "mensaje": "No se detectó ninguna placa",
            "confianza": round(confianza, 4),
            "coordenadas": None
        }

    ymin, xmin, ymax, xmax = cajas[indice]

    coordenadas = {
        "xmin": int(np.clip(xmin, 0, 1) * ancho_original),
        "ymin": int(np.clip(ymin, 0, 1) * alto_original),
        "xmax": int(np.clip(xmax, 0, 1) * ancho_original),
        "ymax": int(np.clip(ymax, 0, 1) * alto_original)
    }

    return {
        "detectada": True,
        "mensaje": "Placa detectada",
        "modelo": "DetectorPlacas_V2",
        "clase": "Placa",
        "confianza": round(confianza, 4),
        "coordenadas": coordenadas
    }
