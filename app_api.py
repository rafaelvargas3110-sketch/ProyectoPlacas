import os
import cv2
import json
import base64
import requests
import sqlite3
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from paddleocr import PaddleOCR

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# === CONFIGURACIÓN FLASK ===
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
CORS(app)

URL_API_PERU = "https://www.regcheck.org.uk/api/reg.asmx/CheckPeru"
API_USERNAME = "Rafael31"

# === BASE DE DATOS ===
DB_PATH = os.path.join(BASE_DIR, "placas.db")

def inicializar_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT,
            marca TEXT,
            modelo TEXT,
            uso TEXT,
            propietario TEXT,
            imagen_url TEXT,
            fecha_consulta TEXT,
            observaciones TEXT DEFAULT ''
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reportes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT UNIQUE,
            descripcion TEXT,
            fecha_reporte TEXT
        )
    ''')
    conn.commit()
    conn.close()

def guardar_consulta(placa, marca, modelo, uso, propietario, imagen_url):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO consultas (placa, marca, modelo, uso, propietario, imagen_url, fecha_consulta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (placa, marca, modelo, uso, propietario, imagen_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def verificar_reporte(placa):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT descripcion FROM reportes WHERE placa = ?", (placa,))
    reporte = cursor.fetchone()
    conn.close()
    return reporte[0] if reporte else None

def guardar_reporte(placa, descripcion):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO reportes (placa, descripcion, fecha_reporte)
        VALUES (?, ?, ?)
    ''', (placa, descripcion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def actualizar_observacion(id_registro, observacion):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE consultas SET observaciones = ? WHERE id = ?", (observacion, id_registro))
    conn.commit()
    conn.close()

inicializar_db()

# === CARGA DE MODELOS ===
try:
    MODELO_DETECTOR = YOLO(os.path.join(BASE_DIR, "best.pt")) # <--- AQUÍ
    OCR_READER = PaddleOCR(use_textline_orientation=True, lang="en")
    print("✅ Modelos cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR al cargar modelos: {e}") # <--- Este error está en tus logs
    MODELO_DETECTOR = None
    OCR_READER = None

# === CONSULTA API REGCHECK ===
def consultar_estado_legal(placa_detectada):
    params = {'RegistrationNumber': placa_detectada.replace("-", "").upper(), 'username': API_USERNAME}
    try:
        response = requests.get(URL_API_PERU, params=params, timeout=15)
        if response.status_code != 200:
            return {"estado": f"⚠ ERROR {response.status_code}: Servidor no disponible.", "imagen_url": None}
        if b"Peru Lookup failed" in response.content:
            return {"estado": f"❌ No se encontró la placa {placa_detectada}.", "imagen_url": None}

        root = ET.fromstring(response.content)
        namespace = {'regcheck': 'http://regcheck.org.uk'}
        json_string_element = root.find('regcheck:vehicleJson', namespace)

        if json_string_element is None or not json_string_element.text:
            return {"estado": f"⚠ No se encontraron datos para {placa_detectada}.", "imagen_url": None}

        vehicle_data = json.loads(json_string_element.text)
        make = vehicle_data.get('Make', 'N/A')
        model = vehicle_data.get('Model', 'N/A')
        year = vehicle_data.get('RegistrationYear', 'N/A')
        vin = vehicle_data.get('VIN', 'No disponible')
        use_type = vehicle_data.get('Use', 'Desconocido')
        owner = vehicle_data.get('Owner', 'No disponible')
        image_url = vehicle_data.get('ImageUrl', None)
        estado = f"✅ Marca: {make}, Modelo: {model}, Año: {year}, Uso: {use_type}, VIN: {vin}, Propietario: {owner}"
        guardar_consulta(placa_detectada, make, model, use_type, owner, image_url)
        return {"estado": estado, "imagen_url": image_url}
    except Exception as e:
        return {"estado": f"⚠ Error conectando con RegCheck: {e}", "imagen_url": None}

# === RUTAS FLASK ===
@app.route('/')
def index():
    return render_template("Index.html")

@app.route('/static/imagenes/<path:filename>')
def imagenes(filename):
    return send_from_directory(os.path.join(STATIC_DIR, "imagenes"), filename)

@app.route('/api/detect_plate', methods=['POST'])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({"status": "error", "error": "No se encontró archivo de imagen."}), 400
    if MODELO_DETECTOR is None or OCR_READER is None:
        return jsonify({"status": "error", "error": "Modelos no cargados correctamente."}), 503

    image_file = request.files['image']
    try:
        image_bytes = image_file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"status": "fail", "message": "No se pudo decodificar la imagen."}), 400

        image = cv2.resize(image, (1280, 720))
        results = MODELO_DETECTOR.predict(image, verbose=False, conf=0.5)

        detected_plate = "PLACA_NO_ENCONTRADA"
        placa_base64 = None

        for r in results:
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                plate_roi = image[y1:y2, x1:x2]
                result = OCR_READER.ocr(plate_roi)
                _, buffer = cv2.imencode('.jpg', plate_roi)
                placa_base64 = base64.b64encode(buffer).decode('utf-8')

                detected_plate = "OCR_FALLIDO"
                if result and isinstance(result[0], dict):
                    rec_texts = result[0].get("rec_texts", [])
                    if rec_texts:
                        detected_plate = "".join(filter(str.isalnum, rec_texts[0])).upper()
                elif result and isinstance(result[0], list):
                    try:
                        plate_text = result[0][0][1][0]
                        detected_plate = "".join(filter(str.isalnum, plate_text)).upper()
                    except Exception:
                        detected_plate = "OCR_FALLIDO"
                break

        if detected_plate in ["PLACA_NO_ENCONTRADA", "OCR_FALLIDO"]:
            return jsonify({"status": "fail", "placa_detectada": detected_plate, "message": "No se logró reconocer la placa."}), 400

        placa_limpia = detected_plate.replace("-", "").replace(" ", "")
        estado_legal_externo = consultar_estado_legal(placa_limpia)

        reporte = verificar_reporte(placa_limpia)
        if reporte:
            estado_legal_externo["estado"] += f" 🚨 ALERTA: {reporte}"

        return jsonify({
            "status": "success",
            "placa_detectada": detected_plate.upper(),
            "estado_legal": estado_legal_externo,
            "placa_imagen": placa_base64
        })

    except Exception as e:
        return jsonify({"status": "error", "error": f"Error interno del servidor: {e}"}), 500

@app.route('/api/historial', methods=['GET'])
def historial():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultas ORDER BY fecha_consulta DESC")
    rows = cursor.fetchall()
    conn.close()
    historial = []
    for r in rows:
        historial.append({
            "id": r[0], "placa": r[1], "marca": r[2], "modelo": r[3],
            "uso": r[4], "propietario": r[5], "imagen_url": r[6],
            "fecha_consulta": r[7], "observaciones": r[8] or ""
        })
    return jsonify({"status": "success", "historial": historial})

@app.route('/api/observacion/<int:id>', methods=['PUT'])
def actualizar_observacion_api(id):
    data = request.get_json()
    obs = data.get("observacion", "")
    actualizar_observacion(id, obs)
    return jsonify({"status": "success"})

@app.route('/api/historial/<int:id>', methods=['DELETE'])
def eliminar_registro(id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM consultas WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": f"Registro {id} eliminado"})

@app.route('/api/historial/exportar', methods=['GET'])
def exportar_historial():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultas ORDER BY fecha_consulta DESC")
    rows = cursor.fetchall()
    conn.close()
    csv_data = "ID,Placa,Marca,Modelo,Uso,Propietario,Fecha,Observaciones\n"
    for r in rows:
        csv_data += f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[7]},{r[8]}\n"
    response = app.response_class(
        response=csv_data,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=historial_consultas.csv"}
    )
    return response

@app.route('/api/reportar', methods=['POST'])
def reportar_vehiculo():
    data = request.get_json()
    placa = data.get("placa", "").strip().upper()
    descripcion = data.get("descripcion", "").strip()
    if not placa or not descripcion:
        return jsonify({"status": "error", "message": "Placa y descripción requeridas"}), 400
    guardar_reporte(placa, descripcion)
    return jsonify({"status": "success", "message": f"Reporte registrado para {placa}"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render asigna el puerto automáticamente
    app.run(host="0.0.0.0", port=port, debug=False)
