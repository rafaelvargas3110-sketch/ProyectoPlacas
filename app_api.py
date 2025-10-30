import os
import cv2
import json
import base64
import requests
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- NUEVAS IMPORTACIONES PARA POSTGRES (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# === CONFIGURACIÓN FLASK ===
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
CORS(app)

# Leemos desde variables de entorno
API_USERNAME = os.environ.get("API_USERNAME", "Rafael32") 
URL_API_PERU = "https://www.regcheck.org.uk/api/reg.asmx/CheckPeru"


# --- NUEVA CONFIGURACIÓN DE BASE DE DATOS (PostgreSQL) ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    # Esto es importante para saber si Render cargó la variable
    print("ALERTA: DATABASE_URL no está configurada.")
    # Fallará aquí si la variable no existe
    raise RuntimeError("DATABASE_URL no está configurada.")

# SQLAlchemy usa 'postgresql' en lugar de 'postgres' en la URL de Render
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Creamos el "motor" de la base de datos
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=5, max_overflow=10)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- DEFINICIÓN DE MODELOS (Tablas) ---
class Consulta(Base):
    __tablename__ = "consultas"
    id = Column(Integer, primary_key=True, index=True)
    placa = Column(String(50), index=True)
    marca = Column(String(100))
    modelo = Column(String(100))
    uso = Column(String(100))
    propietario = Column(String(255))
    imagen_url = Column(String(500))
    fecha_consulta = Column(DateTime, default=datetime.now)
    observaciones = Column(Text, default='')

class Reporte(Base):
    __tablename__ = "reportes"
    id = Column(Integer, primary_key=True, index=True)
    placa = Column(String(50), unique=True, index=True)
    descripcion = Column(Text, nullable=False)
    fecha_reporte = Column(DateTime, default=datetime.now)
    __table_args__ = (UniqueConstraint('placa', name='uq_placa'),)

# --- FUNCIÓN PARA MANEJAR SESIONES DE BD ---
@contextmanager
def get_db_session():
    """Maneja de forma segura las sesiones de la base de datos."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# --- FUNCIONES DE BASE DE DATOS REESCRITAS ---
def inicializar_db():
    """Crea las tablas en la base de datos si no existen."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tablas de la base de datos verificadas/creadas.")
    except Exception as e:
        print(f"❌ Error al inicializar la base de datos: {e}")

def guardar_consulta(placa, marca, modelo, uso, propietario, imagen_url):
    with get_db_session() as db:
        nueva_consulta = Consulta(
            placa=placa,
            marca=marca,
            modelo=modelo,
            uso=uso,
            propietario=propietario,
            imagen_url=imagen_url,
            fecha_consulta=datetime.now()
        )
        db.add(nueva_consulta)
        db.commit()

def verificar_reporte(placa):
    with get_db_session() as db:
        reporte = db.query(Reporte).filter(Reporte.placa == placa).first()
        return reporte.descripcion if reporte else None

def guardar_reporte(placa, descripcion):
    with get_db_session() as db:
        reporte_existente = db.query(Reporte).filter(Reporte.placa == placa).first()
        if reporte_existente:
            reporte_existente.descripcion = descripcion
            reporte_existente.fecha_reporte = datetime.now()
        else:
            nuevo_reporte = Reporte(
                placa=placa,
                descripcion=descripcion,
                fecha_reporte=datetime.now()
            )
            db.add(nuevo_reporte)
        db.commit()

def actualizar_observacion(id_registro, observacion):
    with get_db_session() as db:
        consulta = db.query(Consulta).filter(Consulta.id == id_registro).first()
        if consulta:
            consulta.observaciones = observacion
            db.commit()

# Llama a esta función UNA VEZ al iniciar la app
inicializar_db()

# === CARGA DE MODELOS (Lazy Loading) ===
MODELO_DETECTOR = None
OCR_READER = None

def cargar_modelos():
    """Carga los modelos en memoria en la primera petición."""
    global MODELO_DETECTOR, OCR_READER
    
    if MODELO_DETECTOR is None or OCR_READER is None:
        print("Iniciando carga de modelos (esto puede tardar)...")
        try:
            MODELO_DETECTOR = YOLO(os.path.join(BASE_DIR, "best.pt"))
            OCR_READER = PaddleOCR(use_textline_orientation=True, lang="en")
            print("✅ Modelos cargados correctamente.")
        except Exception as e:
            print(f"❌ ERROR al cargar modelos: {e}")
            MODELO_DETECTOR = None
            OCR_READER = None
    
    return MODELO_DETECTOR, OCR_READER

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
        
        # Guardar en la nueva base de datos Postgres
        guardar_consulta(placa_detectada, make, model, use_type, owner, image_url)
        
        return {"estado": estado, "imagen_url": image_url}
    except Exception as e:
        print(f"❌ Error en consultar_estado_legal: {e}")
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
    
    detector, ocr = cargar_modelos()
    
    if detector is None or ocr is None:
        return jsonify({"status": "error", "error": "Modelos no cargados o fallando al cargar."}), 503

    if 'image' not in request.files:
        return jsonify({"status": "error", "error": "No se encontró archivo de imagen."}), 400
    
    image_file = request.files['image']
    try:
        image_bytes = image_file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"status": "fail", "message": "No se pudo decodificar la imagen."}), 400

        image = cv2.resize(image, (1280, 720))
        results = detector.predict(image, verbose=False, conf=0.5)

        detected_plate = "PLACA_NO_ENCONTRADA"
        placa_base64 = None

        for r in results:
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                plate_roi = image[y1:y2, x1:x2]
                result = ocr.ocr(plate_roi)
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
        print(f"❌ Error en detect_plate: {e}") 
        return jsonify({"status": "error", "error": f"Error interno del servidor: {e}"}), 500

@app.route('/api/historial', methods=['GET'])
def historial():
    try:
        with get_db_session() as db:
            rows = db.query(Consulta).order_by(Consulta.fecha_consulta.desc()).all()
            historial = []
            for r in rows:
                historial.append({
                    "id": r.id, "placa": r.placa, "marca": r.marca, "modelo": r.modelo,
                    "uso": r.uso, "propietario": r.propietario, "imagen_url": r.imagen_url,
                    "fecha_consulta": r.fecha_consulta.strftime("%Y-%m-%d %H:%M:%S"), 
                    "observaciones": r.observaciones or ""
                })
            return jsonify({"status": "success", "historial": historial})
    except Exception as e:
        print(f"❌ Error en historial: {e}")
        return jsonify({"status": "error", "message": "Error al cargar historial"}), 500

@app.route('/api/observacion/<int:id>', methods=['PUT'])
def actualizar_observacion_api(id):
    try:
        data = request.get_json()
        obs = data.get("observacion", "")
        actualizar_observacion(id, obs)
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"❌ Error en actualizar_observacion_api: {e}")
        return jsonify({"status": "error", "message": "Error interno"}), 500


@app.route('/api/historial/<int:id>', methods=['DELETE'])
def eliminar_registro(id):
    try:
        with get_db_session() as db:
            consulta = db.query(Consulta).filter(Consulta.id == id).first()
            if consulta:
                db.delete(consulta)
                db.commit()
                return jsonify({"status": "success", "message": f"Registro {id} eliminado"})
            else:
                return jsonify({"status": "error", "message": "Registro no encontrado"}), 404
    except Exception as e:
        print(f"❌ Error al eliminar: {e}")
        return jsonify({"status": "error", "message": "Error interno"}), 500


@app.route('/api/historial/exportar', methods=['GET'])
def exportar_historial():
    try:
        with get_db_session() as db:
            rows = db.query(Consulta).order_by(Consulta.fecha_consulta.desc()).all()
            csv_data = "ID,Placa,Marca,Modelo,Uso,Propietario,Fecha,Observaciones\n"
            for r in rows:
                csv_data += f"{r.id},{r.placa},{r.marca},{r.modelo},{r.uso},{r.propietario},{r.fecha_consulta.strftime('%Y-%m-%d %H:%M:%S')},{r.observaciones}\n"
            response = app.response_class(
                response=csv_data,
                mimetype='text/csv',
                headers={"Content-Disposition": "attachment;filename=historial_consultas.csv"}
            )
            return response
    except Exception as e:
        print(f"❌ Error al exportar: {e}")
        return jsonify({"status": "error", "message": "Error al exportar datos"}), 500

@app.route('/api/reportar', methods=['POST'])
def reportar_vehiculo():
    try:
        data = request.get_json()
        placa = data.get("placa", "").strip().upper()
        descripcion = data.get("descripcion", "").strip()
        if not placa or not descripcion:
            return jsonify({"status": "error", "message": "Placa y descripción requeridas"}), 400
        guardar_reporte(placa, descripcion)
        return jsonify({"status": "success", "message": f"Reporte registrado para {placa}"})
    except Exception as e:
        print(f"❌ Error al reportar: {e}")
        return jsonify({"status": "error", "message": "Error interno"}), 500

if __name__ == "__main__":
    # Esta sección solo se usa para pruebas locales, NO en Render
    print("Iniciando en modo de desarrollo local (no usar en Render)...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)