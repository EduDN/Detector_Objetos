# Importamos las librerias
from ultralytics import YOLO
import cv2

# Definimos la lista de modelos
modelos = ["platano.pt", "tetra.pt","manzana.pt"]
num_modelos = len(modelos)

# Creamos una lista de instancias de YOLO para cada modelo
yolo_models = [YOLO(modelo) for modelo in modelos]

# Realizar VideoCaptura
cap = cv2.VideoCapture(1)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Bucle para procesar cada modelo y superponer los resultados
    for i in range(num_modelos):
        # Leemos resultados para el modelo actual
        resultados = yolo_models[i].predict(frame, imgsz=640, conf=0.98)

        # Mostramos resultados para el modelo actual con menor opacidad (0.3)
        anotaciones = resultados[0].plot()
        frame = cv2.addWeighted(frame, 1, anotaciones, 0.3, 0)  # Superponer resultados

    # Mostramos nuestros fotogramas con resultados superpuestos
    cv2.imshow("DETECCION Y SEGMENTACION", frame)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
