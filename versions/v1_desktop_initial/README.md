# 🖥️ ORION v1: Desktop Inicial

Versión prototipo inicial del proyecto ORION. Implementa estimación de profundidad monocular basada en **MiDaS** (`MiDaS_small`, `DPT_Large` o `DPT_Hybrid`) con visualización gráfica síncrona mediante ventanas nativas de OpenCV (`cv2.imshow`).

---

## 🚀 Ejecución

Desde la raíz del proyecto:
```bash
python main.py --version 1
```

O directamente desde esta carpeta:
```bash
python main.py
```

### Controles en Ventana
* **`q` / `ESC`:** Salir de la aplicación.
* **`d`:** Alternar modo de visualización de mapa de profundidad.
