#!/usr/bin/env python3
"""
Script de prueba rápida para verificar conectividad con ESP32-CAM.
"""

import requests
import socket

def test_esp32_connection(ip="172.16.121.9"):
    """Prueba la conexión con el ESP32-CAM."""
    print(f"🔍 Probando conexión con ESP32-CAM en {ip}")
    print("=" * 50)
    
    # 1. Test ping básico
    print("1️⃣ Probando conectividad básica...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((ip, 81))
            if result == 0:
                print(f"✅ Puerto 81 abierto en {ip}")
            else:
                print(f"❌ Puerto 81 cerrado en {ip}")
                return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False
    
    # 2. Test HTTP stream
    print("2️⃣ Probando stream HTTP...")
    try:
        url = f"http://{ip}:81/stream"
        response = requests.get(url, timeout=5, stream=True)
        print(f"📊 Status code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Stream HTTP funcionando")
            
            # Leer un poco del stream para verificar
            content_type = response.headers.get('content-type', '').lower()
            print(f"📄 Content-Type: {content_type}")
            
            if any(x in content_type for x in ['multipart', 'mjpeg', 'video', 'image']):
                print("✅ Stream de video detectado")
                return True
            else:
                print("⚠️  Stream responde pero content-type inesperado")
                return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error HTTP: {e}")
        return False

def main():
    print("🧪 Prueba de conectividad ESP32-CAM")
    print("=" * 50)
    
    # Probar IP conocida
    if test_esp32_connection("172.16.121.114"):
        print("\n🎉 ¡ESP32-CAM está funcionando correctamente!")
        print("📝 Comando para usar:")
        print("python main.py --src http://172.16.121.114:81/stream --esp-base http://172.16.121.114 --model-type MiDaS_small --proc-width 160 --async-capture")
    else:
        print("\n❌ No se pudo conectar al ESP32-CAM")
        print("💡 Verificaciones:")
        print("   - ¿El ESP32 está encendido?")
        print("   - ¿Está en la misma red WiFi?")
        print("   - ¿Puedes abrir http://172.16.121.114:81/stream en el navegador?")

if __name__ == "__main__":
    main()
