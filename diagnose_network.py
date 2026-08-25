#!/usr/bin/env python3
"""
Script de diagnóstico para verificar la conectividad de red y encontrar problemas.
"""

import socket
import requests
import subprocess
import platform
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_local_ip():
    """Obtiene la IP local de la máquina."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "192.168.1.1"

def get_network_info():
    """Obtiene información de la red local."""
    local_ip = get_local_ip()
    print(f"🌐 IP local: {local_ip}")
    
    # Determinar rango de red
    if local_ip.startswith("172.16"):
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        print(f"📡 Rango de red: {network}")
        return [str(ip) for ip in network.hosts()]
    elif local_ip.startswith("192.168"):
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        print(f"📡 Rango de red: {network}")
        return [str(ip) for ip in network.hosts()]
    else:
        print(f"⚠️  Red no reconocida: {local_ip}")
        return []

def check_ping(ip):
    """Verifica si una IP responde a ping."""
    try:
        if platform.system().lower() == "windows":
            result = subprocess.run(['ping', '-n', '1', '-w', '1000', ip], 
                                 capture_output=True, timeout=3)
        else:
            result = subprocess.run(['ping', '-c', '1', '-W', '1', ip], 
                                 capture_output=True, timeout=3)
        return result.returncode == 0
    except:
        return False

def check_port(ip, port=81):
    """Verifica si un puerto está abierto."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((ip, port))
            return result == 0
    except:
        return False

def check_http_service(ip, port=81):
    """Verifica si hay un servicio HTTP en el puerto."""
    try:
        url = f"http://{ip}:{port}"
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def scan_common_esp32_ips():
    """Escanea IPs comunes donde suele estar el ESP32."""
    common_ips = [
        "192.168.1.100", "192.168.1.101", "192.168.1.102",
        "192.168.0.100", "192.168.0.101", "192.168.0.102",
        "172.16.121.9", "172.16.121.10", "172.16.121.11",
        "10.0.0.100", "10.0.0.101", "10.0.0.102"
    ]
    
    print("🔍 Escaneando IPs comunes de ESP32...")
    found = []
    
    for ip in common_ips:
        if check_ping(ip):
            print(f"✅ Ping exitoso: {ip}")
            if check_port(ip, 81):
                print(f"🎯 Puerto 81 abierto: {ip}")
                if check_http_service(ip, 81):
                    print(f"🌐 Servicio HTTP activo: {ip}")
                    found.append(ip)
        else:
            print(f"❌ Sin ping: {ip}")
    
    return found

def main():
    print("🔧 Diagnóstico de red para ESP32-CAM")
    print("=" * 50)
    
    # Información de red
    network_ips = get_network_info()
    print(f"📊 Total de IPs a escanear: {len(network_ips)}")
    
    # Escanear IPs comunes primero
    common_found = scan_common_esp32_ips()
    
    if common_found:
        print(f"\n🎉 ESP32-CAM encontrado en IPs comunes: {common_found}")
        for ip in common_found:
            print(f"📝 Comando para usar: python main.py --src http://{ip}:81/stream --esp-base http://{ip}")
        return
    
    print("\n🔍 Escaneando rango completo de red...")
    print("⏳ Esto puede tomar un momento...")
    
    # Escanear rango completo
    found_ips = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ip = {executor.submit(check_ping, ip): ip for ip in network_ips[:50]}  # Limitar a 50 para prueba
        
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                if future.result():
                    print(f"✅ Dispositivo activo: {ip}")
                    if check_port(ip, 81):
                        print(f"🎯 Puerto 81 abierto: {ip}")
                        if check_http_service(ip, 81):
                            print(f"🌐 Servicio HTTP: {ip}")
                            found_ips.append(ip)
            except:
                pass
    
    if found_ips:
        print(f"\n🎉 ESP32-CAM encontrado: {found_ips}")
        for ip in found_ips:
            print(f"📝 Comando: python main.py --src http://{ip}:81/stream --esp-base http://{ip}")
    else:
        print("\n❌ No se encontró ESP32-CAM")
        print("💡 Verificaciones:")
        print("   1. ¿El ESP32 está encendido?")
        print("   2. ¿Está conectado a la misma red WiFi?")
        print("   3. ¿Puedes acceder manualmente a http://172.16.121.9:81/stream ?")
        print("   4. ¿El ESP32 está en un rango de IP diferente?")

if __name__ == "__main__":
    main()
