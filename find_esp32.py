#!/usr/bin/env python3
"""
Script para encontrar automáticamente la IP del ESP32-CAM en la red local.
"""

import socket
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import subprocess
import platform

def get_local_ip():
    """Obtiene la IP local de la máquina."""
    try:
        # Conecta a una dirección externa para determinar la IP local
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "192.168.1.1"  # fallback

def get_network_range():
    """Obtiene el rango de red local."""
    local_ip = get_local_ip()
    network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    return [str(ip) for ip in network.hosts()]

def check_esp32_port(ip, port=81, timeout=2):
    """Verifica si un puerto está abierto en una IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((ip, port))
            return result == 0
    except:
        return False

def check_esp32_stream(ip, timeout=3):
    """Verifica si hay un stream ESP32 en la IP."""
    try:
        url = f"http://{ip}:81/stream"
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            # Verificar que sea realmente un stream MJPEG
            content_type = response.headers.get('content-type', '').lower()
            if 'multipart' in content_type or 'mjpeg' in content_type:
                return True
    except:
        pass
    return False

def check_esp32_control(ip, timeout=2):
    """Verifica si hay un servidor de control ESP32 en la IP."""
    try:
        url = f"http://{ip}/control"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def scan_for_esp32():
    """Escanea la red local buscando ESP32-CAM."""
    print("🔍 Escaneando red local en busca de ESP32-CAM...")
    
    network_ips = get_network_range()
    print(f"📡 Escaneando {len(network_ips)} direcciones IP...")
    
    found_ips = []
    
    # Primero verificar puertos abiertos
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_ip = {
            executor.submit(check_esp32_port, ip): ip 
            for ip in network_ips
        }
        
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                if future.result():
                    found_ips.append(ip)
                    print(f"✅ Puerto 81 abierto en {ip}")
            except:
                pass
    
    if not found_ips:
        print("❌ No se encontraron dispositivos con puerto 81 abierto")
        return None
    
    # Verificar cuáles son realmente ESP32-CAM
    print("🔍 Verificando si son ESP32-CAM...")
    esp32_ips = []
    
    for ip in found_ips:
        if check_esp32_stream(ip) or check_esp32_control(ip):
            esp32_ips.append(ip)
            print(f"🎯 ESP32-CAM encontrado en {ip}")
    
    return esp32_ips

def main():
    """Función principal."""
    print("🚀 Buscador automático de ESP32-CAM")
    print("=" * 40)
    
    esp32_ips = scan_for_esp32()
    
    if not esp32_ips:
        print("❌ No se encontró ningún ESP32-CAM en la red")
        return None
    
    if len(esp32_ips) == 1:
        ip = esp32_ips[0]
        print(f"🎉 ESP32-CAM encontrado: {ip}")
        print(f"📝 Comando para ejecutar:")
        print(f"python main.py --src http://{ip}:81/stream --esp-base http://{ip}")
        return ip
    else:
        print(f"🔍 Se encontraron {len(esp32_ips)} ESP32-CAM:")
        for i, ip in enumerate(esp32_ips, 1):
            print(f"  {i}. {ip}")
        
        # Usar el primero por defecto
        ip = esp32_ips[0]
        print(f"🎯 Usando el primero: {ip}")
        print(f"📝 Comando para ejecutar:")
        print(f"python main.py --src http://{ip}:81/stream --esp-base http://{ip}")
        return ip

if __name__ == "__main__":
    main()
