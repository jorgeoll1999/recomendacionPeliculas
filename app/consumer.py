from .kafka_io import recibir_recomendaciones_kafka

def main():
    """
    Script principal para consumir recomendaciones de Kafka
    """
    print("🎬 Iniciando consumidor de recomendaciones...")
    print("Presiona Ctrl+C para detener")
    try:
        recibir_recomendaciones_kafka()
    except KeyboardInterrupt:
        print("\n👋 Deteniendo consumidor...")

if __name__ == "__main__":
    import os
    import sys
    # Agregar el directorio raíz al path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    main() 