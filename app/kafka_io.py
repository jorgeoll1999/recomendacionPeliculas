from confluent_kafka import Producer, Consumer
import json

def enviar_recomendacion_kafka(user_id, recomendaciones):
    producer_conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(producer_conf)

    mensaje = {
        "user_id": user_id,
        "recomendaciones": recomendaciones
    }

    print(f"📤 Enviando mensaje: {mensaje}")
    producer.produce("recomendaciones", key=str(user_id), value=json.dumps(mensaje))
    producer.flush()

def recibir_recomendaciones_kafka():
    consumer_conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'recomendador-grupo',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe(["recomendaciones"])

    print("📥 Escuchando recomendaciones desde Kafka...")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("❌ Error:", msg.error())
            else:
                print("✅ Mensaje recibido:", msg.value().decode('utf-8'))
    finally:
        consumer.close()
