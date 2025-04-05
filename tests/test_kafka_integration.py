import unittest
import json
import time
from app.kafka_io import enviar_recomendacion_kafka, recibir_recomendaciones_kafka
from app.model import MovieRecommender

class TestKafkaIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recommender = MovieRecommender()
        self.recommender.load_data()
        self.test_user_id = 12
        self.test_n_recommendations = 5
        
    def test_kafka_producer_consumer(self):
        """Test the complete Kafka flow from producer to consumer"""
        # Get recommendations using RF model
        recommendations = self.recommender.get_recommendations(
            self.test_user_id, 
            self.test_n_recommendations, 
            model='rf'
        )
        
        # Send recommendations to Kafka
        message = {
            'user_id': self.test_user_id,
            'recommendations': recommendations,
            'model': 'rf'
        }
        
        # Send message to Kafka
        enviar_recomendacion_kafka(message)
        
        # Give some time for the message to be processed
        time.sleep(2)
        
        # Start consumer and get messages
        received_messages = []
        def message_handler(msg):
            received_messages.append(msg)
            
        # Run consumer for a short time
        recibir_recomendaciones_kafka(message_handler, timeout=5)
        
        # Verify that we received at least one message
        self.assertGreater(len(received_messages), 0)
        
        # Verify message structure
        received_message = received_messages[0]
        self.assertIn('user_id', received_message)
        self.assertIn('recommendations', received_message)
        self.assertIn('model', received_message)
        
        # Verify content
        self.assertEqual(received_message['user_id'], self.test_user_id)
        self.assertEqual(received_message['model'], 'rf')
        self.assertEqual(len(received_message['recommendations']), self.test_n_recommendations)
        
    def test_kafka_message_format(self):
        """Test Kafka message format and serialization"""
        # Create a test message
        test_message = {
            'user_id': 1,
            'recommendations': [
                {
                    'title': 'Test Movie',
                    'genres': 'Action|Adventure',
                    'predicted_rating': 4.5
                }
            ],
            'model': 'rf'
        }
        
        # Verify message can be serialized to JSON
        try:
            json_message = json.dumps(test_message)
            self.assertIsInstance(json_message, str)
        except Exception as e:
            self.fail(f"Message serialization failed: {str(e)}")
            
        # Verify message can be deserialized from JSON
        try:
            deserialized_message = json.loads(json_message)
            self.assertEqual(deserialized_message, test_message)
        except Exception as e:
            self.fail(f"Message deserialization failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 