import unittest
import time
import concurrent.futures
import requests
from app.model import MovieRecommender

class TestLoad(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.base_url = "http://localhost:8082"
        self.test_users = [12, 72, 100, 150, 200]  # Different user IDs to test
        self.n_recommendations = 5
        
    def test_concurrent_requests(self):
        """Test system performance under concurrent requests"""
        def make_request(user_id, model):
            url = f"{self.base_url}/recommend/{user_id}?model={model}"
            start_time = time.time()
            response = requests.get(url)
            end_time = time.time()
            return {
                'user_id': user_id,
                'model': model,
                'status_code': response.status_code,
                'response_time': end_time - start_time
            }
        
        # Test parameters
        num_requests = 10  # Number of concurrent requests
        models = ['rf', 'knn']
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            # Create tasks for both models and different users
            future_to_params = {
                executor.submit(
                    make_request, 
                    self.test_users[i % len(self.test_users)], 
                    models[i % len(models)]
                ): i for i in range(num_requests)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_params):
                result = future.result()
                results.append(result)
                
        # Analyze results
        successful_requests = [r for r in results if r['status_code'] == 200]
        failed_requests = [r for r in results if r['status_code'] != 200]
        
        # Calculate statistics
        response_times = [r['response_time'] for r in successful_requests]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Print results
        print(f"\nLoad Test Results:")
        print(f"Total Requests: {len(results)}")
        print(f"Successful Requests: {len(successful_requests)}")
        print(f"Failed Requests: {len(failed_requests)}")
        print(f"Average Response Time: {avg_response_time:.2f} seconds")
        print(f"Max Response Time: {max_response_time:.2f} seconds")
        
        # Assertions
        self.assertGreater(len(successful_requests), 0, "No successful requests")
        self.assertLess(avg_response_time, 10.0, "Average response time too high")
        
    def test_sequential_requests(self):
        """Test system performance with sequential requests"""
        results = []
        
        for user_id in self.test_users:
            for model in ['rf', 'knn']:
                url = f"{self.base_url}/recommend/{user_id}?model={model}"
                start_time = time.time()
                response = requests.get(url)
                end_time = time.time()
                
                results.append({
                    'user_id': user_id,
                    'model': model,
                    'status_code': response.status_code,
                    'response_time': end_time - start_time
                })
                
                # Small delay between requests
                time.sleep(0.5)
        
        # Analyze results
        successful_requests = [r for r in results if r['status_code'] == 200]
        failed_requests = [r for r in results if r['status_code'] != 200]
        
        # Calculate statistics
        response_times = [r['response_time'] for r in successful_requests]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Print results
        print(f"\nSequential Test Results:")
        print(f"Total Requests: {len(results)}")
        print(f"Successful Requests: {len(successful_requests)}")
        print(f"Failed Requests: {len(failed_requests)}")
        print(f"Average Response Time: {avg_response_time:.2f} seconds")
        
        # Assertions
        self.assertGreater(len(successful_requests), 0, "No successful requests")
        self.assertLess(avg_response_time, 10.0, "Average response time too high")

if __name__ == '__main__':
    unittest.main() 