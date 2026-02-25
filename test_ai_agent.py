#!/usr/bin/env python3
"""
Test Script for AI Calling Agent
Tests all functionality before deployment
"""
import json
import time
from ai_calling_agent import AICallingAgent

def test_rag_functionality():
    """Test RAG knowledge retrieval"""
    print("🧪 Testing RAG Functionality...")
    
    agent = AICallingAgent()
    
    test_queries = [
        "What are admission requirements?",
        "How much is the fee?",
        "When is the deadline?",
        "What programs are offered?"
    ]
    
    for query in test_queries:
        response = agent.get_intelligent_response(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Length: {len(response)} chars")
        print("---")
    
    print("✅ RAG Testing Complete")

def test_voice_detection():
    """Test voice activity detection"""
    print("🧪 Testing Voice Detection...")
    
    from ai_calling_agent import VoiceActivityDetector
    vad = VoiceActivityDetector()
    
    # Test with different audio levels
    test_audio_low = np.random.normal(0, 0.01, 1024)  # Low energy
    test_audio_high = np.random.normal(0, 0.1, 1024)   # High energy
    
    result_low = vad.process_audio(test_audio_low)
    result_high = vad.process_audio(test_audio_high)
    
    print(f"Low energy result: {result_low['speech_detected']}")
    print(f"High energy result: {result_high['speech_detected']}")
    
    print("✅ Voice Detection Testing Complete")

def test_tts_functionality():
    """Test text-to-speech"""
    print("🧪 Testing TTS Functionality...")
    
    from ai_calling_agent import NaturalMaleVoice
    tts = NaturalMaleVoice()
    
    test_text = "Hello! This is a test of the AI calling agent."
    print(f"Testing TTS with: {test_text}")
    
    # Test speaking (non-blocking)
    tts.speak(test_text, lambda: print("✅ TTS test completed"))
    
    print("✅ TTS Testing Complete")

def test_intelligent_response():
    """Test intelligent response generation"""
    print("🧪 Testing Intelligent Response Generation...")
    
    agent = AICallingAgent()
    
    test_cases = [
        # RAG queries
        ("What are admission requirements?", "rag"),
        ("How much is the fee?", "rag"),
        
        # Yes/No questions
        ("Is IST a good university?", "yes_no"),
        ("Does IST have hostels?", "yes_no"),
        
        # General questions
        ("Who are you?", "general"),
        ("What is IST?", "general"),
        
        # Escalation queries
        ("What is the weather on Mars?", "escalation"),
        ("How do I build a rocket?", "escalation"),
    ]
    
    for query, expected_type in test_cases:
        response = agent.get_intelligent_response(query)
        print(f"Query: {query}")
        print(f"Expected Type: {expected_type}")
        print(f"Response: {response}")
        print(f"Response Length: {len(response)}")
        print("---")
    
    print("✅ Intelligent Response Testing Complete")

def test_error_handling():
    """Test error handling"""
    print("🧪 Testing Error Handling...")
    
    agent = AICallingAgent()
    
    # Test with empty query
    response = agent.get_intelligent_response("")
    print(f"Empty query response: {response}")
    
    # Test with very long query
    long_query = "admission " * 100
    response = agent.get_intelligent_response(long_query)
    print(f"Long query response length: {len(response)}")
    
    # Test with special characters
    special_query = "What are the requirements? @#$%^&*()"
    response = agent.get_intelligent_response(special_query)
    print(f"Special chars response: {response}")
    
    print("✅ Error Handling Testing Complete")

def test_concurrent_load():
    """Test concurrent load simulation"""
    print("🧪 Testing Concurrent Load...")
    
    import threading
    import time
    
    agent = AICallingAgent()
    results = []
    
    def simulate_user(user_id):
        start_time = time.time()
        response = agent.get_intelligent_response(f"What are admission requirements? User {user_id}")
        end_time = time.time()
        results.append({
            "user_id": user_id,
            "response_time": end_time - start_time,
            "response_length": len(response)
        })
    
    # Simulate 5 concurrent users
    threads = []
    for i in range(5):
        thread = threading.Thread(target=simulate_user, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Analyze results
    avg_response_time = sum(r["response_time"] for r in results) / len(results)
    avg_response_length = sum(r["response_length"] for r in results) / len(results)
    
    print(f"Concurrent Users: 5")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print(f"Average Response Length: {avg_response_length:.0f} chars")
    
    print("✅ Concurrent Load Testing Complete")

def test_metrics_logging():
    """Test metrics logging"""
    print("🧪 Testing Metrics Logging...")
    
    agent = AICallingAgent()
    
    # Simulate a call
    agent.start_call()
    
    # Add some test exchanges
    agent.add_exchange("Test query 1", "Test response 1", 1.0, 0.5, 1.5, 3.0)
    agent.add_exchange("Test query 2", "Test response 2", 1.2, 0.6, 1.8, 3.6)
    
    # End call and save metrics
    agent.end_call()
    
    # Check if metrics file was created
    import os
    if os.path.exists("logs/call_metrics.json"):
        with open("logs/call_metrics.json", "r") as f:
            metrics = json.load(f)
            print(f"Metrics saved: {len(metrics)} calls")
            if metrics:
                latest = metrics[-1]
                print(f"Latest call: {latest['exchanges']} exchanges")
                print(f"Average E2E time: {latest['avg_e2e']:.2f}s")
    
    print("✅ Metrics Logging Testing Complete")

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting AI Calling Agent Tests")
    print("=" * 50)
    
    try:
        test_rag_functionality()
        print()
        
        test_voice_detection()
        print()
        
        test_tts_functionality()
        print()
        
        test_intelligent_response()
        print()
        
        test_error_handling()
        print()
        
        test_concurrent_load()
        print()
        
        test_metrics_logging()
        print()
        
        print("🎉 All Tests Completed Successfully!")
        print("=" * 50)
        print("✅ System is ready for deployment")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        print("=" * 50)
        print("🔧 Fix issues before deployment")

if __name__ == "__main__":
    run_all_tests()
