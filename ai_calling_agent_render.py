#!/usr/bin/env python3
"""
Render-Optimized AI Calling Agent
Fixed PortAudio issues for cloud deployment
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from flask import Flask, render_template_string, request, jsonify

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logger = logging.getLogger("ai_calling_agent")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parent
LOG_AUDIO_DIR = ROOT_DIR / "logs" / "audio"
LOG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CALL_METRICS_PATH = ROOT_DIR / "logs" / "call_metrics.json"
ESCALATION_PATH = ROOT_DIR / "logs" / "escalations.json"

load_dotenv(ROOT_DIR / ".env.local")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# IST knowledge base
try:
    from ist_knowledge import ISTDocument, load_ist_corpus, search, build_vector_index
    IST_DOCS: list[ISTDocument] = load_ist_corpus()
    build_vector_index(IST_DOCS)
    RAG_AVAILABLE = True
except Exception as e:
    logger.warning(f"RAG not available: {e}")
    IST_DOCS = []
    RAG_AVAILABLE = False

class MockVoiceActivityDetector:
    """Mock VAD for Render (browser handles this)"""
    
    def __init__(self):
        self.sample_rate = 16000
        
    def process_audio(self, audio_data):
        # Browser handles VAD, just return mock data
        return {"speech_detected": False}

class MockTTS:
    """Mock TTS for Render (browser handles this)"""
    
    def __init__(self):
        self.is_speaking = False
        
    def speak(self, text, callback=None, on_interrupt=None):
        # In production, browser handles TTS
        logger.info(f"TTS would speak: {text[:50]}...")
        if callback:
            callback()
    
    def stop_speaking(self):
        self.is_speaking = False

class AICallingAgent:
    """Render-optimized AI calling agent"""
    
    def __init__(self):
        self.voice_detector = MockVoiceActivityDetector()
        self.tts = MockTTS()
        self.call_active = False
        self.metrics = {
            "stt_times": [],
            "llm_times": [],
            "tts_times": [],
            "e2e_times": [],
            "exchanges": []
        }
        
    def start_call(self):
        """Start voice call"""
        self.call_active = True
        self.reset_metrics()
        return {
            "status": "started",
            "greeting": "Hello! Welcome to IST Admissions. How can I help you today?"
        }
        
    def process_text_input(self, user_text):
        """Process text input (for testing on Render)"""
        if not self.call_active:
            return {"error": "Call not active"}
        
        stt_start = time.time()
        stt_time = 0.01  # Mock STT time
        
        # Check for end call
        if self.should_end_call(user_text):
            self.end_call()
            return {"status": "ended", "message": "Thank you for calling IST Admissions!"}
        
        # LLM with RAG
        llm_start = time.time()
        response = self.get_intelligent_response(user_text)
        llm_time = time.time() - llm_start
        
        # Mock TTS
        tts_time = 1.5
        e2e_time = stt_time + llm_time + tts_time
        
        # Store metrics
        self.add_exchange(user_text, response, stt_time, llm_time, tts_time, e2e_time)
        
        return {
            "status": "response",
            "user_query": user_text,
            "ai_response": response,
            "metrics": {
                "stt_time": stt_time,
                "llm_time": llm_time,
                "tts_time": tts_time,
                "e2e_time": e2e_time
            }
        }
    
    def get_intelligent_response(self, query):
        """Get intelligent response with proper RAG and fallback"""
        query_lower = query.lower()
        
        # First try RAG
        if RAG_AVAILABLE and IST_DOCS:
            try:
                relevant_docs = search(query, IST_DOCS, top_k=3)
                
                if relevant_docs:
                    # Use RAG response
                    context = " ".join([doc.text for doc in relevant_docs[:2]])
                    response = self.generate_rag_response(query, context)
                    
                    # Check if response is meaningful
                    if len(response) > 20 and "i don't have information" not in response.lower():
                        return response
            except Exception as e:
                logger.error(f"RAG error: {e}")
        
        # Check if it's a yes/no question
        if self.is_yes_no_question(query):
            return self.answer_yes_no_question(query)
        
        # Check if it can be answered generally
        if self.can_answer_generally(query):
            return self.generate_general_response(query)
        
        # Escalate to human
        return self.escalate_to_human(query)
    
    def generate_rag_response(self, query, context):
        """Generate response using RAG context"""
        try:
            prompt = f"""Based on the following IST information, answer the question accurately and confidently. 
            Do not make up information. If the information is not in the context, say so clearly.
            
            Context: {context}
            
            Question: {query}
            
            Answer:"""
            
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"RAG response error: {e}")
            return "I apologize, I'm having technical difficulties. Please try again."
    
    def is_yes_no_question(self, query):
        """Check if question can be answered with yes/no"""
        yes_no_indicators = [
            "is", "are", "do", "does", "did", "will", "would", "could", "should",
            "can", "has", "have", "was", "were", "am"
        ]
        
        return any(indicator in query_lower.split()[:3] for indicator in yes_no_indicators)
    
    def answer_yes_no_question(self, query):
        """Answer yes/no questions based on general knowledge"""
        general_knowledge = {
            "admission": "Yes, IST offers admissions in various engineering programs.",
            "fee": "Yes, IST has a fee structure for all programs.",
            "hostel": "Yes, IST provides hostel facilities.",
            "scholarship": "Yes, IST offers scholarships to deserving students.",
        }
        
        for key, answer in general_knowledge.items():
            if key in query_lower:
                return answer
        
        return "I don't have specific information to answer that question with yes or no."
    
    def can_answer_generally(self, query):
        """Check if question can be answered generally"""
        general_topics = [
            "hello", "hi", "help", "thank", "bye", "goodbye",
            "who are you", "what is ist", "institute of space technology"
        ]
        
        return any(topic in query_lower for topic in general_topics)
    
    def generate_general_response(self, query):
        """Generate general response"""
        general_responses = {
            "hello": "Hello! I'm here to help you with IST admissions.",
            "who are you": "I am the IST Admissions AI Assistant.",
            "what is ist": "IST is the Institute of Space Technology, a premier engineering university.",
            "help": "I can help you with admissions, programs, fees, and deadlines.",
        }
        
        for key, response in general_responses.items():
            if key in query_lower:
                return response
        
        return "I'm here to help with IST admissions. Please ask about programs, fees, or admission requirements."
    
    def escalate_to_human(self, query):
        """Escalate to human and collect phone number"""
        escalation_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "status": "pending"
        }
        
        # Save escalation
        try:
            if ESCALATION_PATH.exists():
                with open(ESCALATION_PATH, 'r') as f:
                    escalations = json.load(f)
            else:
                escalations = []
            
            escalations.append(escalation_data)
            
            with open(ESCALATION_PATH, 'w') as f:
                json.dump(escalations, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving escalation: {e}")
        
        return "I don't have information about that. Let me connect you with our admissions office. What's your phone number so they can call you back?"
    
    def should_end_call(self, transcript):
        """Check if user wants to end call"""
        end_phrases = [
            'end call', 'goodbye', 'bye', 'thank you bye', 'that\'s all',
            'no more query', 'finish', 'done', 'exit'
        ]
        
        return any(phrase in transcript.lower() for phrase in end_phrases)
    
    def end_call(self):
        """End call and save metrics"""
        self.call_active = False
        self.save_metrics()
        return {"status": "ended"}
    
    def reset_metrics(self):
        """Reset metrics for new call"""
        self.metrics = {
            "stt_times": [],
            "llm_times": [],
            "tts_times": [],
            "e2e_times": [],
            "exchanges": []
        }
    
    def add_exchange(self, user_query, agent_response, stt_time, llm_time, tts_time, e2e_time):
        """Add exchange to metrics"""
        self.metrics["exchanges"].append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "agent_response": agent_response,
            "stt_time": stt_time,
            "llm_time": llm_time,
            "tts_time": tts_time,
            "e2e_time": e2e_time
        })
        
        self.metrics["stt_times"].append(stt_time)
        self.metrics["llm_times"].append(llm_time)
        self.metrics["tts_times"].append(tts_time)
        self.metrics["e2e_times"].append(e2e_time)
    
    def save_metrics(self):
        """Save call metrics"""
        try:
            if CALL_METRICS_PATH.exists():
                with open(CALL_METRICS_PATH, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            call_summary = {
                "timestamp": datetime.now().isoformat(),
                "duration": sum(self.metrics["e2e_times"]),
                "exchanges": len(self.metrics["exchanges"]),
                "avg_stt": sum(self.metrics["stt_times"]) / len(self.metrics["stt_times"]) if self.metrics["stt_times"] else 0,
                "avg_llm": sum(self.metrics["llm_times"]) / len(self.metrics["llm_times"]) if self.metrics["llm_times"] else 0,
                "avg_tts": sum(self.metrics["tts_times"]) / len(self.metrics["tts_times"]) if self.metrics["tts_times"] else 0,
                "avg_e2e": sum(self.metrics["e2e_times"]) / len(self.metrics["e2e_times"]) if self.metrics["e2e_times"] else 0
            }
            
            all_metrics.append(call_summary)
            
            with open(CALL_METRICS_PATH, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

# Flask Web App
app = Flask(__name__)
agent = AICallingAgent()

@app.route('/')
def home():
    """Main web interface"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IST Admissions AI Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            max-width: 700px;
            width: 90%;
            text-align: center;
            color: #333;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .logo {
            font-size: 3em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        .call-button {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 25px 50px;
            font-size: 1.4em;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            text-transform: uppercase;
            box-shadow: 0 10px 20px rgba(40, 167, 69, 0.3);
            margin: 20px 0;
        }
        .call-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(40, 167, 69, 0.4);
        }
        .call-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }
        .chat-container {
            display: none;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 2px solid #e9ecef;
            text-align: left;
            max-height: 400px;
            overflow-y: auto;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
        }
        .user-message {
            background: #007bff;
            color: white;
            text-align: right;
        }
        .ai-message {
            background: #28a745;
            color: white;
            text-align: left;
        }
        .input-container {
            display: none;
            margin: 20px 0;
        }
        .text-input {
            width: 70%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 1.1em;
        }
        .send-button {
            width: 25%;
            padding: 15px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
        }
        .status {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 2px solid #e9ecef;
            text-align: center;
            min-height: 80px;
        }
        .status-text {
            font-size: 1.2em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🎓 IST</div>
            <div class="subtitle">Institute of Space Technology - AI Admissions Assistant</div>
        </div>
        
        <button id="callButton" class="call-button pulse" onclick="startCall()">
            📞 Start AI Call
        </button>
        
        <div id="status" class="status">
            <div class="status-text">Click "Start AI Call" to begin</div>
        </div>
        
        <div id="chatContainer" class="chat-container"></div>
        
        <div id="inputContainer" class="input-container">
            <input type="text" id="textInput" class="text-input" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
            <button class="send-button" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let callActive = false;
        
        function startCall() {
            if (callActive) {
                alert('Call already active!');
                return;
            }
            
            const button = document.getElementById('callButton');
            const status = document.getElementById('status');
            const chat = document.getElementById('chatContainer');
            const input = document.getElementById('inputContainer');
            
            callActive = true;
            button.disabled = true;
            button.innerHTML = '📞 Call Active';
            button.style.background = '#dc3545';
            button.classList.remove('pulse');
            
            chat.style.display = 'block';
            input.style.display = 'block';
            
            status.innerHTML = '<div class="status-text">🤖 AI Assistant is ready...</div>';
            
            // Start call on server
            fetch('/start-call', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Call started:', data);
                    addMessage('ai', data.greeting);
                    status.innerHTML = '<div class="status-text">💬 Chat Active - Type your questions</div>';
                })
                .catch(error => {
                    console.error('Error starting call:', error);
                    status.innerHTML = '<div class="status-text">❌ Error starting call</div>';
                    resetCall();
                });
        }
        
        function sendMessage() {
            const input = document.getElementById('textInput');
            const message = input.value.trim();
            
            if (!message || !callActive) return;
            
            addMessage('user', message);
            input.value = '';
            
            // Send to server
            fetch('/process-text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: message})
            })
            .then(response => response.json())
            .then(data => {
                console.log('Response:', data);
                if (data.status === 'ended') {
                    addMessage('ai', data.message);
                    endCall();
                } else if (data.status === 'response') {
                    addMessage('ai', data.ai_response);
                } else {
                    addMessage('ai', 'Sorry, I encountered an error.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('ai', 'Sorry, I encountered an error.');
            });
        }
        
        function addMessage(sender, text) {
            const chat = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            chat.appendChild(messageDiv);
            chat.scrollTop = chat.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function endCall() {
            callActive = false;
            const button = document.getElementById('callButton');
            const status = document.getElementById('status');
            
            button.disabled = false;
            button.innerHTML = '📞 Start AI Call';
            button.style.background = 'linear-gradient(45deg, #28a745, #20c997)';
            button.classList.add('pulse');
            
            status.innerHTML = '<div class="status-text">✅ Call Ended - Thank you!</div>';
        }
        
        function resetCall() {
            callActive = false;
            const button = document.getElementById('callButton');
            const status = document.getElementById('status');
            
            button.disabled = false;
            button.innerHTML = '📞 Start AI Call';
            button.style.background = 'linear-gradient(45deg, #28a745, #20c997)';
            button.classList.add('pulse');
            
            status.innerHTML = '<div class="status-text">Click "Start AI Call" to begin</div>';
        }
    </script>
</body>
</html>
    ''')

@app.route('/start-call', methods=['POST'])
def start_call():
    """Start AI call"""
    try:
        result = agent.start_call()
        return jsonify({"status": "success", "greeting": result["greeting"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/process-text', methods=['POST'])
def process_text():
    """Process text input (for Render testing)"""
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        
        result = agent.process_text_input(user_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/metrics')
def metrics():
    """Get system metrics"""
    try:
        if CALL_METRICS_PATH.exists():
            with open(CALL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            return jsonify({"status": "success", "metrics": metrics})
        else:
            return jsonify({"status": "no_metrics"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    print("🚀 IST AI Calling Agent Starting (Render-Optimized)...")
    print("🌐 Web server ready for Render deployment")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
