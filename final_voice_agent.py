#!/usr/bin/env python3
"""
FINAL VERSION - Voice-Only AI Calling Agent for Render
Complete voice interface with browser audio processing
No sounddevice/portaudio dependencies - works on any cloud platform
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

logger = logging.getLogger("final_voice_agent")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parent
LOG_AUDIO_DIR = ROOT_DIR / "logs" / "audio"
LOG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CALL_METRICS_PATH = ROOT_DIR / "logs" / "call_metrics.json"
ESCALATION_PATH = ROOT_DIR / "logs" / "escalations.json"

load_dotenv(ROOT_DIR / ".env.local")

# Initialize Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    GROQ_AVAILABLE = True
except Exception as e:
    logger.error(f"Groq not available: {e}")
    GROQ_AVAILABLE = False

# IST knowledge base
try:
    from ist_knowledge import ISTDocument, load_ist_corpus, search, build_vector_index
    IST_DOCS: list[ISTDocument] = load_ist_corpus()
    build_vector_index(IST_DOCS)
    RAG_AVAILABLE = True
    logger.info(f"RAG loaded with {len(IST_DOCS)} documents")
except Exception as e:
    logger.warning(f"RAG not available: {e}")
    IST_DOCS = []
    RAG_AVAILABLE = False

class FinalVoiceAgent:
    """Complete voice-only AI agent for production"""
    
    def __init__(self):
        self.call_sessions = {}
        self.escalation_count = 0
        
    def start_call_session(self, session_id):
        """Start new voice call session"""
        self.call_sessions[session_id] = {
            "start_time": time.time(),
            "exchanges": [],
            "active": True,
            "greeting_sent": False
        }
        
        return {
            "status": "success",
            "greeting": "Hello! Welcome to IST Admissions. I'm here to help you with any questions about programs, fees, admission requirements, or deadlines. How can I assist you today?",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_voice_input(self, session_id, transcript):
        """Process voice transcript and return intelligent response"""
        if session_id not in self.call_sessions:
            return {"status": "error", "message": "Session not found"}
        
        session = self.call_sessions[session_id]
        
        if not session["active"]:
            return {"status": "error", "message": "Session not active"}
        
        # Clean transcript
        transcript = transcript.strip()
        if not transcript:
            return {"status": "error", "message": "Empty transcript"}
        
        # Check for end call
        if self.should_end_call(transcript):
            session["active"] = False
            return {
                "status": "ended",
                "message": "Thank you for calling IST Admissions! Your call has been recorded and our team will follow up if needed. Have a great day!",
                "session_id": session_id,
                "duration": time.time() - session["start_time"]
            }
        
        # Generate intelligent response
        response = self.generate_intelligent_response(transcript)
        
        # Store exchange
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user_query": transcript,
            "ai_response": response,
            "processing_time": time.time() - session["start_time"]
        }
        session["exchanges"].append(exchange)
        
        return {
            "status": "response",
            "ai_response": response,
            "session_id": session_id,
            "transcript": transcript,
            "exchange_count": len(session["exchanges"]),
            "confidence": "high" if len(response) > 30 else "medium"
        }
    
    def generate_intelligent_response(self, query):
        """Generate intelligent response with RAG, fallback, and escalation"""
        query_lower = query.lower()
        
        # 1. Try RAG first (highest priority)
        if RAG_AVAILABLE and IST_DOCS:
            try:
                relevant_docs = search(query, IST_DOCS, top_k=3)
                
                if relevant_docs:
                    # Use RAG response
                    context = " ".join([doc.text for doc in relevant_docs[:2]])
                    response = self.generate_rag_response(query, context)
                    
                    # Validate response quality
                    if self.is_good_response(response, query):
                        logger.info(f"RAG response generated for: {query[:50]}...")
                        return response
            except Exception as e:
                logger.error(f"RAG error: {e}")
        
        # 2. Check yes/no questions
        if self.is_yes_no_question(query):
            response = self.answer_yes_no_question(query)
            if response and len(response) > 10:
                return response
        
        # 3. General knowledge questions
        if self.can_answer_generally(query):
            return self.generate_general_response(query)
        
        # 4. Escalation (last resort)
        return self.escalate_to_human(query)
    
    def generate_rag_response(self, query, context):
        """Generate response using RAG context"""
        if not GROQ_AVAILABLE:
            return "I apologize, but my AI processing service is currently unavailable. Please try again later."
        
        try:
            prompt = f"""You are an IST Admissions Assistant. Based on the provided IST information, answer the question accurately, confidently, and professionally.

IMPORTANT RULES:
- Answer ONLY using the provided context
- Do not make up information
- Be confident in your response
- If information is not in context, say clearly that you don't have that specific information
- Provide helpful, complete answers
- Maintain professional tone

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
            return "I apologize, I'm having technical difficulties processing your request. Please try again."
    
    def is_good_response(self, response, query):
        """Check if response is good quality"""
        if not response or len(response) < 20:
            return False
        
        # Check for generic error responses
        error_phrases = [
            "i don't have information",
            "i don't know",
            "technical difficulties",
            "apologize",
            "error"
        ]
        
        response_lower = response.lower()
        return not any(phrase in response_lower for phrase in error_phrases)
    
    def is_yes_no_question(self, query):
        """Check if question can be answered with yes/no"""
        yes_no_indicators = [
            "is", "are", "do", "does", "did", "will", "would", "could", "should",
            "can", "has", "have", "was", "were", "am"
        ]
        
        query_words = query_lower.split()[:5]  # Check first 5 words
        return any(indicator in query_words for indicator in yes_no_indicators)
    
    def answer_yes_no_question(self, query):
        """Answer yes/no questions based on IST knowledge"""
        query_lower = query.lower()
        
        # IST-specific yes/no answers
        ist_knowledge = {
            "admission": "Yes, IST offers admissions in various engineering programs including Electrical, Mechanical, Aerospace, and Materials Engineering.",
            "fee": "Yes, IST has a structured fee system. Undergraduate programs typically cost between PKR 80,000 to 120,000 per semester.",
            "hostel": "Yes, IST provides excellent hostel facilities for both male and female students with all necessary amenities.",
            "scholarship": "Yes, IST offers multiple scholarship programs including need-based and merit-based scholarships.",
            "engineering": "Yes, IST is a premier engineering university specializing in space technology and related fields.",
            "accredited": "Yes, IST is fully accredited by the Higher Education Commission of Pakistan.",
            "semester": "Yes, IST follows a semester system with Fall and Spring semesters each year.",
            "lab": "Yes, IST has state-of-the-art laboratories for all engineering disciplines.",
            "library": "Yes, IST has a well-stocked library with digital and physical resources."
        }
        
        for key, answer in ist_knowledge.items():
            if key in query_lower:
                return answer
        
        return "I can answer that with yes or no, but I need more specific information about what you're asking regarding IST."
    
    def can_answer_generally(self, query):
        """Check if question can be answered generally"""
        query_lower = query.lower()
        
        general_topics = {
            "hello": "Hello! I'm your IST Admissions Assistant. I can help you with information about programs, admission requirements, fees, deadlines, and campus facilities.",
            "hi": "Hi there! I'm here to help you with IST admissions. What would you like to know?",
            "who are you": "I am the IST Admissions AI Assistant, designed to help you with accurate information about the Institute of Space Technology.",
            "what is ist": "IST is the Institute of Space Technology, Pakistan's premier engineering university specializing in aerospace, mechanical, electrical, and materials engineering programs.",
            "help": "I can help you with: admission requirements, program details, fee structure, scholarship information, hostel facilities, and application deadlines. What would you like to know?",
            "programs": "IST offers undergraduate programs in Electrical Engineering, Mechanical Engineering, Aerospace Engineering, and Materials Engineering, along with various postgraduate programs.",
            "contact": "You can contact IST admissions at their main campus in Islamabad or through their official website for detailed information.",
            "location": "IST is located in Islamabad, Pakistan, with a beautiful campus equipped with modern facilities.",
            "established": "IST was established in 2002 and has since become a leading engineering institution in Pakistan."
        }
        
        for key, response in general_topics.items():
            if key in query_lower:
                return response
        
        return False
    
    def generate_general_response(self, query):
        """Generate general response for common queries"""
        return "I'm here to help with IST admissions. Please ask me about specific programs, admission requirements, fee structure, or application deadlines."
    
    def escalate_to_human(self, query):
        """Escalate to human with phone collection"""
        self.escalation_count += 1
        
        escalation_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "status": "pending",
            "escalation_id": f"ESC_{self.escalation_count:03d}"
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
                
            logger.info(f"Escalation saved: {escalation_data['escalation_id']}")
        except Exception as e:
            logger.error(f"Error saving escalation: {e}")
        
        return f"I don't have specific information about that query. Let me connect you with our human admissions team. Please provide your phone number and they'll call you back within 24 hours. Your escalation ID is {escalation_data['escalation_id']}."
    
    def should_end_call(self, transcript):
        """Check if user wants to end call"""
        end_phrases = [
            'end call', 'goodbye', 'bye', 'thank you bye', 'that\'s all',
            'no more query', 'finish', 'done', 'exit', 'thank you and goodbye',
            'nothing else', 'no more questions', 'that\'s it'
        ]
        
        transcript_lower = transcript.lower()
        return any(phrase in transcript_lower for phrase in end_phrases)
    
    def end_call_session(self, session_id):
        """End call session and save comprehensive metrics"""
        if session_id not in self.call_sessions:
            return {"status": "error", "message": "Session not found"}
        
        session = self.call_sessions[session_id]
        session["active"] = False
        session["end_time"] = time.time()
        
        # Save detailed metrics
        self.save_session_metrics(session_id, session)
        
        return {
            "status": "ended",
            "session_id": session_id,
            "duration": session["end_time"] - session["start_time"],
            "exchanges": len(session["exchanges"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_session_metrics(self, session_id, session):
        """Save detailed call session metrics"""
        try:
            if CALL_METRICS_PATH.exists():
                with open(CALL_METRICS_PATH, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            call_summary = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "duration": session.get("end_time", time.time()) - session["start_time"],
                "exchanges": len(session["exchanges"]),
                "queries": [ex["user_query"] for ex in session["exchanges"]],
                "responses": [ex["ai_response"] for ex in session["exchanges"]],
                "processing_times": [ex["processing_time"] for ex in session["exchanges"]],
                "escalations": self.escalation_count
            }
            
            all_metrics.append(call_summary)
            
            with open(CALL_METRICS_PATH, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
            logger.info(f"Session metrics saved: {session_id}")
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

# Flask Web App with Complete Voice Interface
app = Flask(__name__)
voice_agent = FinalVoiceAgent()

@app.route('/')
def home():
    """Complete voice-based web interface"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IST Admissions AI Voice Assistant</title>
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
            max-width: 800px;
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
        .voice-interface {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            border: 2px solid #e9ecef;
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
            min-width: 250px;
        }
        .call-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px rgba(40, 167, 69, 0.4);
        }
        .call-button:disabled {
            background: #dc3545;
            cursor: not-allowed;
            transform: none;
        }
        .call-button.active {
            background: linear-gradient(45deg, #dc3545, #fd7e14);
            animation: pulse 1.5s infinite;
        }
        .status-panel {
            background: #fff;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border: 2px solid #ddd;
            text-align: center;
        }
        .status-text {
            font-size: 1.3em;
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .voice-indicator {
            display: none;
            background: #dc3545;
            color: white;
            padding: 15px 25px;
            border-radius: 30px;
            margin: 15px 0;
            font-weight: bold;
            font-size: 1.1em;
        }
        .voice-indicator.active {
            display: inline-block;
            animation: pulse 1s infinite;
        }
        .transcript-panel {
            background: #fff;
            border: 2px solid #ddd;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            text-align: left;
            max-height: 300px;
            overflow-y: auto;
            display: none;
        }
        .transcript-title {
            font-weight: bold;
            color: #1e3c72;
            margin-bottom: 10px;
        }
        .transcript-text {
            font-family: 'Courier New', monospace;
            color: #333;
            line-height: 1.5;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #f5c6cb;
        }
        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #c3e6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🎓 IST</div>
            <div class="subtitle">Institute of Space Technology - AI Voice Assistant</div>
        </div>
        
        <div class="voice-interface">
            <button id="callButton" class="call-button pulse" onclick="toggleCall()">
                📞 Start Voice Call
            </button>
            
            <div id="voiceIndicator" class="voice-indicator">
                🎤 Listening...
            </div>
            
            <div class="status-panel">
                <div id="statusText" class="status-text">
                    Click "Start Voice Call" to begin
                </div>
            </div>
            
            <div id="transcriptPanel" class="transcript-panel">
                <div class="transcript-title">📝 Live Transcript:</div>
                <div id="transcriptText" class="transcript-text"></div>
            </div>
        </div>
    </div>

    <script>
        let callActive = false;
        let recognition;
        let sessionId = null;
        let isListening = false;
        let callStartTime = null;
        
        // Initialize speech recognition
        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window) {
                recognition = new webkitSpeechRecognition();
            } else if ('SpeechRecognition' in window) {
                recognition = new SpeechRecognition();
            } else {
                showError('Speech recognition not supported. Please use Chrome browser.');
                return false;
            }
            
            // Configure recognition
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            recognition.maxAlternatives = 1;
            
            // Event handlers
            recognition.onresult = handleSpeechResult;
            recognition.onerror = handleSpeechError;
            recognition.onend = handleSpeechEnd;
            recognition.onstart = handleSpeechStart;
            
            return true;
        }
        
        function handleSpeechResult(event) {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                const transcript = result[0].transcript;
                
                if (result.isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }
            
            if (finalTranscript.trim()) {
                processSpeech(finalTranscript.trim());
            }
            
            updateTranscript(finalTranscript + interimTranscript);
        }
        
        function handleSpeechError(event) {
            console.error('Speech recognition error:', event.error);
            
            if (event.error === 'no-speech') {
                // Continue listening
                updateStatus('🎤 Listening... Speak clearly');
            } else if (event.error === 'not-allowed') {
                showError('Microphone access denied. Please allow microphone access.');
                endCall();
            } else {
                showError('Speech recognition error: ' + event.error);
                setTimeout(() => {
                    if (callActive && isListening) {
                        recognition.start();
                    }
                }, 1000);
            }
        }
        
        function handleSpeechEnd() {
            isListening = false;
            document.getElementById('voiceIndicator').classList.remove('active');
            
            // Restart if still in call
            if (callActive) {
                setTimeout(() => {
                    if (callActive && !isListening) {
                        startListening();
                    }
                }, 500);
            }
        }
        
        function handleSpeechStart() {
            isListening = true;
            document.getElementById('voiceIndicator').classList.add('active');
        }
        
        async function toggleCall() {
            if (!callActive) {
                await startCall();
            } else {
                endCall();
            }
        }
        
        async function startCall() {
            const button = document.getElementById('callButton');
            const statusText = document.getElementById('statusText');
            const transcriptPanel = document.getElementById('transcriptPanel');
            
            try {
                // Initialize speech recognition
                if (!initSpeechRecognition()) {
                    return;
                }
                
                // Request microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                // Start call session
                sessionId = 'session_' + Date.now();
                callStartTime = Date.now();
                
                button.disabled = true;
                button.innerHTML = '📞 End Call';
                button.classList.remove('pulse');
                button.classList.add('active');
                
                transcriptPanel.style.display = 'block';
                updateStatus('🤖 Connecting to AI Assistant...');
                
                // Start call on server
                const response = await fetch('/start-call', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId})
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    speakText(data.greeting);
                    updateStatus('🎤 Listening... Speak clearly');
                    setTimeout(() => startListening(), 2000);
                } else {
                    throw new Error(data.message || 'Failed to start call');
                }
                
            } catch (err) {
                console.error('Error starting call:', err);
                showError('Failed to start call: ' + err.message);
                resetCall();
            }
        }
        
        function startListening() {
            if (recognition && !isListening && callActive) {
                try {
                    recognition.start();
                } catch (err) {
                    console.error('Error starting recognition:', err);
                }
            }
        }
        
        function stopListening() {
            if (recognition && isListening) {
                recognition.stop();
            }
        }
        
        async function processSpeech(transcript) {
            if (!callActive || !transcript.trim()) return;
            
            updateStatus('🧠 Processing your question...');
            stopListening();
            
            try {
                const response = await fetch('/process-voice', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        transcript: transcript
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'ended') {
                    speakText(data.message);
                    endCall();
                } else if (data.status === 'response') {
                    speakText(data.ai_response);
                    updateStatus('🎤 Listening... Next question please');
                    setTimeout(() => startListening(), 2000);
                } else {
                    showError('Error processing your request');
                    setTimeout(() => startListening(), 2000);
                }
                
            } catch (err) {
                console.error('Error processing speech:', err);
                showError('Error processing your request');
                setTimeout(() => startListening(), 2000);
            }
        }
        
        function speakText(text) {
            if ('speechSynthesis' in window) {
                // Cancel any ongoing speech
                window.speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(text);
                
                // Configure voice for natural male speech
                utterance.rate = 0.85;
                utterance.pitch = 0.8;
                utterance.volume = 0.9;
                utterance.lang = 'en-US';
                
                // Try to find male voice
                const voices = window.speechSynthesis.getVoices();
                const maleVoice = voices.find(voice => 
                    voice.name.toLowerCase().includes('male') || 
                    voice.name.toLowerCase().includes('david') ||
                    voice.name.toLowerCase().includes('alex')
                );
                
                if (maleVoice) {
                    utterance.voice = maleVoice;
                }
                
                // Event handlers
                utterance.onstart = () => {
                    updateStatus('🔊 AI Assistant Speaking...');
                };
                
                utterance.onend = () => {
                    updateStatus('🎤 Listening... Next question please');
                };
                
                utterance.onerror = (event) => {
                    console.error('Speech synthesis error:', event);
                    updateStatus('❌ Voice synthesis error');
                };
                
                // Speak
                window.speechSynthesis.speak(utterance);
            } else {
                showError('Speech synthesis not supported in this browser');
            }
        }
        
        function endCall() {
            if (!callActive) return;
            
            callActive = false;
            stopListening();
            
            const button = document.getElementById('callButton');
            const statusText = document.getElementById('statusText');
            const voiceIndicator = document.getElementById('voiceIndicator');
            
            button.disabled = false;
            button.innerHTML = '📞 Start Voice Call';
            button.classList.remove('active');
            button.classList.add('pulse');
            
            voiceIndicator.classList.remove('active');
            
            // Cancel any ongoing speech
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
            
            updateStatus('📞 Ending call...');
            
            // End session on server
            fetch('/end-call', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ended') {
                    showSuccess('Call ended successfully. Thank you for calling IST Admissions!');
                    updateStatus('✅ Call Ended - Thank you!');
                } else {
                    showError('Error ending call');
                }
            })
            .catch(err => {
                console.error('Error ending call:', err);
                showError('Error ending call');
            });
        }
        
        function resetCall() {
            callActive = false;
            stopListening();
            
            const button = document.getElementById('callButton');
            const statusText = document.getElementById('statusText');
            const voiceIndicator = document.getElementById('voiceIndicator');
            const transcriptPanel = document.getElementById('transcriptPanel');
            
            button.disabled = false;
            button.innerHTML = '📞 Start Voice Call';
            button.classList.remove('active');
            button.classList.add('pulse');
            
            voiceIndicator.classList.remove('active');
            transcriptPanel.style.display = 'none';
            
            updateStatus('Click "Start Voice Call" to begin');
            
            // Cancel speech
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
        }
        
        function updateStatus(message) {
            const statusText = document.getElementById('statusText');
            statusText.textContent = message;
        }
        
        function updateTranscript(text) {
            const transcriptText = document.getElementById('transcriptText');
            transcriptText.textContent = text;
        }
        
        function showError(message) {
            updateStatus('❌ ' + message);
        }
        
        function showSuccess(message) {
            updateStatus('✅ ' + message);
        }
        
        // Initialize voices when available
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = function() {
                // Voices loaded
            };
        }
        
        // Page load
        window.addEventListener('load', function() {
            updateStatus('🎤 Click "Start Voice Call" to begin');
        });
    </script>
</body>
</html>
    ''')

@app.route('/start-call', methods=['POST'])
def start_call():
    """Start voice call session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', f"session_{int(time.time())}")
        
        result = voice_agent.start_call_session(session_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Start call error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/process-voice', methods=['POST'])
def process_voice():
    """Process voice transcript"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        transcript = data.get('transcript', '')
        
        if not session_id or not transcript:
            return jsonify({"status": "error", "message": "Missing session_id or transcript"})
        
        result = voice_agent.process_voice_input(session_id, transcript)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Process voice error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/end-call', methods=['POST'])
def end_call():
    """End voice call session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({"status": "error", "message": "Missing session_id"})
        
        result = voice_agent.end_call_session(session_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"End call error: {e}")
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
            return jsonify({"status": "no_metrics", "message": "No metrics available"})
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_available": RAG_AVAILABLE,
        "groq_available": GROQ_AVAILABLE,
        "sessions_active": len([s for s in voice_agent.call_sessions.values() if s.get("active", False)])
    })

if __name__ == "__main__":
    print("🚀 FINAL VERSION - IST AI Voice Agent Starting...")
    print("🎤 Complete voice interface ready")
    print("🧠 RAG system:", "✅ Active" if RAG_AVAILABLE else "❌ Inactive")
    print("🤖 Groq AI:", "✅ Active" if GROQ_AVAILABLE else "❌ Inactive")
    print("🌐 Web server starting...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
