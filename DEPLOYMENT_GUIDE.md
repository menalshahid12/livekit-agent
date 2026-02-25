# 🚀 AI Calling Agent - Render Deployment Guide

## 📋 Prerequisites

1. **Render Account** - Free tier available
2. **Groq API Key** - Get from https://console.groq.com
3. **GitHub Repository** - Push code to GitHub

## 🛠️ Step-by-Step Deployment

### 1. Prepare Your Code

```bash
# Make sure all files are ready
ls -la
# Should see:
# - ai_calling_agent.py
# - render_requirements.txt
# - render_Dockerfile
# - src/ folder with ist_knowledge.py
# - data/ folder with IST knowledge base
```

### 2. Update Environment Variables

Create `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Push to GitHub

```bash
git add .
git commit -m "AI Calling Agent ready for Render"
git push origin main
```

### 4. Deploy on Render

#### Method 1: Web Service (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ai-calling-agent`
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r render_requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 ai_calling_agent:app`
   - **Instance Type**: `Free`

5. Add Environment Variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `PYTHON_VERSION`: `3.11.0`

#### Method 2: Docker Deployment

1. In Render, click "New" → "Web Service"
2. Select "Docker" as runtime
3. Use `render_Dockerfile`
4. Configure same environment variables

### 5. Configure Health Check

Add health check in Render:
- **Path**: `/`
- **Interval**: 30 seconds
- **Timeout**: 30 seconds

## 🔧 Configuration Details

### Web Service Settings

```yaml
# Render Configuration
Name: ai-calling-agent
URL: https://ai-calling-agent.onrender.com
Region: Oregon (or nearest)
Plan: Free
Instances: 1
Auto-Deploy: Enabled
```

### Environment Variables

```bash
GROQ_API_KEY=gsk_your_key_here
PYTHON_VERSION=3.11.0
PORT=5000
```

## 🧪 Testing Your Deployment

### 1. Basic Health Check

```bash
curl https://ai-calling-agent.onrender.com/
```

### 2. Test API Endpoints

```bash
# Test metrics endpoint
curl https://ai-calling-agent.onrender.com/metrics

# Test call start
curl -X POST https://ai-calling-agent.onrender.com/start-call
```

### 3. Full Voice Call Test

1. Open your deployed URL in browser
2. Click "Start AI Call"
3. Allow microphone access
4. Test with questions:
   - "What are admission requirements?"
   - "How much is the fee?"
   - "When is the deadline?"

## 📊 Monitoring & Logs

### View Logs in Render

1. Go to your service dashboard
2. Click "Logs" tab
3. Monitor real-time logs
4. Check for errors

### Key Metrics to Monitor

- Response times
- Error rates
- Memory usage
- Concurrent users

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. Build Timeout
**Problem**: Build takes more than 3 minutes
**Solution**: 
- Use `render_requirements.txt` (minimal dependencies)
- Optimize Docker layers
- Use Render's paid tier for longer builds

#### 2. Memory Issues
**Problem**: Service crashes due to memory
**Solution**:
- Reduce concurrent workers
- Optimize memory usage
- Monitor memory in logs

#### 3. Audio Issues
**Problem**: Microphone not working
**Solution**:
- Check HTTPS (required for microphone)
- Ensure secure context
- Test in different browsers

#### 4. Groq API Errors
**Problem**: API rate limits
**Solution**:
- Check API key validity
- Monitor usage limits
- Implement retry logic

## 📈 Scaling for Multiple Users

### Free Tier Limitations
- **RAM**: 512MB
- **CPU**: Shared
- **Concurrent Users**: ~5-10
- **Build Time**: 3 minutes

### Scaling Options

#### 1. Upgrade to Starter Plan ($7/month)
- **RAM**: 1GB
- **CPU**: Dedicated
- **Concurrent Users**: ~15-20
- **Build Time**: 10 minutes

#### 2. Load Balancing
```python
# In ai_calling_agent.py
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
# Render automatically handles load balancing
```

#### 3. Optimize for Concurrency
```python
# Use gunicorn with multiple workers
# --workers 2 (for free tier)
# --workers 4 (for starter plan)
```

## 🔒 Security Considerations

### 1. API Key Security
- Never commit API keys to Git
- Use Render environment variables
- Rotate keys regularly

### 2. HTTPS Only
- Render provides SSL certificates
- Microphone requires HTTPS
- All traffic encrypted

### 3. Rate Limiting
```python
# Add rate limiting in Flask
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/start-call')
@limiter.limit("10 per minute")
def start_call():
    # Your code here
```

## 📱 Mobile Compatibility

### Responsive Design
The web interface is mobile-ready:
- Touch-friendly buttons
- Responsive layout
- Works on all modern browsers

### Microphone Access
- iOS Safari: Requires user interaction
- Android Chrome: Works well
- Desktop: All browsers supported

## 🎯 Performance Optimization

### 1. Reduce Latency
```python
# Optimize audio processing
CHUNK_SIZE = 1024  # Smaller chunks = faster processing
SAMPLE_RATE = 16000  # Optimal for speech
```

### 2. Cache Responses
```python
# Add caching for common queries
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(query):
    return get_intelligent_response(query)
```

### 3. Optimize RAG
```python
# Limit search results
relevant_docs = search(query, IST_DOCS, top_k=3)  # Not 10
```

## 📞 Testing Multiple Users

### Load Testing
```bash
# Use Apache Bench for testing
ab -n 100 -c 10 https://ai-calling-agent.onrender.com/
```

### Real User Testing
1. Share URL with 7-8 friends
2. Test simultaneously
3. Monitor performance
4. Check logs for issues

## 🎉 Success Metrics

### Key Performance Indicators
- **Response Time**: <3 seconds
- **Success Rate**: >95%
- **Concurrent Users**: 7-8 (free tier)
- **Uptime**: >99%

### User Satisfaction
- Clear voice quality
- Accurate responses
- No disconnections
- Easy to use

## 🔄 Continuous Deployment

### Auto-Deploy Setup
1. In Render dashboard
2. Enable "Auto-Deploy"
3. Push updates to GitHub
4. Automatic deployment

### Blue-Green Deployment
```bash
# Deploy to staging first
git push origin staging
# Test thoroughly
# Then deploy to production
git push origin main
```

## 📞 Support & Maintenance

### Regular Tasks
1. **Monitor Logs**: Daily check
2. **Update Dependencies**: Weekly
3. **Backup Data**: Regular exports
4. **Performance Review**: Monthly

### Emergency Procedures
1. **Service Down**: Check Render status
2. **High Errors**: Scale up or rollback
3. **API Issues**: Verify Groq status
4. **User Issues**: Check browser console

---

## 🎯 Quick Start Checklist

- [ ] Groq API key ready
- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Environment variables set
- [ ] Service deployed
- [ ] Health check passing
- [ ] Voice call tested
- [ ] Multiple users tested
- [ ] Monitoring configured

**Your AI Calling Agent is now ready for production!** 🚀
