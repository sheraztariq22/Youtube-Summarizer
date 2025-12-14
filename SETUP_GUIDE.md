# 🎥 YouTube Video Summarizer - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.11** (recommended) or Python 3.8+
- **Internet connection** (for downloading models and API access)
- **Google account** (for free Gemini API key)

---

## Installation Steps

### Step 1: Clone or Download the Project

```bash
cd your-project-directory
```

### Step 2: Create Virtual Environment (Recommended)

**For Linux/macOS:**
```bash
python3.11 -m venv my_env
source my_env/bin/activate
```

**For Windows:**
```bash
python -m venv my_env
my_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- ✅ `youtube-transcript-api` - For fetching video transcripts
- ✅ `google-generativeai` - Google Gemini API client
- ✅ `sentence-transformers` - For local embeddings
- ✅ `scikit-learn` - For similarity calculations
- ✅ `gradio` - For the web interface
- ✅ `python-dotenv` - For environment variable management

---

## Configuration

### Step 1: Get Your Free Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key

### Step 2: Create Environment File

Create a file named `.env` in your project root directory:

```bash
# Create .env file
touch .env  # Linux/macOS
# or
type nul > .env  # Windows
```

### Step 3: Add Your API Key

Open the `.env` file and add:

```env
# Required: Your Gemini API Key
GEMINI_API_KEY=AIzaSyC...your_actual_key_here

# Optional: Customize these settings
GEMINI_MODEL=gemini-1.5-flash
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Step 4: Verify .gitignore

Make sure your `.gitignore` file includes `.env` to protect your API key:

```gitignore
.env
.env.local
```

---

## Running the Application

### Start the Application

```bash
python app.py
```

You should see:
```
✅ Gemini API configured successfully!
🚀 Starting YouTube Video Summarizer...
📡 Server: 0.0.0.0:7860
🤖 Model: gemini-1.5-flash
📊 Embedding Model: all-MiniLM-L6-v2

Running on local URL:  http://127.0.0.1:7860
```

### Access the Application

Open your browser and navigate to:
- **Local:** http://127.0.0.1:7860
- **Network:** http://YOUR_IP:7860 (accessible from other devices on your network)

---

## Using the Application

### 1. Summarize a Video

1. Paste a YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
2. Click **"🎬 Summarize Video"**
3. Wait for the summary to generate (10-30 seconds)

### 2. Ask Questions

1. Enter the same YouTube URL
2. Type your question in the question box
3. Click **"🔍 Get Answer"**
4. Get an AI-powered answer based on the video content

---

## Troubleshooting

### ❌ Error: "GEMINI_API_KEY not found"

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Verify .env file content
cat .env

# Make sure it contains:
GEMINI_API_KEY=your_actual_key_here
```

### ❌ Error: "Could not fetch transcript"

**Possible causes:**
1. Video doesn't have English captions
2. Invalid YouTube URL
3. Private/restricted video

**Solution:**
- Try a different video with captions
- Check if captions are available by clicking CC button on YouTube

### ❌ Error: "Module not found"

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install google-generativeai sentence-transformers gradio python-dotenv
```

### ❌ Port Already in Use

**Solution:**
```bash
# Change port in .env file
GRADIO_SERVER_PORT=7861

# Or kill the process using the port (Linux/macOS)
lsof -ti:7860 | xargs kill -9
```

### ❌ API Rate Limit Error

**Solution:**
- Wait 60 seconds before trying again
- Gemini free tier: 15 requests/minute, 1500/day
- Consider upgrading to paid tier for higher limits

### ❌ Slow Performance / Model Download

**First run will download models:**
- `all-MiniLM-L6-v2` (~80MB)
- This happens only once
- Models are cached locally

---

## Advanced Configuration

### Using Different Models

Edit `.env` file:

```env
# Faster but less accurate
GEMINI_MODEL=gemini-1.5-flash

# More accurate but slower
GEMINI_MODEL=gemini-1.5-pro
```

### Custom Server Configuration

```env
# Run on specific port
GRADIO_SERVER_PORT=8080

# Local access only
GRADIO_SERVER_NAME=127.0.0.1

# Network access
GRADIO_SERVER_NAME=0.0.0.0
```

### Using Different Embedding Models

```env
# Smaller, faster
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Larger, more accurate
EMBEDDING_MODEL=all-mpnet-base-v2
```

---

## File Structure

```
youtube-video-summarizer/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                   # Your API keys (DO NOT COMMIT)
├── .env.example          # Template for .env file
├── .gitignore            # Protects sensitive files
├── SETUP_GUIDE.md        # This file
└── README.md             # Project documentation
```

---

## Security Best Practices

1. ✅ **Never commit `.env` file** to Git
2. ✅ **Add `.env` to `.gitignore`**
3. ✅ **Don't share your API key publicly**
4. ✅ **Regenerate API key if exposed**
5. ✅ **Use environment variables in production**

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Verify all dependencies are installed
3. Ensure your API key is valid
4. Check the console for error messages
5. Test with a simple YouTube video first

---

## Support

- **Gemini API Documentation:** https://ai.google.dev/docs
- **Gradio Documentation:** https://gradio.app/docs
- **YouTube Transcript API:** https://github.com/jdepoix/youtube-transcript-api

---

## License

This project uses:
- Google Gemini API (subject to Google's terms)
- Free and open-source Python libraries

---

**🎉 Congratulations! You're ready to use the YouTube Video Summarizer!**