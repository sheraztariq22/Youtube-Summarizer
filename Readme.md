# 🎥 YouTube Video Summarizer & Q&A Bot

AI-powered tool to summarize YouTube videos and answer questions about their content using Google Gemini API.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gemini](https://img.shields.io/badge/Gemini-API-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

- 📝 **Automatic Video Summarization** - Generate concise summaries of any YouTube video
- ❓ **Intelligent Q&A** - Ask questions and get accurate answers based on video content
- 🔍 **Smart Context Retrieval** - Uses embeddings for relevant information extraction
- 🆓 **100% Free** - Uses Google Gemini's free tier
- 🔐 **Secure** - API keys stored safely in environment variables
- 🌍 **Works from Pakistan** - No regional restrictions

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup API Key

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Get your free API key: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 3. Run the App

```bash
python app.py
```

### 4. Open in Browser

Navigate to: http://127.0.0.1:7860

---

## 📦 Requirements

```
youtube-transcript-api==1.2.1
google-generativeai==0.3.2
sentence-transformers==2.2.2
scikit-learn==1.3.2
gradio==4.44.1
python-dotenv==1.0.0
```

Full requirements available in `requirements.txt`

---

## 🎯 Usage Examples

### Example 1: Summarize a Video

```
1. Input: https://www.youtube.com/watch?v=dQw4w9WgXcQ
2. Click "Summarize Video"
3. Get: AI-generated summary of the entire video
```

### Example 2: Ask Questions

```
1. Input: https://www.youtube.com/watch?v=dQw4w9WgXcQ
2. Question: "What is the main topic of this video?"
3. Get: Detailed answer based on video content
```

---

## 🏗️ Project Structure

```
youtube-video-summarizer/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── .env                   # API keys (create this)
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
├── SETUP_GUIDE.md        # Detailed setup instructions
└── README.md             # This file
```

---

## 🔧 Configuration Options

Edit your `.env` file to customize:

```env
# Required
GEMINI_API_KEY=your_key_here

# Optional
GEMINI_MODEL=gemini-1.5-flash          # or gemini-1.5-pro
GRADIO_SERVER_PORT=7860                # Change port
GRADIO_SERVER_NAME=0.0.0.0            # Network access
EMBEDDING_MODEL=all-MiniLM-L6-v2      # Embedding model
```

---

## 💡 How It Works

1. **Transcript Extraction**: Fetches video transcript using YouTube API
2. **Text Processing**: Cleans and structures the transcript
3. **Chunking**: Splits text into manageable pieces
4. **Embedding**: Creates vector representations (runs locally)
5. **Similarity Search**: Finds relevant context for questions
6. **LLM Generation**: Uses Gemini to generate summaries/answers

---

## 🆚 Comparison with Original

| Feature | IBM Watson Version | Gemini Version |
|---------|-------------------|----------------|
| API Access | ❌ Not in Pakistan | ✅ Available everywhere |
| Cost | 💰 Paid | 🆓 Free (15 req/min) |
| Setup | Complex | Simple |
| Dependencies | 8 packages | 6 packages |
| Performance | Good | Excellent |

---

## 🐛 Troubleshooting

### API Key Issues
```bash
# Verify .env file exists and contains key
cat .env
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Change port in .env
GRADIO_SERVER_PORT=7861
```

### Transcript Not Available
- Ensure video has English captions
- Try with auto-generated captions enabled
- Check if video is publicly accessible

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

---

## 📊 API Limits (Free Tier)

| Resource | Limit |
|----------|-------|
| Requests per minute | 15 |
| Requests per day | 1,500 |
| Tokens per request | ~30,000 |

Sufficient for personal use and development!

---

## 🔐 Security Notes

- ✅ Never commit `.env` file
- ✅ API keys stored in environment variables
- ✅ `.gitignore` configured to protect credentials
- ✅ No hardcoded secrets in code

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Google Gemini** - Free LLM API
- **Gradio** - Web interface framework
- **Sentence Transformers** - Local embeddings
- **YouTube Transcript API** - Transcript extraction

---

## 📞 Support

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

**Issues?** Check the troubleshooting section or open an issue.

---

## 🌟 Star This Project

If you find this useful, please give it a star! ⭐

---

**Made with ❤️ in Pakistan 🇵🇰**

**Powered by Google Gemini 🤖**