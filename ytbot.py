# Import necessary libraries for the YouTube bot
import gradio as gr
import re  # For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from sentence_transformers import SentenceTransformer  # For creating embeddings locally
import numpy as np  # For numerical operations
from sklearn.metrics.pairwise import cosine_similarity  # For similarity calculations
import google.generativeai as genai  # Google Gemini API for LLM
import os  # For environment variables
from dotenv import load_dotenv  # For loading .env file

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

# Configure Gemini API if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully!")
else:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment variables.")
    print("Please create a .env file with your API key.")

# ============================================================================
# VIDEO ID EXTRACTION
# ============================================================================
def get_video_id(url):
    """
    Extract video ID from YouTube URL.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Video ID or None if not found
    """
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None


# ============================================================================
# TRANSCRIPT EXTRACTION
# ============================================================================
def get_transcript(url):
    """
    Extracts the transcript from a YouTube video.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        list: Transcript data or None if not available
    """
    # Extracts the video ID from the URL
    video_id = get_video_id(url)
    
    if not video_id:
        return None

    try:
        # Create a YouTubeTranscriptApi() object
        ytt_api = YouTubeTranscriptApi()
        
        # Fetch the list of available transcripts for the given YouTube video
        transcripts = ytt_api.list(video_id)
        
        transcript = ""
        for t in transcripts:
            # Check if the transcript's language is English
            if t.language_code == 'en':
                if t.is_generated:
                    # If no transcript has been set yet, use the auto-generated one
                    if len(transcript) == 0:
                        transcript = t.fetch()
                else:
                    # If a manually created transcript is found, use it (overrides auto-generated)
                    transcript = t.fetch()
                    break  # Prioritize the manually created transcript, exit the loop
        
        return transcript if transcript else None
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


# ============================================================================
# TRANSCRIPT PROCESSING
# ============================================================================
def process(transcript):
    """
    Process the transcript into a formatted string.
    
    Args:
        transcript (list): Raw transcript data
        
    Returns:
        str: Processed transcript text
    """
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except (KeyError, TypeError):
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt


# ============================================================================
# TEXT CHUNKING
# ============================================================================
def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    """
    Split the transcript into manageable chunks with overlap.
    
    Args:
        processed_transcript (str): The processed transcript text
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    # Split text by newlines to preserve structure
    lines = processed_transcript.split('\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line.split())
        
        if current_size + line_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('\n'.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_lines = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_lines + [line]
            current_size = sum(len(l.split()) for l in current_chunk)
        else:
            current_chunk.append(line)
            current_size += line_size
    
    # Add the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


# ============================================================================
# GEMINI LLM SETUP
# ============================================================================
def setup_gemini_model(model_name=None):
    """
    Initialize and configure the Gemini model.
    
    Args:
        model_name (str): Name of the Gemini model to use
        
    Returns:
        GenerativeModel: Configured Gemini model instance
    """
    if model_name is None:
        model_name = GEMINI_MODEL
    
    # Create and return a Gemini model instance
    model = genai.GenerativeModel(model_name)
    return model


def initialize_gemini_llm():
    """
    Create and return an instance of the Gemini LLM with default configuration.
    
    Returns:
        GenerativeModel: Gemini model instance
    """
    return setup_gemini_model()


# ============================================================================
# EMBEDDING MODEL SETUP
# ============================================================================
def setup_embedding_model():
    """
    Create and return an instance of the embedding model.
    This uses a local SentenceTransformer model (free and offline).
    
    Returns:
        SentenceTransformer: Embedding model instance
    """
    # Load the embedding model from environment variable or use default
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model


# ============================================================================
# VECTOR STORE CREATION
# ============================================================================
def create_vector_store(chunks, embedding_model):
    """
    Create embeddings for text chunks and store them in a simple vector store.
    
    Args:
        chunks (list): List of text chunks
        embedding_model: The embedding model to use
        
    Returns:
        tuple: (embeddings array, chunks list)
    """
    # Generate embeddings for all chunks
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    
    # Return both embeddings and chunks for later retrieval
    return embeddings, chunks


# ============================================================================
# SIMILARITY SEARCH
# ============================================================================
def perform_similarity_search(embeddings, chunks, query, embedding_model, k=3):
    """
    Search for specific queries within the embedded transcript.
    
    Args:
        embeddings (np.array): Embeddings of text chunks
        chunks (list): Original text chunks
        query (str): The text input for the similarity search
        embedding_model: Model to embed the query
        k (int): The number of similar results to return
        
    Returns:
        list: List of most similar text chunks
    """
    # Encode the query
    query_embedding = embedding_model.encode([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return the corresponding chunks
    results = [chunks[i] for i in top_indices]
    return results


# ============================================================================
# SUMMARY PROMPT CREATION
# ============================================================================
def create_summary_prompt():
    """
    Create a prompt template for summarizing a YouTube video transcript.
    
    Returns:
        str: Prompt template for summarization
    """
    # Define the template for the summary prompt
    template = """You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

Instructions:
1. Summarize the transcript in 2-3 concise paragraphs.
2. Ignore any timestamps in your summary.
3. Focus on the spoken content (Text) of the video.
4. Highlight the key points and main ideas.
5. Make the summary easy to understand and engaging.

Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.

Please summarize the following YouTube video transcript:

{transcript}

Summary:"""
    
    return template


# ============================================================================
# SUMMARY GENERATION
# ============================================================================
def generate_summary(llm, prompt_template, transcript):
    """
    Generate a summary using the Gemini LLM.
    
    Args:
        llm: Gemini model instance
        prompt_template (str): Template for the summary prompt
        transcript (str): The transcript to summarize
        
    Returns:
        str: Generated summary
    """
    # Format the prompt with the transcript
    prompt = prompt_template.replace("{transcript}", transcript[:8000])  # Limit to avoid token limits
    
    try:
        # Generate summary using Gemini
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# ============================================================================
# RETRIEVAL FUNCTION
# ============================================================================
def retrieve(query, embeddings, chunks, embedding_model, k=7):
    """
    Retrieve relevant context based on the user's query.

    Args:
        query (str): The user's query string
        embeddings (np.array): Embeddings of text chunks
        chunks (list): Original text chunks
        embedding_model: Model to embed the query
        k (int): The number of most relevant documents to retrieve

    Returns:
        list: A list of the k most relevant text chunks
    """
    relevant_context = perform_similarity_search(
        embeddings, chunks, query, embedding_model, k=k
    )
    return relevant_context


# ============================================================================
# Q&A PROMPT CREATION
# ============================================================================
def create_qa_prompt_template():
    """
    Create a prompt template for question answering based on video content.
    
    Returns:
        str: Prompt template for Q&A
    """
    # Define the template string
    qa_template = """You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
1. Precise and free from repetition
2. Consistent with the information provided in the video
3. Well-organized and easy to understand
4. Focused on addressing the user's question directly

If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.

If the provided context does not contain enough information to answer the question, please state that clearly.

Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.

Relevant Video Context:
{context}

Based on the above context, please answer the following question:
{question}

Answer:"""
    
    return qa_template


# ============================================================================
# ANSWER GENERATION
# ============================================================================
def generate_answer(question, embeddings, chunks, embedding_model, llm, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question (str): The user's question
        embeddings (np.array): Embeddings of text chunks
        chunks (list): Original text chunks
        embedding_model: Model to embed the query
        llm: Gemini model instance
        k (int): The number of relevant documents to retrieve

    Returns:
        str: The generated answer to the user's question
    """
    # Retrieve relevant context
    relevant_context = retrieve(question, embeddings, chunks, embedding_model, k=k)
    
    # Create the Q&A prompt
    qa_template = create_qa_prompt_template()
    
    # Format the context
    context_text = "\n\n".join(relevant_context)
    
    # Format the full prompt
    prompt = qa_template.replace("{context}", context_text).replace("{question}", question)
    
    try:
        # Generate answer using Gemini
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
# Initialize global variables to store state
fetched_transcript = None
processed_transcript = ""
embeddings = None
chunks = []
embedding_model = None


# ============================================================================
# MAIN SUMMARIZATION FUNCTION
# ============================================================================
def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    # Check if API key is configured
    if not GEMINI_API_KEY:
        return """⚠️ Gemini API key not found!

Please follow these steps:
1. Create a .env file in the project directory
2. Add your API key: GEMINI_API_KEY=your_actual_key_here
3. Get a free API key from: https://makersuite.google.com/app/apikey

Or set the environment variable:
export GEMINI_API_KEY=your_actual_key_here"""
    
    if not video_url:
        return "Please provide a valid YouTube URL."
    
    try:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        
        if not fetched_transcript:
            return "Could not fetch transcript. Please ensure:\n1. The video has English captions available\n2. The URL is correct\n3. The video is publicly accessible"
        
        processed_transcript = process(fetched_transcript)
        
        if not processed_transcript:
            return "No transcript available. Please fetch the transcript first."
        
        # Step 1: Initialize Gemini LLM for summarization
        llm = initialize_gemini_llm()

        # Step 2: Create the summary prompt
        summary_prompt = create_summary_prompt()

        # Step 3: Generate the video summary
        summary = generate_summary(llm, summary_prompt, processed_transcript)
        
        return summary
        
    except Exception as e:
        return f"An error occurred: {str(e)}\n\nPlease check your API key and internet connection."


# ============================================================================
# MAIN Q&A FUNCTION
# ============================================================================
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the vector store based on the user's query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript, embeddings, chunks, embedding_model

    # Check if API key is configured
    if not GEMINI_API_KEY:
        return """⚠️ Gemini API key not found!

Please follow these steps:
1. Create a .env file in the project directory
2. Add your API key: GEMINI_API_KEY=your_actual_key_here
3. Get a free API key from: https://makersuite.google.com/app/apikey

Or set the environment variable:
export GEMINI_API_KEY=your_actual_key_here"""

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            try:
                # Fetch and preprocess transcript
                fetched_transcript = get_transcript(video_url)
                
                if not fetched_transcript:
                    return "Could not fetch transcript. Please ensure:\n1. The video has English captions available\n2. The URL is correct\n3. The video is publicly accessible"
                
                processed_transcript = process(fetched_transcript)
            except Exception as e:
                return f"Error fetching transcript: {str(e)}"
        else:
            return "Please provide a valid YouTube URL."

    if not user_question:
        return "Please provide a valid question."

    if processed_transcript and user_question:
        try:
            # Step 1: Chunk the transcript (only for Q&A)
            if not chunks:
                chunks = chunk_transcript(processed_transcript)

            # Step 2: Initialize embedding model (only once)
            if embedding_model is None:
                embedding_model = setup_embedding_model()

            # Step 3: Create embeddings for transcript chunks (only needed for Q&A)
            if embeddings is None:
                embeddings, chunks = create_vector_store(chunks, embedding_model)

            # Step 4: Initialize Gemini LLM for Q&A
            llm = initialize_gemini_llm()

            # Step 5: Generate the answer
            answer = generate_answer(user_question, embeddings, chunks, embedding_model, llm)
            
            return answer
            
        except Exception as e:
            return f"An error occurred: {str(e)}\n\nPlease check your API key and internet connection."
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as interface:

    gr.Markdown(
        """
        # 🎥 YouTube Video Summarizer and Q&A
        ### Powered by Google Gemini API (Free Tier Available)
        
        ---
        
        **📋 Setup Instructions:**
        1. Create a `.env` file in the project directory
        2. Add your Gemini API key: `GEMINI_API_KEY=your_key_here`
        3. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        4. Install requirements: `pip install -r requirements.txt`
        
        **✨ Features:**
        - 📝 Generate comprehensive video summaries
        - ❓ Ask questions about video content
        - 🔍 Smart context retrieval using embeddings
        - 🆓 Completely free with Gemini API
        - 🔐 Secure API key management with environment variables
        
        ---
        """
    )

    # Input field for YouTube URL
    video_url = gr.Textbox(
        label="🔗 YouTube Video URL", 
        placeholder="https://www.youtube.com/watch?v=...",
        info="Enter the URL of the YouTube video you want to analyze"
    )
    
    # Create tabs for different functionalities
    with gr.Tab("📝 Video Summary"):
        gr.Markdown("### Generate a comprehensive summary of the video")
        summarize_btn = gr.Button("🎬 Summarize Video", variant="primary", size="lg")
        summary_output = gr.Textbox(
            label="Video Summary", 
            lines=10,
            placeholder="Your video summary will appear here..."
        )
    
    with gr.Tab("❓ Ask Questions"):
        gr.Markdown("### Ask any question about the video content")
        question_input = gr.Textbox(
            label="Your Question", 
            placeholder="What is the main topic discussed in the video?",
            info="Ask specific questions about the video content"
        )
        question_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
        answer_output = gr.Textbox(
            label="Answer to Your Question", 
            lines=10,
            placeholder="Your answer will appear here..."
        )

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)
    
    gr.Markdown(
        """
        ---
        
        **💡 Tips:**
        - Make sure the video has English captions (auto-generated or manual)
        - For best results, ask specific questions
        - The summary focuses on main points and key takeaways
        - You can ask multiple questions about the same video
        
        **🔧 Troubleshooting:**
        - If you get an API key error, check your .env file
        - If transcript fetch fails, verify the video has captions enabled
        - For rate limit errors, wait a few moments and try again
        
        **🔐 Security:**
        - Your API key is stored securely in the .env file
        - Never share your .env file or commit it to version control
        - The .gitignore file is configured to protect your credentials
        """
    )

# Launch the app with specified server name and port
if __name__ == "__main__":
    print(f"🚀 Starting YouTube Video Summarizer...")
    print(f"📡 Server: {SERVER_NAME}:{SERVER_PORT}")
    print(f"🤖 Model: {GEMINI_MODEL}")
    print(f"📊 Embedding Model: {EMBEDDING_MODEL_NAME}")
    
    interface.launch(server_name=SERVER_NAME, server_port=SERVER_PORT)