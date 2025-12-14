# Import necessary libraries for the YouTube bot
import gradio as gr
import re  # For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # For Gemini LLM and embeddings
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
SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

# Configure Gemini API if key is available
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
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
    # Multiple regex patterns to match different YouTube URL formats
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',  # Standard format
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',  # Short format with parameters
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',  # Embed format
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',  # Old format
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',  # Shorts format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            # Clean up any remaining parameters (e.g., ?si=xxx or &t=xxx)
            video_id = video_id.split('?')[0].split('&')[0]
            return video_id
    
    return None


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
        print(f"❌ Could not extract video ID from URL: {url}")
        return None

    try:
        print(f"🔍 Fetching transcript for video ID: {video_id}")
        
        # Fetch the list of available transcripts for the given YouTube video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        
        # Try to find English transcript (manual first, then auto-generated)
        try:
            # Try to get manual English transcript first
            transcript = transcript_list.find_transcript(['en'])
            print(f"✅ Found English transcript (manual: {not transcript.is_generated})")
            return transcript.fetch()
        except:
            # If manual English not found, try auto-generated
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                print(f"✅ Found auto-generated English transcript")
                return transcript.fetch()
            except:
                print(f"❌ No English transcript found for video {video_id}")
                
                # Try to list all available transcripts for debugging
                try:
                    available = [t.language_code for t in transcript_list]
                    print(f"Available languages: {', '.join(available)}")
                except:
                    pass
                
                return None
        
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
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
    # Check if transcript is None or empty
    if transcript is None:
        return ""
    
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
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


# ============================================================================
# GEMINI LLM SETUP (Using LangChain)
# ============================================================================
def setup_credentials():
    """
    Set up credentials for Gemini API.
    
    Returns:
        tuple: (model_id, api_key)
    """
    # Define the model ID for the Gemini model being used
    model_id = GEMINI_MODEL
    
    # Get API key from environment
    api_key = GEMINI_API_KEY
    
    # Return the model ID and API key for later use
    return model_id, api_key


def define_parameters():
    """
    Define parameters for the Gemini model.
    
    Returns:
        dict: Model parameters
    """
    # Return a dictionary containing the parameters for the Gemini model
    return {
        "temperature": 0.1,  # Low temperature for more deterministic outputs
        "max_output_tokens": 900,  # Maximum tokens to generate
    }


def initialize_gemini_llm(model_id, api_key, parameters):
    """
    Create and return an instance of the ChatGoogleGenerativeAI LLM.
    
    Args:
        model_id (str): Model identifier
        api_key (str): Google API key
        parameters (dict): Model parameters
        
    Returns:
        ChatGoogleGenerativeAI: Gemini LLM instance
    """
    # Create and return an instance of ChatGoogleGenerativeAI with the specified configuration
    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        temperature=parameters.get("temperature", 0.1),
        max_output_tokens=parameters.get("max_output_tokens", 900)
    )


# ============================================================================
# EMBEDDING MODEL SETUP (Using LangChain)
# ============================================================================
def setup_embedding_model(api_key):
    """
    Create and return an instance of GoogleGenerativeAIEmbeddings.
    
    Args:
        api_key (str): Google API key
        
    Returns:
        GoogleGenerativeAIEmbeddings: Embedding model instance
    """
    # Create and return an instance of GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )


# ============================================================================
# FAISS INDEX CREATION (Using LangChain)
# ============================================================================
def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    Args:
        chunks (list): List of text chunks
        embedding_model: The embedding model to use
        
    Returns:
        FAISS: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)


# ============================================================================
# SIMILARITY SEARCH (Using LangChain)
# ============================================================================
def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    
    Args:
        faiss_index: The FAISS index containing embedded text chunks
        query (str): The text input for the similarity search
        k (int): The number of similar results to return (default is 3)
        
    Returns:
        list: List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results


# ============================================================================
# SUMMARY PROMPT CREATION (Using LangChain)
# ============================================================================
def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    Returns:
        PromptTemplate: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt


# ============================================================================
# SUMMARY CHAIN CREATION (Using LangChain)
# ============================================================================
def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
    
    Args:
        llm: Language model instance
        prompt (PromptTemplate): PromptTemplate instance
        verbose (bool): Boolean to enable verbose output (default: True)
        
    Returns:
        LLMChain: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


# ============================================================================
# RETRIEVAL FUNCTION (Using LangChain)
# ============================================================================
def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Args:
        query (str): The user's query string
        faiss_index (FAISS): The FAISS index containing the embedded documents
        k (int): The number of most relevant documents to retrieve (default is 7)

    Returns:
        list: A list of the k most relevant documents (or document chunks)
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context


# ============================================================================
# Q&A PROMPT CREATION (Using LangChain)
# ============================================================================
def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks
    """
    # Define the template string
    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


# ============================================================================
# Q&A CHAIN CREATION (Using LangChain)
# ============================================================================
def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
        prompt_template (PromptTemplate): The prompt template to use
        verbose (bool): Whether to enable verbose output for the chain

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering
    """
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


# ============================================================================
# ANSWER GENERATION (Using LangChain)
# ============================================================================
def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question (str): The user's question
        faiss_index (FAISS): The FAISS index containing the embedded documents
        qa_chain (LLMChain): The question-answering chain to use
        k (int): The number of relevant documents to retrieve

    Returns:
        str: The generated answer to the user's question
    """
    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)

    # Generate answer using the QA chain
    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
# Initialize an empty string to store the processed transcript after fetching and preprocessing
fetched_transcript = None
processed_transcript = ""


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
        
        # Check if transcript was successfully fetched
        if fetched_transcript is None:
            return """❌ Could not fetch transcript. Please ensure:
1. The video has English captions available (auto-generated or manual)
2. The URL is correct and properly formatted
3. The video is publicly accessible
4. Try enabling captions on YouTube first

Example format: https://www.youtube.com/watch?v=VIDEO_ID"""
        
        processed_transcript = process(fetched_transcript)
        
        if not processed_transcript:
            return "⚠️ Transcript was fetched but appears to be empty. Please try another video."
    except Exception as e:
        return f"❌ Error fetching transcript: {str(e)}\n\nPlease check the URL and try again."

    if processed_transcript:
        # Step 1: Set up Gemini credentials
        model_id, api_key = setup_credentials()

        # Step 2: Initialize Gemini LLM for summarization using LangChain
        llm = initialize_gemini_llm(model_id, api_key, define_parameters())

        # Step 3: Create the summary prompt and chain using LangChain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary using LangChain
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."


# ============================================================================
# MAIN Q&A FUNCTION
# ============================================================================
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user's query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
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

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        # Step 1: Chunk the transcript using LangChain
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up Gemini credentials
        model_id, api_key = setup_credentials()

        # Step 3: Initialize Gemini LLM for Q&A using LangChain
        llm = initialize_gemini_llm(model_id, api_key, define_parameters())

        # Step 4: Create FAISS index for transcript chunks using LangChain
        embedding_model = setup_embedding_model(api_key)
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain using LangChain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index and LangChain
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as interface:

    gr.Markdown(
        """
        # 🎥 YouTube Video Summarizer and Q&A
        ### Powered by Google Gemini API + LangChain
        
        ---
        
        **📋 Setup Instructions:**
        1. Create a `.env` file in the project directory
        2. Add your Gemini API key: `GEMINI_API_KEY=your_key_here`
        3. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        4. Install requirements: `pip install -r requirements.txt`
        
        **✨ Features:**
        - 📝 Generate comprehensive video summaries using LangChain
        - ❓ Ask questions about video content with LangChain Q&A chains
        - 🔍 Smart context retrieval using FAISS vector store
        - 🆓 Completely free with Gemini API
        - 🔐 Secure API key management with environment variables
        - 🔗 Full LangChain integration maintained
        
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

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False, visible=False)

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
        
        **🔗 LangChain Components Used:**
        - LLMChain for summarization and Q&A
        - PromptTemplate for structured prompts
        - FAISS for vector storage
        - RecursiveCharacterTextSplitter for text chunking
        - GoogleGenerativeAIEmbeddings for embeddings
        """
    )

# Launch the app with specified server name and port
if __name__ == "__main__":
    print(f"🚀 Starting YouTube Video Summarizer with LangChain...")
    print(f"📡 Server: {SERVER_NAME}:{SERVER_PORT}")
    print(f"🤖 Model: {GEMINI_MODEL}")
    print(f"🔗 Using LangChain for orchestration")
    
    interface.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=True)