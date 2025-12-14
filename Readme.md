pip install virtualenv 
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env

To test the application, you can use the YouTube video link https://www.youtube.com/watch?v=T-D1OfcDW1M. This video offers a high-level introduction to RAG from a trusted source and can help ground the LLM’s responses, reducing the likelihood of hallucinations.

Steps to generate the summary
Input the video URL: Enter the following URL into the input field labeled "YouTube Video URL":
https://www.youtube.com/watch?v=T-D1OfcDW1M
Summarize the video: Click the Summarize Video button. The application will fetch the transcript and generate a summary based on the content of the video.
View the summary: Once the summarization is complete, the generated summary will be displayed in the Video Summary text box.
Example questions
After summarizing the video, you can engage further by asking specific questions:

Question: How does one reduce hallucinations?
This question can’t be answered accurately without context, as the term ‘hallucination’ can refer either to a psychological condition in humans or to the generation of false or misleading outputs by large language models (LLMs). Fortunately, in this case, we have a video transcript that provides the necessary context. To confirm this, simply paste the question into the Ask a Question About the Video input field and click the Ask a Question button.

Question: Which problems does RAG solve, according to the video?
In this case we are asking for information that is specifically contained in the video. In order to obtain a context-aware response, paste the question into the Ask a Question About the Video input field and click the Ask a Question button.