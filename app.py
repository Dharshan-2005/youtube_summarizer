import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse, parse_qs
import re

# Load environment variables 
load_dotenv()
# Get the API key
API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize the model with correct model name
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

# Prompt template for chunking
chunk_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI that summarizes YouTube transcripts."),
    ("user", "Summarize the following transcript segment into concise bullet points:\n\n{chunk}")
])

# Create the partial summarization chain - Updated for newer LangChain
from langchain.chains import LLMChain
map_chain = LLMChain(llm=llm, prompt=chunk_prompt)

# Reduce summary prompt to combine partial summaries
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional summarizer who creates structured reports."),
    ("user", """Combine the following partial summaries into one cohesive final summary with sections:
    - Title/Context
    - Key Insights
    - Final Takeaway
    
    Partial summaries:
    {summaries}
    """)
])

# Create reduce chain
reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if "youtu.be" in url:
        return urlparse(url).path.lstrip("/")
    if "youtube.com" in url:
        query = parse_qs(urlparse(url).query)
        if "v" in query:
            return query["v"][0]
    # Fallback for embedded URLs
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Invalid URL format")

def chunk_text(text, chunk_overlap=200, chunk_size=1000):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size
    )
    return splitter.split_text(text)

# Streamlit app
st.title("YouTube Video Summarizer")
url = st.text_input("Enter the YouTube video URL")

if st.button("Summarize"):
    if url:
        try:
            # Extract video ID
            video_id = extract_video_id(url)
            
            # Get transcript - Using instance method for your API version
            try:
                # Create an instance of YouTubeTranscriptApi
                api = YouTubeTranscriptApi()
                
                # Get list of available transcripts
                transcript_list_obj = api.list(video_id)
                
                # Try to find English transcript first
                transcript = None
                try:
                    transcript = transcript_list_obj.find_transcript(['en'])
                except:
                    try:
                        # Try to find any manually created transcript
                        transcript = transcript_list_obj.find_manually_created_transcript(['en'])
                    except:
                        try:
                            # Try to find any generated transcript
                            transcript = transcript_list_obj.find_generated_transcript(['en'])
                        except:
                            # Get the first available transcript
                            transcripts = list(transcript_list_obj)
                            if transcripts:
                                transcript = transcripts[0]
                            else:
                                raise Exception("No transcripts available")
                
                # Fetch the actual transcript content
                transcript_list = transcript.fetch()
                
            except Exception as e:
                st.error(f"Could not retrieve transcript. Error: {e}")
                st.error("Make sure the video has captions/subtitles available and the URL is correct.")
                st.stop()
            
            # Combine transcript text - Handle FetchedTranscriptSnippet objects
            transcript = ""
            for item in transcript_list:
                try:
                    # Try to access as dictionary first (fallback)
                    text = item['text']
                except (TypeError, KeyError):
                    # Handle FetchedTranscriptSnippet objects
                    try:
                        text = item.text
                    except AttributeError:
                        # Try other common attributes
                        try:
                            text = str(item)
                        except:
                            continue
                transcript += " " + text
            
            # Check if transcript is empty
            if not transcript.strip():
                st.error("No transcript available for this video.")
                st.stop()
            
            st.success(f"Transcript retrieved successfully! Length: {len(transcript)} characters")
            
            # Split into chunks
            chunks = chunk_text(transcript)
            st.info(f"Split transcript into {len(chunks)} chunks")
            
            # Map step - create partial summaries using invoke instead of run
            partial_summaries = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Use invoke instead of deprecated run method
                    result = map_chain.invoke({"chunk": chunk})
                    summary = result["text"] if "text" in result else str(result)
                    partial_summaries.append(summary)
                except Exception as e:
                    st.error(f"Error processing chunk {i+1}: {e}")
                    continue
                progress_bar.progress((i + 1) / len(chunks))
            
            # Reduce step - combine summaries using invoke
            st.info("Combining partial summaries...")
            try:
                result = reduce_chain.invoke({"summaries": "\n\n".join(partial_summaries)})
                final_summary = result["text"] if "text" in result else str(result)
            except Exception as e:
                st.error(f"Error combining summaries: {e}")
                st.stop()
            
            # Display results
            st.subheader("Final Summary")
            st.write(final_summary)
            
        except ValueError as ve:
            st.error(f"URL Error: {ve}")
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("Make sure the video has captions/subtitles available.")
    else:
        st.warning("Please enter a valid YouTube URL")
