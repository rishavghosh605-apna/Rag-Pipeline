import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Check API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_api_key and not anthropic_api_key:
    st.error("⚠️ No API keys found! Please add either OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_count' not in st.session_state:
    st.session_state.document_count = 0
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "OpenAI" if openai_api_key else "Anthropic"

# Main title and description
st.title("📚 RAG Chat Assistant")
st.markdown("Upload PDF documents and chat with them using AI!")

# Sidebar for document upload
with st.sidebar:
    st.header("🤖 Model Selection")
    
    # Model selection
    available_models = []
    if openai_api_key:
        available_models.append("OpenAI (GPT-3.5 + OpenAI Embeddings)")
    if anthropic_api_key:
        available_models.append("Anthropic (Claude + HuggingFace Embeddings)")
    
    if available_models:
        selected_model = st.selectbox(
            "Choose AI Pipeline:",
            available_models,
            help="Select which AI pipeline to use. Each option uses different services for both embeddings and chat."
        )
        st.session_state.selected_model = selected_model.split()[0]  # Get just "OpenAI" or "Anthropic"
        
        # Show pipeline details
        if st.session_state.selected_model == "OpenAI":
            st.success("✅ Using OpenAI for both embeddings and chat")
        else:
            st.success("✅ Using HuggingFace for embeddings, Anthropic for chat")
            st.info("ℹ️ Note: First-time use will download the embedding model (~90MB)")
    else:
        st.error("No API keys configured!")
    
    st.markdown("---")
    st.header("📄 Document Upload")
    st.markdown("---")
    
    # File uploader widget
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload"
    )
    
    # Display upload status
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) selected")
        
        # Display file names
        st.markdown("**Selected files:**")
        for file in uploaded_files:
            st.markdown(f"- {file.name}")
        
        # Process button
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Create a temporary directory
                temp_dir = tempfile.mkdtemp()
                
                try:
                    all_documents = []
                    
                    # Process each uploaded file
                    progress_bar = st.progress(0)
                    for idx, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        # Save uploaded file temporarily
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load PDF
                        try:
                            loader = PyPDFLoader(temp_path)
                            documents = loader.load()
                            all_documents.extend(documents)
                        except Exception as pdf_error:
                            st.error(f"❌ Failed to read PDF '{uploaded_file.name}'")
                            st.error(f"Error details: {str(pdf_error)}")
                            st.info("Please ensure the PDF is not corrupted or password-protected.")
                            continue
                    
                    # Check if any documents were loaded
                    if not all_documents:
                        st.error("❌ No documents were successfully loaded!")
                        st.info("Please check your PDF files and try again.")
                        st.stop()
                    
                    # Split documents into chunks
                    st.info("Splitting documents into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    splits = text_splitter.split_documents(all_documents)
                    
                    # Create embeddings and vector store
                    st.info("Creating embeddings...")
                    
                    try:
                        # Choose embedding provider based on selected model
                        if st.session_state.selected_model == "OpenAI":
                            st.info("Using OpenAI embeddings...")
                            embeddings = OpenAIEmbeddings()
                        else:  # Anthropic selected
                            st.info("Using HuggingFace embeddings (this may take a moment on first run)...")
                            # Using a lightweight model for better performance
                            try:
                                embeddings = HuggingFaceEmbeddings(
                                    model_name="all-MiniLM-L6-v2",
                                    model_kwargs={
                                        'device': 'cpu',
                                        'trust_remote_code': False
                                    },
                                    encode_kwargs={'normalize_embeddings': True}
                                )
                            except Exception as hf_error:
                                if "meta tensor" in str(hf_error).lower():
                                    st.info("Trying alternative embedding model...")
                                    embeddings = HuggingFaceEmbeddings(
                                        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': False}
                                    )
                                else:
                                    raise hf_error
                        
                        # Create vector store
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings
                        )
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "quota" in error_msg and st.session_state.selected_model == "OpenAI":
                            st.error("❌ OpenAI API quota exceeded!")
                            st.info("💡 Try switching to Anthropic pipeline which uses free HuggingFace embeddings.")
                            st.markdown("**Solutions:**")
                            st.markdown("- Switch to Anthropic pipeline in the model selection")
                            st.markdown("- Wait for your OpenAI quota to reset")
                            st.markdown("- Check your usage at: https://platform.openai.com/usage")
                        elif "api" in error_msg and "key" in error_msg:
                            st.error(f"❌ API Key Error: {str(e)}")
                            st.info("Please check that your API key is valid and active.")
                        elif "connection" in error_msg or "network" in error_msg:
                            st.error("❌ Network connection error!")
                            st.info("Please check your internet connection and try again.")
                        else:
                            st.error(f"❌ Error creating embeddings: {str(e)}")
                            st.info("Error type: " + type(e).__name__)
                        st.stop()
                    
                    # Create QA chain with selected model
                    st.info(f"Setting up QA chain with {st.session_state.selected_model}...")
                    
                    # Initialize LLM based on selection
                    try:
                        if st.session_state.selected_model == "OpenAI":
                            llm = ChatOpenAI(
                                model_name="gpt-3.5-turbo",
                                temperature=0
                            )
                        else:  # Anthropic
                            llm = ChatAnthropic(
                                model="claude-instant-1.2",
                                temperature=0
                            )
                    except Exception as llm_error:
                        st.error(f"❌ Failed to initialize {st.session_state.selected_model} model")
                        st.error(f"Error details: {str(llm_error)}")
                        if "api" in str(llm_error).lower():
                            st.info("Please check your API key is valid and has the necessary permissions.")
                        st.stop()
                    
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        return_source_documents=True
                    )
                    
                    # Update document count
                    st.session_state.document_count = len(uploaded_files)
                    
                    st.success(f"✅ Processed {len(uploaded_files)} documents successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    
                finally:
                    # Clean up temporary directory
                    shutil.rmtree(temp_dir)
    else:
        st.info("👆 Please select PDF files to upload")
    
    # Show current status
    st.markdown("---")
    st.markdown("**📊 Status**")
    if st.session_state.vectorstore:
        st.success(f"✅ {st.session_state.document_count} documents loaded")
    else:
        st.warning("⚠️ No documents loaded yet")
    
# Main chat area
st.header("💬 Chat Interface")

# Show different message based on document status
if st.session_state.vectorstore:
    # Chat is ready
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"- Page: {source.get('page', 'N/A')}")
                        st.markdown(f"- Content: {source.get('content', '')[:200]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from QA chain
                    response = st.session_state.qa_chain({"query": prompt})
                    answer = response['result']
                    
                    # Extract source information
                    sources = []
                    if response.get('source_documents'):
                        for doc in response['source_documents']:
                            sources.append({
                                'page': doc.metadata.get('page', 'N/A'),
                                'content': doc.page_content
                            })
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show sources if available
                    if sources:
                        with st.expander("📖 View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"- Page: {source['page']}")
                                st.markdown(f"- Content: {source['content'][:200]}...")
                                st.markdown("---")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = str(e)
                    if "rate limit" in error_msg.lower():
                        st.error("❌ Rate limit exceeded!")
                        st.info("Please wait a moment before trying again.")
                    elif "context length" in error_msg.lower() or "token" in error_msg.lower():
                        st.error("❌ Message too long!")
                        st.info("Try asking a shorter question or reducing the number of documents.")
                    elif "api" in error_msg.lower() and "key" in error_msg.lower():
                        st.error("❌ API authentication error!")
                        st.info("Please check your API key configuration.")
                    else:
                        st.error(f"❌ Error generating response: {error_msg}")
                        st.info(f"Error type: {type(e).__name__}")
                    
else:
    st.info("👈 Please upload PDF documents from the sidebar to start chatting!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("Built with ❤️ using Streamlit")
with col2:
    current_model = st.session_state.selected_model if 'selected_model' in st.session_state else "AI"
    if current_model == "OpenAI":
        st.markdown("Powered by OpenAI (Embeddings + GPT-3.5)")
    else:
        st.markdown("Powered by HuggingFace + Anthropic Claude")
with col3:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() 