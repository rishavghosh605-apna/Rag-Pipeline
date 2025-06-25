# 📚 RAG Chat Assistant

A simple yet powerful Retrieval-Augmented Generation (RAG) pipeline built with Streamlit that allows you to upload PDF documents and chat with them using OpenAI's GPT models.

## ✨ Features

- 📄 **Multi-PDF Support**: Upload and process multiple PDF documents at once
- 🔍 **Smart Chunking**: Automatically splits documents into optimal chunks for processing
- 💬 **Interactive Chat**: Natural conversation interface with your documents
- 📖 **Source Citations**: See exactly which parts of your documents the AI is referencing
- 🚀 **Easy Setup**: Simple installation process with helpful setup script
- 💾 **Session Memory**: Maintains conversation history during your session

## 🛠️ Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd rag-pipeline
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   This will:
   - Check your Python version
   - Install all dependencies
   - Create a .env file from the template

3. **Add your OpenAI API key**
   - Open the `.env` file
   - Replace `your-openai-api-key-here` with your actual API key

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📖 How to Use

1. **Start the App**: Run `streamlit run app.py` and open the provided URL in your browser

2. **Upload Documents**: 
   - Click on "Browse files" in the sidebar
   - Select one or more PDF files
   - Click "🚀 Process Documents"

3. **Start Chatting**:
   - Once documents are processed, type your question in the chat input
   - The AI will answer based on the content of your uploaded documents
   - Click "View Sources" to see which parts of the documents were used

4. **Clear History**: Use the "Clear Chat History" button to start a fresh conversation

## 🔧 Configuration

You can modify these settings in `app.py`:

- **Model**: Change `gpt-3.5-turbo` to `gpt-4` for more advanced responses
- **Temperature**: Adjust the `temperature` parameter (0-1) for response creativity
- **Chunk Size**: Modify `chunk_size` in the text splitter for different document processing
- **Retrieved Documents**: Change `k` value in search_kwargs to retrieve more/fewer sources

## 📁 Project Structure

```
rag-pipeline/
├── app.py              # Main Streamlit application
├── setup.py            # Setup helper script
├── requirements.txt    # Python dependencies
├── env_example.txt     # Environment variables template
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## 🤔 Troubleshooting

### "OpenAI API key not found!"
- Make sure you've created a `.env` file from `env_example.txt`
- Ensure your API key is correctly added to the `.env` file

### "No module named 'langchain'"
- Run `pip install -r requirements.txt` to install all dependencies

### "Error processing documents"
- Check that your PDFs are not password-protected
- Ensure you have enough OpenAI API credits
- Try with smaller PDF files first

## 💡 Tips

- **Better Results**: Upload documents that are related to each other for more coherent conversations
- **Specific Questions**: Ask specific questions rather than broad ones
- **Multiple Sources**: Upload multiple documents to create a comprehensive knowledge base
- **Cost Management**: Monitor your OpenAI API usage to manage costs

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure
- The app processes documents locally before sending to OpenAI

## 🚀 Future Enhancements

Potential improvements you could make:
- Support for more file formats (TXT, DOCX, etc.)
- Persistent vector storage
- User authentication
- Export chat history
- Fine-tuning options
- Local LLM support

## 📝 License

This project is open source and available under the MIT License.

---

Built with ❤️ using [Streamlit](https://streamlit.io/), [LangChain](https://langchain.com/), and [OpenAI](https://openai.com/) 