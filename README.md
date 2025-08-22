# MindMapMe - AI Learning Assistant

Transform your handwritten PDF notes into interactive learning tools using AI-powered RAG (Retrieval-Augmented Generation) technology.

## Features

- **📄 PDF Processing**: Upload handwritten PDF notes for AI analysis
- **🗺️ Concept Maps**: Visualize relationships between concepts with interactive graphs
- **🃏 Flashcards**: Generate and study with interactive flashcards
- **❓ Q&A Assistant**: Ask questions about your notes and get AI-powered answers
- **📊 Progress Tracking**: Monitor your learning progress with detailed statistics

## Technology Stack

- **Backend**: Python Flask
- **AI/ML**: LangChain Community, GROQ API
- **Vector Database**: ChromaDB
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Plotly.js
- **PDF Processing**: PyPDF2

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- GROQ API key (get one from [https://console.groq.com/](https://console.groq.com/))

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd MindMapMe

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

**Important**: Replace `your_groq_api_key_here` with your actual GROQ API key.

### Step 5: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage Guide

### 1. Upload PDF Notes

1. Open your browser and navigate to `http://localhost:5000`
2. Click "Choose File" or drag and drop your PDF file
3. Wait for the AI to process your notes (this may take a few minutes)

### 2. Explore Concept Maps

1. Click on "Concept Maps" from the dashboard
2. View the interactive graph showing relationships between concepts
3. Hover over nodes to see concept definitions
4. Concepts are color-coded by importance (red=high, yellow=medium, green=low)

### 3. Study with Flashcards

1. Navigate to "Flashcards" from the dashboard
2. Click on cards to flip them and reveal answers
3. Use the "Correct" or "Incorrect" buttons to track your progress
4. Filter cards by category using the category badges
5. Monitor your study statistics in real-time

### 4. Ask Questions

1. Go to "Q&A Assistant" from the dashboard
2. Type your question in the chat interface
3. Use suggested questions for quick access to common queries
4. Get AI-powered answers based on your uploaded notes

## API Endpoints

- `POST /api/upload` - Upload and process PDF files
- `GET /api/concept-map` - Retrieve concept map data
- `GET /api/flashcards` - Get generated flashcards
- `POST /api/ask` - Ask questions about the notes
- `GET /api/status` - Check processing status

## File Structure

```
MindMapMe/
├── app.py                 # Main Flask application
├── rag_processor.py       # RAG processing logic
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/           # HTML templates
│   ├── dashboard.html   # Main dashboard
│   ├── snapmap.html     # Concept map visualization
│   ├── flashcards.html  # Flashcards interface
│   └── qna.html        # Q&A chat interface
├── uploads/            # Temporary file uploads
└── chroma_db/         # Vector database storage
```

## Troubleshooting

### Common Issues

1. **GROQ API Key Error**
   - Ensure your GROQ API key is correctly set in the `.env` file
   - Verify the key is valid and has sufficient credits

2. **PDF Processing Fails**
   - Check that the PDF file is not corrupted
   - Ensure the PDF contains readable text (not just images)
   - Try with a smaller PDF file first

3. **Dependencies Installation Issues**
   - Update pip: `pip install --upgrade pip`
   - Install system dependencies if needed (e.g., `apt-get install python3-dev` on Ubuntu)

4. **Port Already in Use**
   - Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

### Performance Tips

- For large PDFs, processing may take several minutes
- The system works best with handwritten notes that have been digitized
- Ensure good internet connection for GROQ API calls

## Development

### Adding New Features

1. **New Learning Tools**: Add new routes in `app.py` and corresponding templates
2. **Custom AI Models**: Modify `rag_processor.py` to use different LLM providers
3. **Enhanced Visualizations**: Extend the concept map with additional graph layouts

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the error logs in the console
3. Ensure all dependencies are correctly installed
4. Verify your GROQ API key is valid and has credits

## Acknowledgments

- LangChain Community for the RAG framework
- GROQ for the fast LLM API
- ChromaDB for vector storage
- Plotly for interactive visualizations
- Bootstrap for the responsive UI framework
