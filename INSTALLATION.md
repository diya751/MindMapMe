# MindMapMe Installation Guide

## Quick Start (Automated Setup)

### Option 1: Using the Setup Script (Recommended)

1. **Run the automated setup script:**
   ```bash
   python setup.py
   ```

2. **Follow the prompts and complete the setup**

3. **Edit the .env file** with your GROQ API key

4. **Activate virtual environment and run:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   
   python app.py
   ```

---

## Manual Installation Steps

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **GROQ API key** (get one from [https://console.groq.com/](https://console.groq.com/))

### Step 1: Download and Extract

1. Download the project files
2. Extract to a folder of your choice
3. Open terminal/command prompt in the project directory

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

**Important:** Replace `your_groq_api_key_here` with your actual GROQ API key.

### Step 5: Create Required Directories

```bash
mkdir uploads
mkdir chroma_db
```

### Step 6: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

---

## Getting Your GROQ API Key

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and paste it in your `.env` file

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Error
**Error:** "Python 3.8 or higher is required"
**Solution:** 
- Update Python to version 3.8 or higher
- Download from [python.org](https://python.org)

#### 2. Virtual Environment Issues
**Error:** "Failed to create virtual environment"
**Solution:**
```bash
# Windows
python -m pip install --upgrade pip
python -m venv venv

# macOS/Linux
python3 -m pip install --upgrade pip
python3 -m venv venv
```

#### 3. Dependency Installation Fails
**Error:** "Failed to install dependencies"
**Solution:**
```bash
# Update pip first
pip install --upgrade pip

# Install dependencies with verbose output
pip install -r requirements.txt -v

# If specific packages fail, install them individually
pip install langchain
pip install langchain-community
pip install langchain-groq
# ... etc
```

#### 4. GROQ API Key Error
**Error:** "Invalid API key" or "Authentication failed"
**Solution:**
- Verify your API key is correct in the `.env` file
- Check that you have credits in your GROQ account
- Ensure the key is not expired

#### 5. Port Already in Use
**Error:** "Address already in use"
**Solution:**
- Change the port in `app.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)
  ```
- Or kill the process using port 5000

#### 6. PDF Processing Fails
**Error:** "Processing failed"
**Solution:**
- Ensure the PDF contains readable text (not just images)
- Try with a smaller PDF file first
- Check that the PDF is not corrupted
- Verify the PDF is not password-protected

#### 7. Memory Issues
**Error:** "Out of memory" or slow processing
**Solution:**
- Close other applications to free up memory
- Use a smaller PDF file
- Increase system RAM if possible

---

## System Requirements

### Minimum Requirements
- **RAM:** 4GB
- **Storage:** 2GB free space
- **Internet:** Required for GROQ API calls

### Recommended Requirements
- **RAM:** 8GB or more
- **Storage:** 5GB free space
- **Internet:** Stable broadband connection

---

## Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Ensure Python is added to PATH
- Use backslashes in paths: `venv\Scripts\activate`

### macOS
- Use Terminal
- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Linux (Ubuntu/Debian)
- May need to install additional packages:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip python3-venv
  ```

---

## Verification Steps

After installation, verify everything works:

1. **Check the application starts:**
   ```bash
   python app.py
   ```
   Should show: "Running on http://0.0.0.0:5000"

2. **Access the web interface:**
   - Open browser to `http://localhost:5000`
   - Should see the MindMapMe dashboard

3. **Test PDF upload:**
   - Upload a small PDF file
   - Should see processing status
   - Check that concept map and flashcards are generated

4. **Test Q&A:**
   - Go to Q&A section
   - Ask a question about your uploaded notes
   - Should receive an AI-generated answer

---

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review the console output for error messages
3. Ensure all prerequisites are met
4. Verify your GROQ API key is valid
5. Try with a fresh virtual environment

For additional help, refer to the main README.md file.
