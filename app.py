import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.utils
import networkx as nx

from rag_processor import RAGProcessor
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize RAG processor
rag_processor = RAGProcessor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/snapmap')
def snapmap():
    """Serve the concept map page"""
    return render_template('snapmap.html')

@app.route('/flashcards')
def flashcards():
    """Serve the flashcards page"""
    return render_template('flashcards.html')

@app.route('/qna')
def qna():
    """Serve the Q&A page"""
    return render_template('qna.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Process the PDF
            result = rag_processor.process_pdf(filepath)
            
            if 'error' in result:
                return jsonify(result), 400
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'message': 'PDF processed successfully',
                'data': result
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/concept-map')
def get_concept_map():
    """Get the concept map data"""
    if not rag_processor.concept_map:
        return jsonify({'error': 'No concept map available'}), 404
    
    # Convert to Plotly format for visualization
    G = rag_processor.concept_map
    
    # Create node positions using spring layout
    pos = nx.spring_layout(G)
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node attributes
        attrs = G.nodes[node]
        definition = attrs.get('definition', '')
        importance = attrs.get('importance', 'medium')
        
        node_text.append(f"<b>{node}</b><br>{definition}")
        
        # Color based on importance
        if importance == 'high':
            node_colors.append('#ff6b6b')
        elif importance == 'medium':
            node_colors.append('#4ecdc4')
        else:
            node_colors.append('#45b7d1')
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        relationship = G.edges[edge]['relationship']
        edge_text.append(relationship)
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="top center",
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        hovertext=node_text
    ))
    
    # Update layout
    fig.update_layout(
        title='Concept Map',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return jsonify({
        'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        'nodes': [{ 'id': node, **G.nodes[node] } for node in G.nodes()],
        'edges': [{ 'source': u, 'target': v, **G.edges[u, v] } for u, v in G.edges()]
    })

@app.route('/api/flashcards')
def get_flashcards():
    """Get flashcards data"""
    # Check if flashcards have been processed and stored
    if not hasattr(rag_processor, 'flashcards') or not rag_processor.flashcards:
        return jsonify({'error': 'No flashcards available'}), 404
    
    flashcards = rag_processor.flashcards
    
    return jsonify({
        'flashcards': flashcards
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle Q&A requests"""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    try:
        answer = rag_processor.answer_question(question)
        return jsonify({
            'question': question,
            'answer': answer
        })
    except Exception as e:
        return jsonify({'error': f'Error answering question: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get processing status"""
    has_vectorstore = rag_processor.vectorstore is not None
    has_concept_map = len(rag_processor.concept_map.nodes()) > 0
    
    return jsonify({
        'has_documents': has_vectorstore,
        'has_concept_map': has_concept_map,
        'total_nodes': len(rag_processor.concept_map.nodes()) if has_concept_map else 0,
        'total_edges': len(rag_processor.concept_map.edges()) if has_concept_map else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
