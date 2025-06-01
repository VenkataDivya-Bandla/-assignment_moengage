from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import subprocess
import json
import os
import sys

app = Flask(__name__)
CORS(app)

# Read the HTML content
def load_html_template():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html><body>
        <h1>Error: index.html not found</h1>
        <p>Please make sure index.html is in the same directory as this Flask app.</p>
        </body></html>
        """

@app.route('/')
def index():
    """Serve the main HTML page"""
    html_content = load_html_template()
    return render_template_string(html_content)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_article():
    """Handle the analysis request"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        # Get the URL from the request
        data = request.get_json()
        print(f"Received data: {data}")  # Debug log
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        
        # Validate URL format (basic check)
        if not url.startswith(('https://', 'http://')):
            return jsonify({'error': 'Please provide a valid URL starting with http:// or https://'}), 400
        
        # Run the analyzer.py script
        try:
            print(f"Running analyzer with URL: {url}")  # Debug log
            
            # Set environment variables for UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'en_US.UTF-8'
            
            # Run analyzer.py with the URL as an argument and proper encoding
            result = subprocess.run([
                sys.executable, 'analyzer.py', url, '--format', 'json'
            ], 
            capture_output=True, 
            text=True, 
            encoding='utf-8',  # Explicitly set encoding to UTF-8
            errors='replace',  # Replace problematic characters instead of failing
            timeout=300,  # 5-minute timeout
            env=env)
            
            print(f"Analyzer return code: {result.returncode}")  # Debug log
            print(f"Analyzer stdout length: {len(result.stdout)}")  # Debug log
            
            if result.returncode != 0:
                print(f"Analyzer error: {result.stderr}")
                return jsonify({
                    'error': 'Analysis failed',
                    'details': result.stderr,
                    'return_code': result.returncode
                }), 500
            
            # Parse the JSON output from analyzer.py
            try:
                # Clean the output - sometimes there might be extra text before/after JSON
                stdout_lines = result.stdout.strip().split('\n')
                json_start = -1
                json_end = -1
                
                # Find the JSON content
                for i, line in enumerate(stdout_lines):
                    if line.strip().startswith('{'):
                        json_start = i
                        break
                
                if json_start == -1:
                    # If no JSON found, try to parse the entire output
                    analysis_result = json.loads(result.stdout)
                else:
                    # Find the end of JSON
                    brace_count = 0
                    for i in range(json_start, len(stdout_lines)):
                        line = stdout_lines[i]
                        brace_count += line.count('{') - line.count('}')
                        if brace_count == 0 and '}' in line:
                            json_end = i
                            break
                    
                    if json_end == -1:
                        json_end = len(stdout_lines) - 1
                    
                    json_content = '\n'.join(stdout_lines[json_start:json_end + 1])
                    analysis_result = json.loads(json_content)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw output: {result.stdout}")
                
                # Try to extract any useful information from the output
                lines = result.stdout.strip().split('\n')
                clean_lines = [line for line in lines if not line.startswith('✅') and not line.startswith('❌')]
                
                return jsonify({
                    'error': 'Failed to parse analysis results as JSON',
                    'details': f'JSON parsing error: {str(e)}',
                    'raw_output': result.stdout[:1000],  # First 1000 chars for debugging
                    'clean_lines': clean_lines[:10]  # First 10 clean lines
                }), 500
            
            return jsonify(analysis_result)
            
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'Analysis timed out. Please try again with a simpler URL.'}), 408
        except FileNotFoundError:
            return jsonify({'error': 'analyzer.py not found. Please ensure it exists in the same directory.'}), 500
        except Exception as e:
            print(f"Subprocess error: {str(e)}")
            return jsonify({'error': f'Failed to run analyzer: {str(e)}'}), 500
    
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Documentation Analyzer API is running'})

if __name__ == '__main__':
    # Check if required files exist
    required_files = ['analyzer.py', 'index.html']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        print("Please ensure all required files are in the same directory as this Flask app.")
    
    print("Starting Documentation Analyzer Server...")
    print("Access the application at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/analyze")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)