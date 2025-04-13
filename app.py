# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import PyPDF2
import json
from werkzeug.utils import secure_filename
from jsonschema import validate
from dotenv import load_dotenv
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool
import yaml
from smolagents import OpenAIServerModel

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['JSON_SCHEMA'] = {
    "type": "object",
    "properties": {
        "personal_details": {"type": "object"},
        "education": {"type": "array"},
        "work_experience": {"type": "array"},
        "skills": {"type": "array"},
        "projects": {"type": "array"},
        "analysis": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "sentiment_score": {"type": "number"},
                "sentiment_breakdown": {
                    "type": "object",
                    "properties": {
                        "Positive": {"type": "number"},
                        "Neutral": {"type": "number"},
                        "Negative": {"type": "number"}
                    }
                },
                "improvements": {"type": "array"}
            },
            "required": ["summary", "sentiment_score"]
        }
    },
    "required": ["personal_details", "education", "work_experience", "skills", "analysis"]
}

# loading LLM prompt
with open("prompts2.yaml", "r") as file:
    prompt_templates = yaml.safe_load(file)

# initializing the agent
cv_agent = CodeAgent(
    tools=[],
    model=OpenAIServerModel(
        api_key=os.getenv("DEEPSEEK_TOKEN"),
        api_base="https://api.deepseek.com/v1",
        model_id="deepseek-chat"
    )
)

def preprocess_text(text):
    return text.strip()

def process_uploaded_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            pdf_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
            return preprocess_text(pdf_text)
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None


def format_to_json(extracted_data):
    try:
        structured_data = {
            "personal_details": extracted_data.get("personal_details", {}),
            "education": extracted_data.get("education", []),
            "work_experience": extracted_data.get("work_experience", []),
            "skills": extracted_data.get("skills", []),
            "certifications": extracted_data.get("certifications", []),
            "languages": extracted_data.get("languages", []),
            "projects": extracted_data.get("projects", []),
            "interests": extracted_data.get("interests", []),
            "keywords": extracted_data.get("keywords", []),
            "summary": extracted_data.get("summary", ""),
            "suitable_job_titles": extracted_data.get("suitable_job_titles", []),
            "location": extracted_data.get("location", {}),
            "sentiment_description": extracted_data.get("sentiment_description", ""),
            "sentiment_score": extracted_data.get("sentiment_score", 0),
            "sentiment_analysis": extracted_data.get("sentiment_analysis", {}),
            "Improvements": extracted_data.get("Improvements", [])
        }
        validate(instance=structured_data, schema=app.config['JSON_SCHEMA'])
        return structured_data
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return None

def save_json_output(data, filename):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.json")
    
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return output_path
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")
        return None

def clean_and_validate_json(raw_response: str) -> dict:
    """Clean and validate JSON response from LLM."""
    try:
        # removing the markdown code blocks and extra formatting
        clean_json = raw_response.replace('```json', '').replace('```', '').strip()
        clean_json = clean_json.replace('\\]', ']').replace('\\[', '[')
        clean_json = clean_json.replace('\\"', '"').replace("'", '"')
        parsed = json.loads(clean_json)
        
        # post-processing, converting any field expected to be a string that is found as a list into a single string
        def fix_field(val):
            if isinstance(val, list):
                # Join the list items with a space separator
                return ' '.join(str(item) for item in val)
            return val

        # listing of keys that should be strings (adjusted based on the JSON schema)
        expected_string_keys = ["summary", "sentiment_description"]
        for key in expected_string_keys:
            if key in parsed and not isinstance(parsed[key], str):
                parsed[key] = fix_field(parsed[key])
                
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic JSON: {clean_json}")
        raise ValueError("Invalid JSON response from AI model")

def parse_llm_output(raw_response: str) -> dict:
    """Convert structured text response to JSON"""
    result = {
        "personal_details": {},
        "education": [],
        "work_experience": [],
        "skills": [],
        "projects": [],
        "analysis": {
            "summary": "",
            "sentiment_score": 0.5,
            "sentiment_breakdown": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "improvements": []
        }
    }

    current_section = None
    lines = raw_response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # handling section headers
        if line.startswith('###'):
            current_section = line.strip('#').strip().lower().replace(' ', '_')
            continue
            
        # parsing sections
        try:
            if current_section == 'personal_details':
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    result['personal_details'][key.strip('- ').lower()] = value.strip()
                    
            elif current_section == 'education':
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    result['education'].append({
                        "institution": parts[0].strip('- '),
                        "degree": parts[1] if len(parts) > 1 else "",
                        "dates": parts[2] if len(parts) > 2 else "",
                        "details": parts[3] if len(parts) > 3 else ""
                    })
                    
            elif current_section == 'work_experience':
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    result['work_experience'].append({
                        "company": parts[0].strip('- '),
                        "role": parts[1] if len(parts) > 1 else "",
                        "duration": parts[2] if len(parts) > 2 else "",
                        "description": parts[3] if len(parts) > 3 else ""
                    })
                    
            elif current_section == 'skills':
                if ': ' in line:
                    category, skills = line.split(': ', 1)
                    result['skills'].append({
                        "category": category.strip('- '),
                        "items": [s.strip() for s in skills.split(',')]
                    })
                    
            elif current_section == 'projects':
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    result['projects'].append({
                        "name": parts[0].strip('- '),
                        "description": parts[1] if len(parts) > 1 else "",
                        "technologies": parts[2].split(', ') if len(parts) > 2 else [],
                        "link": parts[3] if len(parts) > 3 else ""
                    })
                    
            elif current_section == 'analysis':
                if line.startswith('Professional Summary:'):
                    result['analysis']['summary'] = line.split(': ', 1)[1].strip()
                elif line.startswith('Sentiment Score:'):
                    try:
                        result['analysis']['sentiment_score'] = float(line.split(': ')[1].split(' ')[0])
                    except:
                        pass
                elif line.startswith('Positive:'):
                    result['analysis']['sentiment_breakdown']['Positive'] = float(line.split(': ')[1].strip('%'))
                elif line.startswith('Neutral:'):
                    result['analysis']['sentiment_breakdown']['Neutral'] = float(line.split(': ')[1].strip('%'))
                elif line.startswith('Negative:'):
                    result['analysis']['sentiment_breakdown']['Negative'] = float(line.split(': ')[1].strip('%'))
                elif line.startswith('- '):
                    result['analysis']['improvements'].append(line.strip('- '))
        except Exception as e:
            print(f"Error parsing line: {line}\n{str(e)}")
            continue
    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # getting the data from the form
    summary = request.form.get('summary', '')
    experience = request.form.get('experience', '')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        pdf_content = process_uploaded_pdf(save_path)
        try:
            formatted_prompt = prompt_templates['cv_extraction'].format(
                cv_text=pdf_content,
                summary_text=summary,
                experience_text=experience
            )
            
            raw_response = cv_agent.run(formatted_prompt)
            # parsing the llm reponse:
            parsed_data = parse_llm_output(raw_response)
            structured_data = {
                "personal_details": parsed_data.get("personal_details", {}),
                "education": parsed_data.get("education", []),
                "work_experience": parsed_data.get("work_experience", []),
                "skills": parsed_data.get("skills", []),
                "projects": parsed_data.get("projects", []),
                "analysis": parsed_data.get("analysis", {})
            }
            # validating the schema
            validate(instance=structured_data, schema=app.config['JSON_SCHEMA'])
            if not structured_data:
                raise ValueError("JSON validation failed")
            
            # saving output
            json_path = save_json_output(structured_data, filename)
            
            return render_template('results.html', 
                                    data=raw_response,
                                    json_path=json_path)
        
        except Exception as e:
            return render_template('error.html', message=str(e))
    
    return redirect(request.url)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)