# --- app.py (전체 수정본) ---
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, json, base64, uuid
from datetime import datetime
from PIL import Image
from io import BytesIO
from detect_analysis_gpt_module import run_detect_analysis_gpt_advice

app = Flask(__name__)
app.secret_key = 'secret_key'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HOSPITALS = [
    {"name": "병원1", "director": "김철수", "lat": 37.5665, "lon": 126.9780},
    {"name": "병원2", "director": "박영희", "lat": 37.5651, "lon": 126.9895},
    {"name": "병원3", "director": "이민호", "lat": 37.5700, "lon": 126.9820}
]

def load_user_history():
    if not os.path.exists("user_history.json"):
        return [] 
    with open("user_history.json", "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [] 
        except:
            return []

def save_user_history(data):
    with open("user_history.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    session['current_session_id'] = session_id
    return render_template("index.html", session_id=session_id)

@app.route('/upload_drawing', methods=['POST'])
def upload_drawing():
    data_url = request.form.get('drawing')
    name = request.form.get('name', 'drawing')
    header, encoded = data_url.split(',', 1)
    image_data = base64.b64decode(encoded)

    session_id = session.get('current_session_id')
    user_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(user_dir, exist_ok=True)

    original = Image.open(BytesIO(image_data)).convert('RGBA')
    white_bg = Image.new('RGBA', original.size, (255, 255, 255, 255))
    white_bg.paste(original, (0, 0), mask=original)
    final = white_bg.convert('RGB')

    final.save(os.path.join(user_dir, f'{name}.jpg'), 'JPEG')
    return ('', 204)

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    data_url = request.form.get('photo')
    header, encoded = data_url.split(',', 1)
    image_data = base64.b64decode(encoded)

    session_id = session.get('current_session_id')
    user_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(user_dir, exist_ok=True)

    img = Image.open(BytesIO(image_data))
    img.save(os.path.join(user_dir, 'photo.png'), 'PNG')
    return ('', 204)

@app.route('/distance')
def distance():
    return jsonify(HOSPITALS)

@app.route('/result')
def result():
    session_id = session.get('current_session_id')
    user_dir = os.path.join(UPLOAD_FOLDER, session_id)

    htp_analysis, recommendations = run_detect_analysis_gpt_advice(session_id)

    session['cached_analysis'] = htp_analysis
    session['cached_recommendations'] = recommendations
    
    labels = ['House', 'Tree', 'Person1', 'Person2']
    formatted_analysis = ""
    for label, analysis in zip(labels, htp_analysis):
        formatted_analysis += f"[{label}]\n"
        for group in analysis:
            for sentence in group:
                formatted_analysis += f"- {sentence}\n"
        formatted_analysis += "\n"

    return render_template("result.html",
        formatted_analysis=formatted_analysis.strip(),
        recommendations=recommendations)

@app.route('/view_result/<session_id>')
def view_result(session_id):
    user_dir = f'static/uploads/{session_id}'

    history = load_user_history()
    entry = next((h for h in history if h['id'] == session_id), None)
    if entry is None or not entry.get('analysis'):
        return "분석 결과가 없습니다. 먼저 검사를 진행해주세요.", 400

    htp_analysis = entry['analysis']
    recommendations = entry['recommendations']

    image_paths = {
        'house': f'/{user_dir}/house.jpg',
        'tree': f'/{user_dir}/tree.jpg',
        'person1': f'/{user_dir}/person1.jpg',
        'person2': f'/{user_dir}/person2.jpg',
        'photo': f'/{user_dir}/photo.png'
    }

    labels = ['House', 'Tree', 'Person1', 'Person2']
    formatted_analysis = ""
    for label, analysis in zip(labels, htp_analysis):
        flat_sentences = {sentence for group in analysis for sentence in group}
        if flat_sentences:
            formatted_analysis += f"[{label}]\n"
            for sentence in sorted(flat_sentences):  
                formatted_analysis += f"- {sentence}\n"
            formatted_analysis += "\n"

    return render_template("view_result.html",
        image_paths=image_paths,
        formatted_analysis=formatted_analysis.strip(),
        recommendations=recommendations)



@app.route('/booking_done', methods=['POST'])
def booking_done():
    session_id = session.get('current_session_id')
    history = load_user_history()
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    selected_hospital = request.form.get("hospital")
    
    htp_analysis = session.get('cached_analysis')
    recommendations = session.get('cached_recommendations')

    new_entry = {
        "id": session_id,
        "hospital": selected_hospital,
        "datetime": now,
        "read_by_doctor": False,
        "analysis": htp_analysis,
        "recommendations": recommendations
    }
    history.append(new_entry)
    save_user_history(history)

    return render_template("booking_done.html")


@app.route('/cancel_last_booking', methods=['POST'])
def cancel_last_booking():
    session_id = session.get('current_session_id')
    history = load_user_history()
    new_history = [h for h in history if h['id'] != session_id]
    save_user_history(new_history)
    return redirect(url_for('history'))

@app.route('/cancel_booking/<int:index>', methods=['POST'])
def cancel_booking_by_index(index):
    history = load_user_history()
    if 0 <= index < len(history):
        del history[index]
        save_user_history(history)
    return redirect(url_for('history'))


@app.route('/history')
def history():
    history = load_user_history()
    return render_template("history.html", history=history)

@app.route('/hospital')
def hospital():
    reviews_path = os.path.join(os.path.dirname(__file__), 'reviews.json')
    with open(reviews_path, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    return render_template("hospital_list.html", hospitals=HOSPITALS, reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)
