"""
Flask Web Application & Anti-Cheat REST Server
Project: In-Game Cheat Detection System
File: backend/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import get_db_connection, init_db
from backend.prediction import evaluate_telemetry, FEATURE_COLUMNS
from backend.auth import login_required, role_required, authenticate_user, register_user

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.environ.get('SECRET_KEY', 'anti_cheat_super_secret_jwt_2026')

# Initialize DB on start
try:
    init_db()
except Exception as e:
    print(f"Database initialization status: {e}")

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    if session.get('role') == 'moderator':
        return redirect(url_for('moderator_dashboard'))
    return redirect(url_for('player_dashboard'))

# -------------------------------------------------------------
# AUTHENTICATION (POST /login, POST /register, GET /logout)
# -------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid credentials. Try player1 / password123")
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    role = request.form.get('role', 'player').strip()
    
    success, message = register_user(username, password, role)
    if success:
        return redirect(url_for('login', registered='success'))
    return render_template('login.html', register_error=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# -------------------------------------------------------------
# PLAYER DASHBOARD & GAMEPLAY (POST /gameplay)
# -------------------------------------------------------------
@app.route('/player')
@role_required('player')
def player_dashboard():
    user_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT gs.session_id, gs.yaw, gs.pitch, gs.delta_yaw, gs.delta_pitch, gs.accel_yaw, gs.accel_pitch,
               pr.prediction_id, pr.prediction, pr.probability, pr.risk_level, pr.created_at,
               COALESCE(mr.decision, 'Under Review') as decision
        FROM game_sessions gs
        LEFT JOIN predictions pr ON gs.session_id = pr.session_id
        LEFT JOIN moderator_reviews mr ON pr.prediction_id = mr.prediction_id
        WHERE gs.player_id = %s
        ORDER BY gs.created_at DESC
        LIMIT 20
    """, (user_id,))
    sessions_history = cursor.fetchall()
    
    total_sessions = len(sessions_history)
    suspicious_count = sum(1 for s in sessions_history if s.get('prediction') == 'SUSPICIOUS')
    cursor.close()
    conn.close()
    
    return render_template('player_dashboard.html', 
                           sessions=sessions_history,
                           total_sessions=total_sessions,
                           suspicious_count=suspicious_count)

@app.route('/gameplay', methods=['GET', 'POST'])
@role_required('player')
def gameplay():
    if request.method == 'POST':
        try:
            features = [float(request.form[col]) for col in FEATURE_COLUMNS]
        except (KeyError, ValueError) as err:
            return render_template('gameplay.html', error=f"Invalid telemetry payload: {err}")
            
        # Store in Database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO game_sessions (player_id, yaw, pitch, delta_yaw, delta_pitch, accel_yaw, accel_pitch,
                                       target_x, target_y, target_z, position_x, position_y, position_z, sensitivity, new_sequence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (session['user_id'], *features))
        session_id = cursor.lastrowid
        conn.commit()
        
        # Inference
        eval_result = evaluate_telemetry(features)
        prob = eval_result['probability']
        pred_label = eval_result['prediction']
        risk = eval_result['risk_level']
        
        cursor.execute("""
            INSERT INTO predictions (session_id, prediction, probability, risk_level)
            VALUES (%s, %s, %s, %s)
        """, (session_id, pred_label, prob, risk))
        prediction_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        return render_template('prediction.html',
                               prediction=pred_label,
                               probability=prob,
                               risk_level=risk,
                               prediction_id=prediction_id,
                               features=eval_result['metrics'])
                               
    return render_template('gameplay.html')

# -------------------------------------------------------------
# REST INFERENCE ENDPOINT (POST /predict)
# -------------------------------------------------------------
@app.route('/predict', methods=['POST'])
@app.route('/api/telemetry', methods=['POST'])
def predict_endpoint():
    data = request.get_json() if request.is_json else request.form.to_dict()
    if not data:
        return jsonify({"status": "error", "message": "No payload provided"}), 400
        
    try:
        features = [float(data[k]) for k in FEATURE_COLUMNS]
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Missing or invalid feature: {e}"}), 400
        
    result = evaluate_telemetry(features)
    return jsonify({
        "status": "success",
        "verdict": result['prediction'],
        "probability": result['probability'],
        "risk_level": result['risk_level'],
        "alert_threshold": result['alert_threshold'],
        "flagged_for_review": (result['prediction'] == "SUSPICIOUS")
    })

# -------------------------------------------------------------
# SESSIONS API (GET /sessions)
# -------------------------------------------------------------
@app.route('/sessions', methods=['GET'])
@login_required
def get_sessions():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if session.get('role') == 'moderator':
        cursor.execute("""
            SELECT gs.*, pr.prediction, pr.probability, pr.risk_level, u.username
            FROM game_sessions gs
            LEFT JOIN predictions pr ON gs.session_id = pr.session_id
            JOIN users u ON gs.player_id = u.user_id
            ORDER BY gs.created_at DESC LIMIT 50
        """)
    else:
        cursor.execute("""
            SELECT gs.*, pr.prediction, pr.probability, pr.risk_level
            FROM game_sessions gs
            LEFT JOIN predictions pr ON gs.session_id = pr.session_id
            WHERE gs.player_id = %s
            ORDER BY gs.created_at DESC LIMIT 50
        """, (session['user_id'],))
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify({"status": "success", "count": len(data), "sessions": data})

# -------------------------------------------------------------
# FLAGGED SESSIONS (GET /flagged, GET /moderator)
# -------------------------------------------------------------
@app.route('/flagged', methods=['GET'])
@app.route('/moderator', methods=['GET'])
@role_required('moderator')
def moderator_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT pr.prediction_id, pr.session_id, u.username, pr.prediction, pr.probability, pr.risk_level, 
               pr.created_at, COALESCE(mr.decision, 'Under Review') as decision,
               gs.delta_yaw, gs.delta_pitch, gs.accel_yaw, gs.accel_pitch
        FROM predictions pr
        JOIN game_sessions gs ON pr.session_id = gs.session_id
        JOIN users u ON gs.player_id = u.user_id
        LEFT JOIN moderator_reviews mr ON pr.prediction_id = mr.prediction_id
        ORDER BY pr.created_at DESC
    """)
    sessions = cursor.fetchall()
    
    total_scans = len(sessions)
    suspicious_total = sum(1 for s in sessions if s['prediction'] == 'SUSPICIOUS')
    pending_reviews = sum(1 for s in sessions if s['decision'] == 'Under Review')
    confirmed_cheaters = sum(1 for s in sessions if s['decision'] == 'Confirmed Suspicious')
    
    cursor.close()
    conn.close()
    
    if request.path == '/flagged' and request.headers.get('Accept') == 'application/json':
        flagged_list = [s for s in sessions if s['prediction'] == 'SUSPICIOUS']
        return jsonify({"flagged_count": len(flagged_list), "flagged_sessions": flagged_list})
        
    return render_template('moderator_dashboard.html', 
                           sessions=sessions,
                           total_scans=total_scans,
                           suspicious_total=suspicious_total,
                           pending_reviews=pending_reviews,
                           confirmed_cheaters=confirmed_cheaters)

# -------------------------------------------------------------
# MODERATOR REVIEW (POST /review, GET /review/<id>)
# -------------------------------------------------------------
@app.route('/review', methods=['POST'])
@app.route('/review/<int:prediction_id>', methods=['GET', 'POST'])
@role_required('moderator')
def review_session(prediction_id=None):
    if prediction_id is None:
        prediction_id = int(request.form.get('prediction_id') or request.json.get('prediction_id'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    if request.method == 'POST':
        decision = request.form.get('decision') or (request.json and request.json.get('decision'))
        comments = request.form.get('comments') or (request.json and request.json.get('comments', ''))
        moderator_id = session['user_id']
        
        if conn.is_sqlite:
            cursor.execute("""
                INSERT INTO moderator_reviews (prediction_id, moderator_id, decision, comments)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE decision = %s, comments = %s
            """, (prediction_id, moderator_id, decision, comments, decision, comments))
        else:
            cursor.execute("""
                INSERT INTO moderator_reviews (prediction_id, moderator_id, decision, comments)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE decision = VALUES(decision), comments = VALUES(comments)
            """, (prediction_id, moderator_id, decision, comments))
            
        conn.commit()
        cursor.close()
        conn.close()
        
        if request.is_json:
            return jsonify({"status": "success", "message": "Review submitted successfully"})
        return redirect(url_for('moderator_dashboard'))
        
    # GET: fetch details
    cursor.execute("""
        SELECT pr.*, gs.*, u.username, mr.decision, mr.comments, mr.reviewed_at, mu.username as moderator_name
        FROM predictions pr
        JOIN game_sessions gs ON pr.session_id = gs.session_id
        JOIN users u ON gs.player_id = u.user_id
        LEFT JOIN moderator_reviews mr ON pr.prediction_id = mr.prediction_id
        LEFT JOIN users mu ON mr.moderator_id = mu.user_id
        WHERE pr.prediction_id = %s
    """, (prediction_id,))
    session_details = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not session_details:
        return redirect(url_for('moderator_dashboard'))
        
    return render_template('review.html', session=session_details)

# -------------------------------------------------------------
# ANALYTICS (GET /analytics)
# -------------------------------------------------------------
@app.route('/analytics', methods=['GET'])
@role_required('moderator')
def analytics():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT pr.risk_level, pr.prediction, COUNT(*) as count
        FROM predictions pr
        GROUP BY pr.risk_level, pr.prediction
    """)
    breakdown = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*) as total FROM game_sessions")
    total_sessions = cursor.fetchone()['total'] or 0
    
    cursor.execute("SELECT COUNT(*) as cheats FROM predictions WHERE prediction = 'SUSPICIOUS'")
    total_cheats = cursor.fetchone()['cheats'] or 0
    
    cursor.close()
    conn.close()
    
    cheat_rate = round((total_cheats / total_sessions * 100), 2) if total_sessions > 0 else 0
    
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            "total_sessions": total_sessions,
            "total_flagged": total_cheats,
            "cheat_detection_rate": cheat_rate,
            "breakdown": breakdown
        })
        
    return render_template('analytics.html', 
                           breakdown=breakdown,
                           total_sessions=total_sessions,
                           total_cheats=total_cheats,
                           cheat_rate=cheat_rate)

# -------------------------------------------------------------
# DEMO HELPER: LOAD SAMPLE ROW FROM REAL DATASET
# -------------------------------------------------------------
@app.route('/api/sample-row', methods=['GET'])
def get_sample_row():
    """Returns a real sample row from final_dataset.csv for live viva demonstration"""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'final_dataset.csv'))
    target_type = request.args.get('type', 'random') # 'legit', 'cheat', or 'random'
    
    if not os.path.exists(csv_path):
        return jsonify({"status": "error", "message": "Dataset not found"}), 404
        
    df = pd.read_csv(csv_path)
    if target_type == 'legit':
        filtered = df[df['cheat'] == 0]
    elif target_type == 'cheat':
        filtered = df[df['cheat'] == 1]
    else:
        filtered = df
        
    sample = filtered.sample(n=1).iloc[0].to_dict()
    # Cast to standard python types
    clean_sample = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else (float(v) if isinstance(v, (float, np.floating, int, np.integer)) and k != 'new_sequence' else str(v))) for k, v in sample.items()}
    clean_sample['new_sequence'] = int(sample['new_sequence'])
    return jsonify({"status": "success", "row": clean_sample})

if __name__ == '__main__':
    print("Aegis In-Game Cheat Detection Server starting on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
