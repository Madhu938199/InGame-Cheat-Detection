"""
Stage 10: Comprehensive Test Suite Using Real Dataset Rows
Project: In-Game Cheat Detection System
File: tests/test_system.py
"""

import os
import sys
import unittest
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from backend.app import app
from backend.database import get_db_connection, init_db
from backend.prediction import FEATURE_COLUMNS, evaluate_telemetry

class TestAntiCheatSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test_secret_key'
        cls.client = app.test_client()
        init_db()

        # Load real dataset rows for authentic telemetry testing
        dataset_path = os.path.join(BASE_DIR, 'dataset', 'final_dataset.csv')
        cls.df = pd.read_csv(dataset_path)
        cls.legit_row = cls.df[cls.df['cheat'] == 0].iloc[0].to_dict()
        cls.cheat_row = cls.df[cls.df['cheat'] == 1].iloc[0].to_dict()

    def test_01_valid_player_login(self):
        """Test successful authentication for seeded player account"""
        response = self.client.post('/login', data={
            'username': 'player1',
            'password': 'password123'
        }, follow_redirects=False)
        self.assertIn(response.status_code, [200, 302])
        with self.client.session_transaction() as sess:
            self.assertEqual(sess.get('username'), 'player1')
            self.assertEqual(sess.get('role'), 'player')

    def test_02_invalid_login_credentials(self):
        """Test authentication rejection on invalid credentials"""
        response = self.client.post('/login', data={
            'username': 'player1',
            'password': 'wrongpassword'
        }, follow_redirects=True)
        self.assertIn(b"Invalid", response.data)

    def test_03_role_authorization_protection(self):
        """Verify players cannot access restricted moderator endpoints"""
        # Login as player
        self.client.post('/login', data={'username': 'player1', 'password': 'password123'})
        response = self.client.get('/moderator', follow_redirects=False)
        # Should redirect or reject
        self.assertIn(response.status_code, [302, 403])

    def test_04_prediction_with_real_legitimate_row(self):
        """Evaluate ML prediction on an authentic legitimate row from final_dataset.csv"""
        payload = {k: float(self.legit_row[k]) for k in FEATURE_COLUMNS}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('probability', data)
        self.assertIn('verdict', data)
        self.assertIn('risk_level', data)
        print(f"\n[Test] Real Legit Row -> Probability: {data['probability']}%, Risk: {data['risk_level']}, Verdict: {data['verdict']}")

    def test_05_prediction_with_real_cheater_row(self):
        """Evaluate ML prediction on an authentic cheater row from final_dataset.csv"""
        payload = {k: float(self.cheat_row[k]) for k in FEATURE_COLUMNS}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        print(f"[Test] Real Cheater Row -> Probability: {data['probability']}%, Risk: {data['risk_level']}, Verdict: {data['verdict']}")

    def test_06_input_validation(self):
        """Test API handles missing telemetry fields gracefully"""
        incomplete_payload = {"yaw": 12.0}
        response = self.client.post('/predict', json=incomplete_payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data['status'], 'error')

    def test_07_moderator_review_workflow(self):
        """Test moderator reviewing a prediction and persisting audit comments"""
        # 1. Login as player and submit telemetry
        self.client.post('/login', data={'username': 'player1', 'password': 'password123'})
        form_data = {k: str(self.cheat_row[k]) for k in FEATURE_COLUMNS}
        self.client.post('/gameplay', data=form_data)

        # 2. Login as moderator
        self.client.post('/login', data={'username': 'moderator1', 'password': 'password123'})
        # Find latest prediction_id
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT prediction_id FROM predictions ORDER BY prediction_id DESC LIMIT 1")
        row = cur.fetchone()
        pred_id = row['prediction_id']
        cur.close()
        conn.close()

        # 3. Post review
        review_resp = self.client.post(f'/review/{pred_id}', data={
            'decision': 'Confirmed Suspicious',
            'comments': 'Verified snap-to-target kinematic acceleration violation.'
        }, follow_redirects=True)
        self.assertEqual(review_resp.status_code, 200)

        # 4. Verify review in DB
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM moderator_reviews WHERE prediction_id = %s", (pred_id,))
        review_entry = cur.fetchone()
        cur.close()
        conn.close()
        self.assertIsNotNone(review_entry)
        self.assertEqual(review_entry['decision'], 'Confirmed Suspicious')

    def test_08_analytics_api_endpoint(self):
        """Test analytics JSON summary"""
        self.client.post('/login', data={'username': 'moderator1', 'password': 'password123'})
        resp = self.client.get('/analytics', headers={'Accept': 'application/json'})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('total_sessions', data)
        self.assertIn('cheat_detection_rate', data)

if __name__ == '__main__':
    print("=" * 80)
    print(" RUNNING SYSTEM TEST SUITE USING REAL DATASET ROWS")
    print("=" * 80)
    unittest.main()
