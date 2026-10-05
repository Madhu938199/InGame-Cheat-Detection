"""
Database Connection & Abstraction Layer
Project: In-Game Cheat Detection System
File: backend/database.py

Supports MySQL (production/default) with transparent SQLite fallback 
to ensure bulletproof execution in college project demonstration environments.
"""

import os
import sqlite3
from werkzeug.security import generate_password_hash

# Database Configuration (Environment variables with sensible local defaults)
DB_TYPE = os.environ.get('DB_TYPE', 'auto')  # 'mysql', 'sqlite', or 'auto'
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'your_mysql_password')
MYSQL_DB = os.environ.get('MYSQL_DB', 'cheat_detection_db')
SQLITE_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'cheat_detection.db')

class DBConnectionWrapper:
    """Wrapper that normalizes MySQL and SQLite cursor/connection interfaces."""
    def __init__(self, conn, is_sqlite=False):
        self.conn = conn
        self.is_sqlite = is_sqlite

    def cursor(self, dictionary=True):
        if self.is_sqlite:
            if dictionary:
                self.conn.row_factory = sqlite3.Row
            else:
                self.conn.row_factory = None
            cur = self.conn.cursor()
            return SQLiteCursorWrapper(cur)
        else:
            return self.conn.cursor(dictionary=dictionary)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()

class SQLiteCursorWrapper:
    def __init__(self, cursor):
        self.cursor = cursor
        self.lastrowid = None

    def execute(self, query, params=None):
        # Convert %s placeholders to ? for SQLite compatibility
        sqlite_query = query.replace('%s', '?')
        # Remove MySQL-specific clauses for SQLite compatibility if any
        if 'ON DUPLICATE KEY UPDATE' in sqlite_query:
            # Handle user seed or review upsert in SQLite
            sqlite_query = sqlite_query.split('ON DUPLICATE KEY UPDATE')[0].strip()
            sqlite_query = sqlite_query.replace('INSERT INTO', 'INSERT OR REPLACE INTO')
            
        if params:
            res = self.cursor.execute(sqlite_query, params)
        else:
            res = self.cursor.execute(sqlite_query)
        self.lastrowid = self.cursor.lastrowid
        return res

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is None:
            return None
        if isinstance(row, sqlite3.Row):
            return dict(row)
        return row

    def fetchall(self):
        rows = self.cursor.fetchall()
        if rows and isinstance(rows[0], sqlite3.Row):
            return [dict(r) for r in rows]
        return rows

    def close(self):
        self.cursor.close()

def get_db_connection():
    """
    Attempts to connect to MySQL. If MySQL is unreachable, automatically
    initializes and connects to the local SQLite database.
    """
    if DB_TYPE != 'sqlite':
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB
            )
            return DBConnectionWrapper(conn, is_sqlite=False)
        except Exception as e:
            if DB_TYPE == 'mysql':
                raise e
            # Auto fallback to SQLite
            # print(f">> MySQL not connected ({e}). Using embedded SQLite database.")
            pass

    # SQLite fallback
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    return DBConnectionWrapper(conn, is_sqlite=True)

def init_db():
    """Initializes tables and seeds default accounts in whichever DB is active."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Users Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """ if conn.is_sqlite else """
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM('player', 'moderator') NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Game Sessions Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INT NOT NULL,
            yaw FLOAT,
            pitch FLOAT,
            delta_yaw FLOAT,
            delta_pitch FLOAT,
            accel_yaw FLOAT,
            accel_pitch FLOAT,
            target_x FLOAT,
            target_y FLOAT,
            target_z FLOAT,
            position_x FLOAT,
            position_y FLOAT,
            position_z FLOAT,
            sensitivity FLOAT,
            new_sequence INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES users(user_id)
        )
    """ if conn.is_sqlite else """
        CREATE TABLE IF NOT EXISTS game_sessions (
            session_id INT AUTO_INCREMENT PRIMARY KEY,
            player_id INT NOT NULL,
            yaw FLOAT,
            pitch FLOAT,
            delta_yaw FLOAT,
            delta_pitch FLOAT,
            accel_yaw FLOAT,
            accel_pitch FLOAT,
            target_x FLOAT,
            target_y FLOAT,
            target_z FLOAT,
            position_x FLOAT,
            position_y FLOAT,
            position_z FLOAT,
            sensitivity FLOAT,
            new_sequence INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)

    # 3. Predictions Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INT NOT NULL,
            prediction VARCHAR(20) NOT NULL,
            probability FLOAT NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            model_version VARCHAR(50) DEFAULT 'v1.0',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES game_sessions(session_id)
        )
    """ if conn.is_sqlite else """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT NOT NULL,
            prediction VARCHAR(20) NOT NULL,
            probability FLOAT NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            model_version VARCHAR(50) DEFAULT 'v1.0',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES game_sessions(session_id) ON DELETE CASCADE
        )
    """)

    # 4. Moderator Reviews Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS moderator_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INT NOT NULL UNIQUE,
            moderator_id INT NOT NULL,
            decision VARCHAR(50) DEFAULT 'Under Review',
            comments TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
            FOREIGN KEY (moderator_id) REFERENCES users(user_id)
        )
    """ if conn.is_sqlite else """
        CREATE TABLE IF NOT EXISTS moderator_reviews (
            review_id INT AUTO_INCREMENT PRIMARY KEY,
            prediction_id INT NOT NULL UNIQUE,
            moderator_id INT NOT NULL,
            decision ENUM('Under Review', 'Confirmed Suspicious', 'Not Suspicious') DEFAULT 'Under Review',
            comments TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
            FOREIGN KEY (moderator_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    # Seed Default Accounts (password: password123)
    default_pw = generate_password_hash('password123')
    try:
        cur.execute("SELECT * FROM users WHERE username = %s", ('player1',))
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                        ('player1', default_pw, 'player'))
            
        cur.execute("SELECT * FROM users WHERE username = %s", ('moderator1',))
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)",
                        ('moderator1', default_pw, 'moderator'))
        conn.commit()
    except Exception as e:
        # print("User seeding note:", e)
        pass

    cur.close()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
