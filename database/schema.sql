-- =======================================================
-- In-Game Cheat Detection System (Aegis Anti-Cheat)
-- STAGE 7: Comprehensive Relational Database Schema
-- Database Target: MySQL / MariaDB (cheat_detection_db)
-- =======================================================

CREATE DATABASE IF NOT EXISTS cheat_detection_db;
USE cheat_detection_db;

-- 1. Users Table (Core Authentication & System Accounts)
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('player', 'moderator') NOT NULL DEFAULT 'player',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_role (role)
) ENGINE=InnoDB;

-- 2. Players Table (Profile, UUID Mapping, & Trust Ratings)
CREATE TABLE IF NOT EXISTS players (
    player_id INT PRIMARY KEY,
    player_uuid VARCHAR(64) NOT NULL UNIQUE,
    total_matches INT DEFAULT 0,
    trust_score FLOAT DEFAULT 100.0,
    account_status ENUM('active', 'flagged', 'suspended') DEFAULT 'active',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_player_uuid (player_uuid)
) ENGINE=InnoDB;

-- 3. Game Sessions & Telemetry Input Table
CREATE TABLE IF NOT EXISTS game_sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT NOT NULL,
    yaw FLOAT NOT NULL,
    pitch FLOAT NOT NULL,
    delta_yaw FLOAT NOT NULL,
    delta_pitch FLOAT NOT NULL,
    accel_yaw FLOAT NOT NULL,
    accel_pitch FLOAT NOT NULL,
    target_x FLOAT NOT NULL,
    target_y FLOAT NOT NULL,
    target_z FLOAT NOT NULL,
    position_x FLOAT NOT NULL,
    position_y FLOAT NOT NULL,
    position_z FLOAT NOT NULL,
    sensitivity FLOAT NOT NULL,
    new_sequence INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_session_player (player_id),
    INDEX idx_session_time (created_at)
) ENGINE=InnoDB;

-- 4. ML Predictions & Real-Time Risk Scores
-- Stores algorithm output strictly separated from human moderator review
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL UNIQUE,
    prediction ENUM('LEGITIMATE', 'SUSPICIOUS') NOT NULL,
    probability FLOAT NOT NULL,           -- Continuous risk probability (0.00% - 100.00%)
    risk_level ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    model_version VARCHAR(50) NOT NULL DEFAULT 'v2.0-group-aware',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES game_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_prediction_risk (risk_level),
    INDEX idx_prediction_label (prediction)
) ENGINE=InnoDB;

-- 5. Moderator Reviews Table (Human-in-the-Loop Audit Queue)
-- Explicitly isolated from ML prediction to allow audit trail and false-positive tracking
CREATE TABLE IF NOT EXISTS moderator_reviews (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL UNIQUE,
    moderator_id INT NOT NULL,
    decision ENUM('Under Review', 'Confirmed Suspicious', 'Not Suspicious') NOT NULL DEFAULT 'Under Review',
    comments TEXT,
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    FOREIGN KEY (moderator_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_review_decision (decision)
) ENGINE=InnoDB;

-- =======================================================
-- Initial Demonstration Seed Accounts (Pass: password123)
-- =======================================================
INSERT INTO users (username, password_hash, role) 
VALUES 
('player1', 'scrypt:32768:8:1$K5zQ7jA91yE7$b1d7d5668e1c625eb2eb23b090bcad76b5c165313dcb899c75549e3734638c4b1263c7b64923e3e0fce5982855fce72b8364f3316d99b19dfb48e6584283842c', 'player'),
('moderator1', 'scrypt:32768:8:1$K5zQ7jA91yE7$b1d7d5668e1c625eb2eb23b090bcad76b5c165313dcb899c75549e3734638c4b1263c7b64923e3e0fce5982855fce72b8364f3316d99b19dfb48e6584283842c', 'moderator')
ON DUPLICATE KEY UPDATE role=VALUES(role);

-- Seed Player Profile for player1
INSERT INTO players (player_id, player_uuid, total_matches, trust_score, account_status)
SELECT user_id, '25202025-75e3-47f0-9f2b-1e6e9a088193', 12, 98.5, 'active'
FROM users WHERE username = 'player1'
ON DUPLICATE KEY UPDATE total_matches=VALUES(total_matches);
