"""
ChessInsight Configuration
Team 029 - CSE6242 Spring 2026
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VIZ_DIR = PROJECT_ROOT / "visualizations"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, VIZ_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data settings
PGN_FILE_PATH = RAW_DATA_DIR / "data_1m_games.pgn"  # Path to the 1M games PGN file
MIN_GAMES_PER_PLAYER = 20  # Minimum games for player-level analysis
SAMPLE_SIZE = 200000  # Process 200k games - substantial sample from 1M game dataset

# Skill tier definitions (Elo ranges) - Per proposal: 4 tiers
SKILL_TIERS = {
    'Beginner': (0, 1200),       # <1200
    'Intermediate': (1200, 1600), # 1200-1600
    'Advanced': (1600, 2000),     # 1600-2000
    'Expert': (2000, 4000)        # 2000+
}

# Game phase definitions (by move number)
GAME_PHASES = {
    'opening': (1, 10),
    'middlegame': (11, 25),
    'endgame': (26, 200)
}

# Time controls to include (in seconds for base time)
VALID_TIME_CONTROLS = {
    'bullet': (60, 120),      # 1-2 minutes
    'blitz': (180, 600),      # 3-10 minutes
    'rapid': (600, 1800),     # 10-30 minutes
}

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_CLUSTERS_RANGE = (3, 8)  # Range to search for optimal k

# Feature categories
TIME_FEATURES = [
    'avg_time_opening', 'avg_time_middlegame', 'avg_time_endgame',
    'time_variance_opening', 'time_variance_middlegame', 'time_variance_endgame',
    'low_time_move_ratio', 'time_trouble_frequency'
]

ACCURACY_FEATURES = [
    'blunder_rate', 'mistake_rate', 'avg_centipawn_loss', 'accuracy_percentage'
]

COMPLEXITY_FEATURES = [
    'avg_position_complexity', 'material_imbalance_freq', 'piece_activity_score'
]

OPENING_FEATURES = [
    'opening_aggression_score', 'book_deviation_move'
]

ALL_FEATURES = TIME_FEATURES + ACCURACY_FEATURES + COMPLEXITY_FEATURES + OPENING_FEATURES
