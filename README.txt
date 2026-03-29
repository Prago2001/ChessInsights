================================================================================
CHESSINSIGHT - Chess Player Analytics and Behavioral Classification
Team 029 - CSE6242 Data and Visual Analytics - Spring 2026
================================================================================

DESCRIPTION
-----------
ChessInsight is a chess analytics platform that analyzes Lichess game data to:
1. Extract behavioral features from chess games (time usage, accuracy, etc.)
2. Classify players into skill tiers (Beginner to Master) using machine learning
3. Identify player behavioral archetypes through clustering analysis
4. Generate interactive visualizations of the analysis results

The system processes PGN (Portable Game Notation) files containing chess games
with clock annotations to extract time-based features and other behavioral
metrics.

Key Features:
- Parses standard PGN format with clock annotations
- Extracts 41 game-level and 30 player-level features
- Random Forest classifier for skill tier prediction
- K-Means clustering for behavioral archetype identification
- Comprehensive visualization suite


INSTALLATION
------------
1. Ensure Python 3.9+ is installed on your system

2. Navigate to the project directory:
   cd FS

3. Install required dependencies:
   pip install -r requirements.txt

   Required packages include:
   - python-chess (PGN parsing)
   - pandas, numpy (data manipulation)
   - scikit-learn, xgboost (machine learning)
   - matplotlib, seaborn, plotly (visualization)
   - tqdm (progress bars)

4. IMPORTANT - Download the dataset:
   The data file is NOT included due to its large size (2.2 GB).

   You must download and place the PGN file in the data/raw/ folder:

   Option A - Use our preprocessed sample:
   - Download data_1m_games.pgn from [your shared link]
   - Place it in: FS/data/raw/data_1m_games.pgn

   Option B - Download from Lichess directly:
   - Visit: https://database.lichess.org/
   - Download any monthly database (e.g., lichess_db_standard_rated_2024-01.pgn.zst)
   - Decompress and rename to data_1m_games.pgn
   - Place in: FS/data/raw/

   The file path should be: FS/data/raw/data_1m_games.pgn


EXECUTION
---------
Run the complete analysis pipeline:

   python run_analysis.py

Command-line options:
   --n-games N       Number of games to process (default: 100000)
   --synthetic       Use synthetic data for testing (no PGN file needed)
   --force-reload    Re-parse PGN file, ignoring cached data

Examples:
   # Run with 50,000 games
   python run_analysis.py --n-games 50000

   # Test without data file (synthetic data)
   python run_analysis.py --synthetic

   # Force re-processing of data
   python run_analysis.py --force-reload

Output Files:
   data/processed/     - Cached parquet files (for faster subsequent runs)
   models/             - Trained models and metrics (JSON, CSV, PKL)
   visualizations/     - Generated charts (PNG files)

Expected runtime:
   - First run (parsing PGN): Varies based on n_games and system specs
   - Subsequent runs (cached): Significantly faster due to parquet caching


PROJECT STRUCTURE
-----------------
FS/
├── README.txt              # This file
├── requirements.txt        # Python dependencies
├── config.py               # Configuration settings
├── run_analysis.py         # Main pipeline script
├── data/
│   ├── raw/                # Place data_1m_games.pgn here
│   └── processed/          # Cached processed data (auto-generated)
├── src/
│   ├── data_loader.py      # PGN parsing and data loading
│   ├── feature_extractor.py # Feature engineering
│   ├── classifier.py       # Skill tier classification
│   ├── clustering.py       # Behavioral clustering
│   └── visualizations.py   # Chart generation
├── models/                 # Saved models and results
├── visualizations/         # Generated charts
└── notebooks/              # Jupyter notebooks (optional)


CONFIGURATION
-------------
Edit config.py to customize:
- SAMPLE_SIZE: Number of games to process (default: 100000)
- MIN_GAMES_PER_PLAYER: Minimum games for player-level analysis (default: 20)
- SKILL_TIERS: Elo ranges for skill classification
- VALID_TIME_CONTROLS: Which time controls to include


TROUBLESHOOTING
---------------
1. "PGN file not found" error:
   - Ensure data_1m_games.pgn is in FS/data/raw/
   - Check file permissions

2. Memory errors with large datasets:
   - Reduce --n-games parameter
   - Process in batches

3. Missing dependencies:
   - Run: pip install -r requirements.txt
   - For conda: conda install -c conda-forge python-chess

4. Slow parsing:
   - First run parses PGN and caches to parquet
   - Subsequent runs use cache (much faster)


DATASET INFORMATION
-------------------
Source: Lichess.org rated games database
Format: PGN with clock annotations
Expected fields per game:
- Player names and ratings (Elo)
- Time control and termination type
- Opening ECO code
- Move sequence with clock times [%clk H:MM:SS]

Validation: Games without required fields are automatically filtered.


CONTACT
-------
Team 029 - Georgia Tech CSE6242
Spring 2026
