"""
Data Loader Module - Download and parse Lichess games
Team 029 - CSE6242 Spring 2026
"""

import chess.pgn
import pandas as pd
import numpy as np
import requests
import io
import re
import bz2
import gzip
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Generator
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SKILL_TIERS, VALID_TIME_CONTROLS

# Number of rows per parquet chunk file
CHUNK_SIZE = 1000


def download_lichess_sample(output_path: Optional[Path] = None,
                            num_games: int = 10000) -> Path:
    """
    Download a sample of games from Lichess database.
    Uses the Lichess elite database or API for smaller samples.

    Args:
        output_path: Path to save the downloaded file
        num_games: Approximate number of games to download

    Returns:
        Path to the downloaded/saved file
    """
    if output_path is None:
        output_path = RAW_DATA_DIR / "lichess_sample.pgn"

    # For demonstration, we'll use Lichess API to get recent games
    # In production, you'd download from database.lichess.org
    print(f"Note: For full dataset, download from https://database.lichess.org/")
    print(f"Using Lichess API for sample data...")

    return output_path


def parse_clock_time(clock_str: str) -> Optional[float]:
    """
    Parse clock time from PGN format (e.g., '0:05:30' or '1:30:00') to seconds.

    Args:
        clock_str: Clock time string from PGN

    Returns:
        Time in seconds, or None if parsing fails
    """
    if not clock_str:
        return None

    try:
        # Handle format H:MM:SS or M:SS
        parts = clock_str.split(':')
        if len(parts) == 3:
            hours, mins, secs = map(float, parts)
            return hours * 3600 + mins * 60 + secs
        elif len(parts) == 2:
            mins, secs = map(float, parts)
            return mins * 60 + secs
        else:
            return float(clock_str)
    except (ValueError, AttributeError):
        return None


def parse_time_control(tc_str: str) -> Optional[Dict]:
    """
    Parse time control string (e.g., '300+3' for 5 min + 3 sec increment).

    Args:
        tc_str: Time control string

    Returns:
        Dictionary with base_time and increment, or None
    """
    if not tc_str or tc_str == '-':
        return None

    try:
        if '+' in tc_str:
            base, inc = tc_str.split('+')
            return {'base_time': int(base), 'increment': int(inc)}
        else:
            return {'base_time': int(tc_str), 'increment': 0}
    except (ValueError, AttributeError):
        return None


def get_time_control_category(base_time: int) -> Optional[str]:
    """
    Categorize time control based on base time.

    Args:
        base_time: Base time in seconds

    Returns:
        Category name ('bullet', 'blitz', 'rapid') or None
    """
    for category, (min_time, max_time) in VALID_TIME_CONTROLS.items():
        if min_time <= base_time <= max_time:
            return category
    return None


def parse_single_game(game: chess.pgn.Game) -> Optional[Dict]:
    """
    Parse a single chess game from PGN format.

    Args:
        game: python-chess Game object

    Returns:
        Dictionary with game data, or None if invalid
    """
    headers = game.headers

    # Skip if missing essential headers
    required_headers = ['White', 'Black', 'Result', 'WhiteElo', 'BlackElo', 'TimeControl']
    if not all(h in headers for h in required_headers):
        return None

    # Parse ratings
    try:
        white_elo = int(headers.get('WhiteElo', 0))
        black_elo = int(headers.get('BlackElo', 0))
        if white_elo < 600 or black_elo < 600:
            return None
    except (ValueError, TypeError):
        return None

    # Parse time control
    tc = parse_time_control(headers.get('TimeControl', ''))
    if tc is None:
        return None

    tc_category = get_time_control_category(tc['base_time'])
    if tc_category is None:
        return None

    # Parse result
    result = headers.get('Result', '*')
    if result not in ['1-0', '0-1', '1/2-1/2']:
        return None

    # Extract moves and clock times
    moves = []
    clock_times_white = []
    clock_times_black = []

    node = game
    move_num = 0

    while node.variations:
        node = node.variation(0)
        move_num += 1

        move_data = {
            'move_num': move_num,
            'move_uci': node.move.uci() if node.move else None,
            'move_san': node.san() if node.move else None,
        }

        # Extract clock time from comment
        comment = node.comment
        clock_match = re.search(r'\[%clk\s+(\d+:\d+:\d+)\]', comment)
        if clock_match:
            clock_time = parse_clock_time(clock_match.group(1))
            if move_num % 2 == 1:  # White's move
                clock_times_white.append(clock_time)
            else:  # Black's move
                clock_times_black.append(clock_time)

        moves.append(move_data)

    # Skip very short games
    if len(moves) < 10:
        return None

    # Determine skill tier for both players
    def get_skill_tier(elo):
        for tier, (low, high) in SKILL_TIERS.items():
            if low <= elo < high:
                return tier
        return 'Master'

    game_data = {
        'white_player': headers.get('White', ''),
        'black_player': headers.get('Black', ''),
        'white_elo': white_elo,
        'black_elo': black_elo,
        'white_skill_tier': get_skill_tier(white_elo),
        'black_skill_tier': get_skill_tier(black_elo),
        'result': result,
        'time_control': headers.get('TimeControl', ''),
        'time_control_category': tc_category,
        'base_time': tc['base_time'],
        'increment': tc['increment'],
        'num_moves': len(moves),
        'opening_eco': headers.get('ECO', ''),
        'opening_name': headers.get('Opening', ''),
        'date': headers.get('Date', ''),
        'termination': headers.get('Termination', ''),
        'moves': moves,
        'clock_times_white': clock_times_white,
        'clock_times_black': clock_times_black,
    }

    return game_data


def parse_pgn_file(pgn_path: Path, max_games: int = None) -> Generator[Dict, None, None]:
    """
    Parse games from a PGN file.

    Args:
        pgn_path: Path to PGN file
        max_games: Maximum number of games to parse (None for all)

    Yields:
        Dictionary with game data for each valid game
    """
    games_parsed = 0

    # Handle compressed files
    if str(pgn_path).endswith('.bz2'):
        file_handle = bz2.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
    elif str(pgn_path).endswith('.gz'):
        file_handle = gzip.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
    else:
        file_handle = open(pgn_path, 'r', encoding='utf-8', errors='ignore')

    try:
        while True:
            if max_games and games_parsed >= max_games:
                break

            game = chess.pgn.read_game(file_handle)
            if game is None:
                break

            game_data = parse_single_game(game)
            if game_data:
                games_parsed += 1
                yield game_data

    finally:
        file_handle.close()


def create_games_dataframe(games: List[Dict]) -> pd.DataFrame:
    """
    Convert list of game dictionaries to a pandas DataFrame.

    Args:
        games: List of game dictionaries

    Returns:
        DataFrame with game data (excluding move details)
    """
    # Extract game-level data (exclude detailed moves for the summary DataFrame)
    game_records = []

    for game in games:
        record = {k: v for k, v in game.items()
                  if k not in ['moves', 'clock_times_white', 'clock_times_black']}
        record['has_clock_data'] = len(game.get('clock_times_white', [])) > 0
        game_records.append(record)

    return pd.DataFrame(game_records)


def split_dataframe_to_parquet_chunks(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    chunk_size: int = CHUNK_SIZE
) -> List[Path]:
    """
    Split a DataFrame into multiple parquet files, each containing up to
    `chunk_size` rows. Files are saved as:
        <output_dir>/chunks/<prefix>_part_0000.parquet
        <output_dir>/chunks/<prefix>_part_0001.parquet
        ...

    This makes each file small enough to commit to GitHub individually,
    enabling asynchronous / parallel pushes without hitting the 100 MB
    per-file limit.

    To read all chunks back as a single DataFrame later, simply pass the
    chunks directory to pandas:
        df = pd.read_parquet("<output_dir>/chunks/")

    Args:
        df:         The source DataFrame to split.
        output_dir: Parent directory under which a 'chunks' subfolder is created.
        prefix:     Filename prefix, e.g. 'games_processed' or 'games_full'.
        chunk_size: Maximum number of rows per file (default: 1000).

    Returns:
        List of Path objects for every chunk file written.
    """
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: List[Path] = []
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # ceiling division

    print(f"Splitting {total_rows} rows into {num_chunks} chunk(s) "
          f"of up to {chunk_size} rows each → {chunks_dir}")

    for chunk_index, start_row in enumerate(range(0, total_rows, chunk_size)):
        chunk_df = df.iloc[start_row: start_row + chunk_size]
        chunk_filename = f"{prefix}_part_{chunk_index:04d}.parquet"
        chunk_path = chunks_dir / chunk_filename
        chunk_df.to_parquet(chunk_path, index=False)
        chunk_paths.append(chunk_path)
        print(f"  Saved chunk {chunk_index:04d}: {chunk_filename} "
              f"({len(chunk_df)} rows)")

    print(f"Done — {len(chunk_paths)} chunk file(s) written to {chunks_dir}")
    return chunk_paths


def load_or_create_dataset(pgn_path: Optional[Path] = None,
                           max_games: int = 10000,
                           force_reload: bool = False) -> pd.DataFrame:
    """
    Load processed dataset or create it from PGN file.
    Parquet output is automatically split into CHUNK_SIZE-row files
    inside <PROCESSED_DATA_DIR>/chunks/ for lightweight GitHub storage.

    Args:
        pgn_path: Path to PGN file (uses sample if None)
        max_games: Maximum games to process
        force_reload: Force re-processing even if cached chunks exist

    Returns:
        DataFrame with processed game data
    """
    chunks_dir = PROCESSED_DATA_DIR / "chunks"
    existing_chunks = sorted(chunks_dir.glob("games_processed_part_*.parquet"))

    if existing_chunks and not force_reload:
        print(f"Loading cached dataset from {len(existing_chunks)} chunk(s) in {chunks_dir}")
        return pd.read_parquet(chunks_dir)

    if pgn_path is None:
        pgn_path = RAW_DATA_DIR / "lichess_sample.pgn"

    if not pgn_path.exists():
        print(f"PGN file not found at {pgn_path}")
        print("Please download from https://database.lichess.org/")
        return pd.DataFrame()

    print(f"Parsing games from {pgn_path}...")
    games = list(tqdm(parse_pgn_file(pgn_path, max_games),
                      total=max_games, desc="Parsing games"))

    # Build summary DataFrame and split into chunks
    df = create_games_dataframe(games)
    split_dataframe_to_parquet_chunks(df, PROCESSED_DATA_DIR, "games_processed")

    # Build full DataFrame (with moves) and split into chunks
    games_full_df = pd.DataFrame(games)
    split_dataframe_to_parquet_chunks(games_full_df, PROCESSED_DATA_DIR, "games_full")

    return df


def get_dataset_stats(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for the dataset.

    Args:
        df: Games DataFrame

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_games': len(df),
        'unique_white_players': df['white_player'].nunique(),
        'unique_black_players': df['black_player'].nunique(),
        'rating_range': (df[['white_elo', 'black_elo']].min().min(),
                        df[['white_elo', 'black_elo']].max().max()),
        'avg_game_length': df['num_moves'].mean(),
        'time_control_distribution': df['time_control_category'].value_counts().to_dict(),
        'skill_tier_distribution_white': df['white_skill_tier'].value_counts().to_dict(),
        'skill_tier_distribution_black': df['black_skill_tier'].value_counts().to_dict(),
        'games_with_clock_data': df['has_clock_data'].sum() if 'has_clock_data' in df.columns else 0,
    }

    return stats


if __name__ == "__main__":
    # Test the data loader
    print("ChessInsight Data Loader")
    print("=" * 50)

    # Check for existing PGN files
    pgn_files = list(RAW_DATA_DIR.glob("*.pgn*"))
    if pgn_files:
        print(f"Found PGN files: {pgn_files}")
        df = load_or_create_dataset(pgn_files[0], max_games=1000)
    else:
        print("No PGN files found in data/raw/")
        print("Please download from https://database.lichess.org/")
        print("\nExample: Download lichess_db_standard_rated_2024-01.pgn.zst")
