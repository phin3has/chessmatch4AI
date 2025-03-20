import chess

def debug_position(fen, attempted_move):
    """Debug a specific chess position and attempted move."""
    board = chess.Board(fen)
    print(f"FEN: {fen}")
    print(f"Attempting to play: {attempted_move}")
    
    # List all legal moves in SAN format
    print("\nLegal moves in SAN format:")
    legal_moves_san = []
    for move in board.legal_moves:
        san = board.san(move)
        legal_moves_san.append(san)
    print(", ".join(legal_moves_san))
    
    # Check if the attempted move is legal
    print(f"\nIs '{attempted_move}' in the list of legal moves? {attempted_move in legal_moves_san}")
    
    # Try to parse the move directly
    try:
        move = board.parse_san(attempted_move)
        print(f"Move '{attempted_move}' parsed successfully as {move.uci()}")
    except ValueError as e:
        print(f"Error parsing '{attempted_move}': {e}")

# Debug the first position (where bxc3 failed)
print("DEBUGGING POSITION 1 (bxc3)")
debug_position("r1bqkb1r/pp3ppp/2n2n2/8/8/2N5/PPP1PPPP/R1BQKBNR b KQkq - 0 1", "bxc3")

print("\n" + "="*50 + "\n")

# Debug the second position (where Bd6 failed)
print("DEBUGGING POSITION 2 (Bd6)")
debug_position("r1b2k1r/pp1K2pp/2p2p2/8/3N1B2/8/PPP2PPP/R5Q1 b - - 0 1", "Bd6")
