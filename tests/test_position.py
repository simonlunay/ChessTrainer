from __future__ import annotations

import chess
import numpy as np

from chesstrainer.vision.labels import CLASS_NAMES, NUM_CLASSES
from chesstrainer.vision.position import (
    PositionTracker,
    infer_state,
    labels_of,
    repair,
)

L = {name: i for i, name in enumerate(CLASS_NAMES)}


def probs_for(board: chess.Board, p: float = 0.97) -> np.ndarray:
    """Confident one-hot-ish probabilities (square order) for a board."""
    labels = labels_of(board)
    out = np.full((64, NUM_CLASSES), (1 - p) / (NUM_CLASSES - 1), np.float32)
    out[np.arange(64), labels] = p
    return out


def set_square(probs: np.ndarray, sq: chess.Square, dist: dict[str, float]) -> None:
    probs[sq] = 1e-4
    for name, v in dist.items():
        probs[sq, L[name]] = v


# ------------------------------------------------------------------------------- repair


def test_two_white_kings_keeps_more_confident_and_uses_next_best_class() -> None:
    board = chess.Board()
    probs = probs_for(board)
    # d1 holds the queen but the classifier calls it a king (queen second).
    set_square(probs, chess.D1, {"K": 0.55, "Q": 0.40, "B": 0.05})
    labels, notes = repair(probs)
    assert labels[chess.E1] == L["K"]  # 0.97 > 0.55: the real king stays
    assert labels[chess.D1] == L["Q"]  # reclassified to its next most likely class
    assert any("two white kings" in n for n in notes)


def test_two_kings_reclassified_square_never_becomes_a_king_again() -> None:
    board = chess.Board()
    probs = probs_for(board)
    set_square(probs, chess.D8, {"k": 0.6, "K": 0.3, "q": 0.1})
    labels, _ = repair(probs)
    assert labels[chess.D8] == L["q"]
    assert (labels == L["k"]).sum() == 1 and (labels == L["K"]).sum() == 1


def test_missing_king_is_restored_from_most_king_like_square() -> None:
    board = chess.Board()
    probs = probs_for(board)
    set_square(probs, chess.E8, {"q": 0.6, "k": 0.35, "empty": 0.05})
    labels, notes = repair(probs)
    assert labels[chess.E8] == L["k"]
    assert any("no black king" in n for n in notes)


def test_pawn_on_back_rank_is_reclassified() -> None:
    board = chess.Board()
    probs = probs_for(board)
    set_square(probs, chess.B1, {"P": 0.5, "N": 0.45, "empty": 0.05})
    labels, _ = repair(probs)
    assert labels[chess.B1] == L["N"]


def test_clean_position_is_untouched() -> None:
    labels, notes = repair(probs_for(chess.Board()))
    assert np.array_equal(labels, labels_of(chess.Board()))
    assert notes == []


# ------------------------------------------------------------------------------- state inference


def test_infer_castling_and_assumed_turn() -> None:
    board, assumed, _ = infer_state(labels_of(chess.Board()), white_bottom=False)
    assert board.castling_rights == chess.Board().castling_rights
    assert board.turn == chess.BLACK and assumed  # bottom side assumed to move


def test_side_in_check_must_move() -> None:
    b = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")  # white king in check from e2 rook
    board, assumed, _ = infer_state(labels_of(b), white_bottom=False)
    assert board.turn == chess.WHITE and not assumed


def test_last_move_highlight_gives_turn_and_en_passant() -> None:
    b = chess.Board()
    b.push_san("e4")
    board, assumed, _ = infer_state(labels_of(b), True, highlight_squares=[chess.E2, chess.E4])
    assert board.turn == chess.BLACK and not assumed
    assert board.ep_square == chess.E3


# ------------------------------------------------------------------------------- tracking


def feed(tracker: PositionTracker, probs: np.ndarray, frames: int = 2, **kw: object) -> object:
    reading = None
    for _ in range(frames):
        r = tracker.update(probs, True, **kw)  # type: ignore[arg-type]
        reading = r or reading
    return reading


def test_tracker_follows_game_with_castling_and_en_passant() -> None:
    tracker = PositionTracker()
    game = chess.Board()
    feed(tracker, probs_for(game))
    for san in ["e4", "Nf6", "e5", "d5", "exd6", "e6", "Nf3", "Be7", "Bb5+", "c6", "O-O"]:
        game.push_san(san)
        r = feed(tracker, probs_for(game))
        assert r is not None, san
        assert tracker.current is not None
        assert tracker.current.fen == game.fen(), san
    assert tracker.current is not None and tracker.current.source == "history"


def test_single_frame_flicker_is_not_committed() -> None:
    tracker = PositionTracker(stable_frames=2)
    game = chess.Board()
    feed(tracker, probs_for(game))
    moved = game.copy()
    moved.push_san("e4")
    assert tracker.update(probs_for(moved), True) is None  # first sighting only
    assert tracker.update(probs_for(game), True) is None  # back: nothing changed
    assert tracker.current is not None and tracker.current.fen == game.fen()


def test_history_fixes_low_confidence_square() -> None:
    tracker = PositionTracker()
    game = chess.Board()
    feed(tracker, probs_for(game))
    game.push_san("Nf3")
    probs = probs_for(game)
    set_square(probs, chess.F3, {"B": 0.55, "N": 0.4, "empty": 0.05})  # classifier unsure
    feed(tracker, probs)
    assert tracker.current is not None
    assert tracker.current.board.piece_at(chess.F3) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert tracker.current.moves == [chess.Move.from_uci("g1f3")]


def test_two_moves_between_frames() -> None:
    tracker = PositionTracker()
    game = chess.Board()
    feed(tracker, probs_for(game))
    game.push_san("d4")
    game.push_san("d5")
    feed(tracker, probs_for(game))
    assert tracker.current is not None and tracker.current.fen == game.fen()
    assert len(tracker.current.moves) == 2


def test_takeback_returns_to_earlier_position_with_its_state() -> None:
    tracker = PositionTracker()
    game = chess.Board()
    feed(tracker, probs_for(game))
    snapshots = []
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4"]:
        game.push_san(san)
        feed(tracker, probs_for(game))
        snapshots.append(game.fen())
    earlier = chess.Board(snapshots[1])  # after 1...e5, white to move
    feed(tracker, probs_for(earlier))
    assert tracker.current is not None and tracker.current.fen == snapshots[1]


def test_unrelated_position_becomes_fresh_reading() -> None:
    tracker = PositionTracker()
    feed(tracker, probs_for(chess.Board()))
    other = chess.Board("8/5k2/8/8/3Q4/8/2q5/6K1 w - - 0 1")
    r = feed(tracker, probs_for(other))
    assert r is not None and tracker.current is not None
    assert tracker.current.source == "fresh"
    assert tracker.current.board.board_fen() == other.board_fen()
