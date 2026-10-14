from __future__ import annotations

from pathlib import Path

import chess
import pytest

from chesstrainer.levels import all_levels, get_level
from chesstrainer.openings.book import Book, default_book, name_matches, pos_key
from chesstrainer.openings.practice import DAY, PracticeStore
from chesstrainer.openings.repertoire import all_specs, find_spec
from chesstrainer.openings.trainer import TrainerSession


@pytest.fixture(scope="module")
def book() -> Book:
    return default_book()


def play(session: TrainerSession, sans: str, board: chess.Board | None = None) -> list[str]:
    """Push SAN moves one by one, feed each position to the session, return the event kinds."""
    board = board if board is not None else chess.Board()
    kinds: list[str] = []
    if not board.move_stack:
        kinds += [e.kind for e in session.observe(board)]
    for san in sans.split():
        board.push_san(san)
        kinds += [e.kind for e in session.observe(board.copy())]
    return kinds


# -- book and repertoire -----------------------------------------------------------------------


def test_book_indexes_lines_and_names_positions(book: Book) -> None:
    assert len(book.entries) > 3000
    b = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]:
        b.push_san(san)
    entry = book.entry_at(pos_key(b))
    assert entry is not None and entry.name == "Italian Game: Giuoco Piano"
    assert entry.eco == "C50"


def test_name_matching_rules() -> None:
    assert name_matches("Queen's Pawn Game: London System", "London System")
    assert name_matches("Indian Defense: Accelerated London System", "London System")
    assert name_matches("Queen's Gambit Declined: Exchange Variation", "Queen's Gambit")
    assert not name_matches("Semi-Slav Defense", "Slav Defense")
    assert name_matches("Grünfeld Defense: Exchange Variation", "Grunfeld Defense")
    full = "Italian Game: Giuoco Pianissimo"
    assert name_matches("Italian Game: Giuoco Pianissimo, Canal Variation", full)
    assert not name_matches("Italian Game: Giuoco Piano", full)


def test_search_finds_variations(book: Book) -> None:
    hits = book.search("najdorf")
    assert "Sicilian Defense: Najdorf Variation" in hits
    assert all("najdorf" in h.lower() for h in hits)


def test_repertoire_lines_are_legal_and_annotated() -> None:
    specs = all_specs()  # parsing raises on an illegal move
    assert specs
    for spec in specs:
        assert spec.ideas, spec.name
        for line in spec.lines:
            b = chess.Board()
            for move in line:
                if b.turn == spec.side:
                    assert spec.note(b.san(move)) is not None, (spec.name, b.san(move))
                b.push(move)


def test_beginner_gets_two_or_three_curated_openings_per_side(book: Book) -> None:
    beginner = get_level("beginner")
    for side, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        names = beginner.openings[side]
        assert 2 <= len(names) <= 3
        for name in names:
            spec = find_spec(name, color)
            assert spec is not None and spec.lines, (name, side)
    for level in all_levels():  # every recommendation exists in the dataset
        for names in level.openings.values():
            for name in names:
                assert book.matching(name), name


# -- guided mode -------------------------------------------------------------------------------


def test_guided_italian_follows_the_curated_line_and_completes(book: Book) -> None:
    s = TrainerSession.create(book, "Italian Game", chess.WHITE, get_level("beginner"))
    assert s.state == "waiting" and "starting position" in s.prompt()
    board = chess.Board()
    assert [e.kind for e in s.observe(board)] == ["start"]
    assert s.arrow_move() == chess.Move.from_uci("e2e4")
    kinds = play(s, "e4 e5 Nf3 Nc6 Bc4 Bc5 c3", board)
    assert kinds == ["user_correct", "opp_book", "user_correct", "opp_book"] + [
        "user_correct",
        "opp_book",
        "user_correct",
    ]
    assert s.variation_name().startswith("Italian Game: ")
    assert "Book replies" in s.prompt()
    kinds = play(s, "Nf6", board)
    assert kinds == ["opp_book", "complete"]  # 8 plies = beginner depth
    assert s.state == "complete" and s.arrow_move() is None


def test_opponent_leaving_book_is_flagged_for_grading(book: Book) -> None:
    s = TrainerSession.create(book, "Italian Game", chess.WHITE, get_level("expert"))
    board = chess.Board()
    play(s, "e4 e5 Nf3", board)
    board.push_san("a5")  # not in the Lichess book
    events = s.observe(board.copy())
    assert [e.kind for e in events] == ["opp_left"]
    e = events[0]
    assert e.grade and e.mover == chess.BLACK and "left the book" in e.message
    assert s.state == "left" and s.arrow_move() is None
    assert s.prompt().startswith("Out of book")
    board.push_san("d4")  # play continues out of book without new events
    assert s.observe(board.copy()) == [] and s.state == "left"


def test_opponent_choosing_another_opening(book: Book) -> None:
    s = TrainerSession.create(book, "Italian Game", chess.WHITE, get_level("novice"))
    board = chess.Board()
    play(s, "e4", board)
    board.push_san("c5")
    events = s.observe(board.copy())
    assert [e.kind for e in events] == ["opp_other_opening"]
    assert "Sicilian Defense" in events[0].message and not events[0].grade


def test_user_deviations_and_takeback(book: Book) -> None:
    s = TrainerSession.create(book, "Italian Game", chess.WHITE, get_level("expert"))
    board = chess.Board()
    play(s, "e4 e5 Nf3 Nc6", board)
    # A book move of another opening (Ruy Lopez) ends the line, without a grade.
    alt = board.copy()
    alt.push_san("Bb5")
    events = s.observe(alt)
    assert [e.kind for e in events] == ["user_other_opening"]
    assert "Ruy Lopez" in events[0].message and "3. Bc4" in events[0].message
    assert not events[0].grade and s.state == "left"
    # Taking it back returns to the line.
    events = s.observe(board.copy())
    assert [e.kind for e in events] == ["takeback"]
    assert s.arrow_move() == chess.Move.from_uci("f1c4")
    # Another book move of the same opening is accepted but flagged.
    play(s, "Bc4 Bc5", board)
    alt = board.copy()
    alt.push_san("b4")  # Evans Gambit instead of the line's c3
    events = s.observe(alt)
    assert [e.kind for e in events] == ["user_other"] and s.in_book
    assert "4. c3" in events[0].message
    s.observe(board.copy())
    # A move outside the book leaves it and asks for grading against the book move.
    bad = board.copy()
    bad.push_san("h4")
    events = s.observe(bad)
    assert [e.kind for e in events] == ["user_left"] and events[0].grade
    assert events[0].expected == chess.Move.from_uci("c2c3") and s.state == "left"
    assert [e.kind for e in s.observe(board.copy())] == ["takeback"] and s.in_book


def test_two_moves_between_frames_and_mid_line_start(book: Book) -> None:
    s = TrainerSession.create(book, "Scandinavian Defense", chess.BLACK, get_level("expert"))
    board = chess.Board()
    play(s, "e4 d5 exd5", board)
    board.push_san("Qxd5")
    board.push_san("Nc3")  # both moves happened between two readings
    assert [e.kind for e in s.observe(board.copy())] == ["user_correct", "opp_book"]
    assert s.arrow_move() == chess.Move.from_uci("d5a5")
    # A fresh session can pick the game up in the middle of the line.
    fresh = TrainerSession.create(book, "Scandinavian Defense", chess.BLACK, get_level("expert"))
    assert [e.kind for e in fresh.observe(board.copy())] == ["resync"]
    assert fresh.arrow_move() == chess.Move.from_uci("d5a5")
    assert len(fresh.history) == len(board.move_stack) + 1


def test_transposition_is_recognised(book: Book) -> None:
    s = TrainerSession.create(book, "London System", chess.WHITE, get_level("expert"))
    board = chess.Board()
    kinds = play(s, "d4 Nf6 Bf4 d5 e3 e6", board)
    assert "opp_left" not in kinds and s.in_book
    assert s.arrow_move() == chess.Move.from_uci("g1f3")


def test_dataset_only_opening_still_trains(book: Book) -> None:
    s = TrainerSession.create(
        book, "Sicilian Defense: Najdorf Variation", chess.BLACK, get_level("expert")
    )
    board = chess.Board()
    play(s, "e4", board)
    assert s.arrow_move() == chess.Move.from_uci("c7c5")
    play(s, "c5 Nf3", board)
    assert s.arrow_move() == chess.Move.from_uci("d7d6")


def test_unknown_position_stops_tracking(book: Book) -> None:
    s = TrainerSession.create(book, "Italian Game", chess.WHITE, get_level("beginner"))
    play(s, "e4 e5")
    other = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    assert [e.kind for e in s.observe(other)] == ["lost"] and s.state == "waiting"


# -- practice mode -----------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def test_spaced_repetition_schedule(tmp_path: Path) -> None:
    clock = Clock()
    store = PracticeStore(tmp_path / "o.sqlite", clock=clock)
    card = store.card("Italian Game", "white", "k", "e2e4")
    assert card.new and store.is_due(card)
    card = store.review(card, "e2e4")
    assert card.interval_days == 1 and not store.is_due(card)
    clock.t += DAY
    card = store.review(store.card("Italian Game", "white", "k", "e2e4"), "e2e4")
    assert card.interval_days == 3
    clock.t += 3 * DAY
    card = store.review(card, "e2e4")
    assert card.interval_days == pytest.approx(3 * 2.6)  # ease 2.5 + 2 x 0.05
    card = store.review(card, "d2d4")  # a miss: due again in 10 minutes, ladder restarts
    assert card.reps == 0 and card.lapses == 1 and card.ease == pytest.approx(2.45)
    assert not store.is_due(card)
    clock.t += 601
    assert store.is_due(store.card("Italian Game", "white", "k", "e2e4"))


def test_practice_hides_arrows_records_answers_and_reveals_after_a_miss(
    book: Book, tmp_path: Path
) -> None:
    store = PracticeStore(tmp_path / "o.sqlite", clock=Clock())
    s = TrainerSession.create(
        book, "Italian Game", chess.WHITE, get_level("beginner"), "practice", store
    )
    board = chess.Board()
    s.observe(board)
    assert s.expected_move() == chess.Move.from_uci("e2e4") and s.arrow_move() is None
    assert "play the book move" in s.prompt()
    kinds = play(s, "e4 e5 Nf3 Nc6", board)
    assert kinds == ["user_correct", "opp_book", "user_correct", "opp_book"]
    wrong = board.copy()
    wrong.push_san("d4")  # Scotch: book, but not this line
    assert [e.kind for e in s.observe(wrong)] == ["user_other"]  # Scotch Gambit is an Italian line
    s.observe(board.copy())  # take it back: now the arrow shows the answer
    assert s.arrow_move() == chess.Move.from_uci("f1c4")
    assert s.score == {"correct": 2, "missed": 1}
    cards = store.cards("Italian Game", "white")
    assert len(cards) == 3 and sum(c.lapses for c in cards.values()) == 1
    assert "1 missed" in s.practice_summary()


def test_practice_suggests_the_reply_with_due_positions(book: Book, tmp_path: Path) -> None:
    store = PracticeStore(tmp_path / "o.sqlite", clock=Clock())
    s = TrainerSession.create(
        book, "Italian Game", chess.BLACK, get_level("intermediate"), "practice", store
    )
    board = chess.Board()
    play(s, "e4 e5 Nf3 Nc6 Bc4", board)  # Black to move: our turn, no suggestion
    assert s.practice_reply() is None
    play(s, "Bc5", board)
    reply = s.practice_reply()
    assert reply is not None and reply in s.tree.book_moves(board)
    assert "To practise what is due" in s.prompt()
