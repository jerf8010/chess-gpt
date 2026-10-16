"""
chess_gpt.rules — Reglas para revisar partidas generadas (no depende de PyTorch).

Una jugada cuenta como error si:
  - python-chess no la puede jugar (ilegal, ambigua o notación inválida)
  - dice 'x' y no captura nada        → "captura falsa"
  - captura algo y no dice 'x'        → "captura sin x"
  - dice '#' y no es mate             → "mate falso"
  - dice '+' y no es jaque            → "jaque falso"

Las pruebas viven en tests/test_rules.py  →  pytest
"""
import re
import chess

RESULTADOS = {"1-0", "0-1", "1/2-1/2"}

PIEZAS = {"N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN, "K": chess.KING}
NOMBRES = {chess.PAWN: "peón", chess.KNIGHT: "caballo", chess.BISHOP: "alfil",
           chess.ROOK: "torre", chess.QUEEN: "dama", chess.KING: "rey"}


# ======================= LECTURA DE JUGADAS =======================
def limpiar_token(token):
    """'12.Nf3' → 'Nf3'. Un formato raro se regresa tal cual para que cuente como error."""
    if token and token[0].isdigit() and "." in token:
        return token.split(".", 1)[1]
    return token


def jugadas(texto_partida):
    """Genera solo las jugadas de una partida: sin números de jugada ni resultado."""
    for token in texto_partida.split():
        if token in RESULTADOS:
            continue
        yield limpiar_token(token)


def contar_plies(texto_partida):
    """Número de medias jugadas de una partida."""
    return sum(1 for _ in jugadas(texto_partida))


def partidas_completas(texto):
    """Partidas que terminan en '\\n' (lo que sigue al último '\\n' está cortado)."""
    pedazos = texto.split("\n")
    return [p for p in pedazos[:-1] if p]


# ======================= VALIDACIÓN =======================
def primera_ilegal_detalle(texto_partida):
    """Regresa (ply, jugada, tablero, error) del primer error, o None si la partida es correcta.
    El tablero siempre es la posición ANTES de la jugada problemática.
    `error` es una excepción de python-chess o un texto como "captura falsa"."""
    board = chess.Board()
    for ply, token in enumerate(jugadas(texto_partida), start=1):
        # 1) interpretar la jugada SIN jugarla todavía
        try:
            move = board.parse_san(token)
        except ValueError as error:
            return ply, token, board.copy(), error
        if move == chess.Move.null():                 # '--' es la "jugada nula": no es ajedrez real
            return ply, token, board.copy(), chess.InvalidMoveError("jugada nula")

        # 2) ¿la 'x' dice la verdad?
        dice_captura = "x" in token
        if dice_captura != board.is_capture(move):    # is_capture incluye la captura al paso
            motivo = "captura falsa" if dice_captura else "captura sin x"
            return ply, token, board.copy(), motivo

        # 3) jugarla y revisar '#' y '+'
        board.push(move)
        if token.endswith("#") and not board.is_checkmate():
            board.pop()
            return ply, token, board.copy(), "mate falso"
        if token.endswith("+") and not board.is_check():
            board.pop()
            return ply, token, board.copy(), "jaque falso"

    return None


def primera_ilegal(texto_partida):
    """Versión simple: solo el número de media jugada del primer error, o None."""
    resultado = primera_ilegal_detalle(texto_partida)
    return None if resultado is None else resultado[0]


# ======================= DIAGNÓSTICO =======================
def tipo_de_pieza(jugada):
    """'Nf3' → 'caballo', 'e4' → 'peón', 'O-O' → 'enroque'."""
    if not jugada:
        return "formato (sin jugada)"
    if jugada.startswith("O"):
        return "enroque"
    if jugada[0] in PIEZAS:
        return NOMBRES[PIEZAS[jugada[0]]]
    return "peón"


def causa(jugada, board, error):
    """Clasifica POR QUÉ la jugada es un error en ese tablero."""
    if isinstance(error, str):                        # errores de anotación: "captura falsa", "mate falso"...
        return error
    if not jugada:
        return "formato: número sin jugada"
    if isinstance(error, chess.InvalidMoveError):
        return "notación inválida"
    if isinstance(error, chess.AmbiguousMoveError):
        return "ambigua (dos piezas pueden ir ahí)"
    if jugada.startswith("O"):
        return "enroque imposible"

    casillas = re.findall(r"[a-h][1-8]", jugada)
    if not casillas:
        return "notación inválida"
    destino = chess.parse_square(casillas[-1])
    tipo = PIEZAS.get(jugada[0], chess.PAWN)
    color = board.turn

    if not board.pieces(tipo, color):
        return "esa pieza ya no existe"
    ocupante = board.piece_at(destino)
    if ocupante and ocupante.color == color:
        return "casilla ocupada por pieza propia"
    for m in board.pseudo_legal_moves:
        if m.to_square == destino and board.piece_type_at(m.from_square) == tipo:
            return "dejaría a su rey en jaque"
    return "ninguna pieza de ese tipo alcanza la casilla"
