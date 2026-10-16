"""
chess_gpt.web — Servidor local (Flask) para jugar contra el modelo en el navegador.

API (todas responden con el estado completo de la partida en JSON):
  GET  /api/estado
  POST /api/nueva      {"color": "w" | "b", "modo", "temperatura"}
  POST /api/jugar      {"uci": "e2e4", "modo", "temperatura"}
  POST /api/deshacer   {"modo", "temperatura"}
"""
import os
import threading

import chess
from flask import Flask, jsonify, request

MOTIVOS = {
    "CHECKMATE": "checkmate",
    "STALEMATE": "stalemate",
    "INSUFFICIENT_MATERIAL": "insufficient material",
    "SEVENTYFIVE_MOVES": "the 75-move rule",
    "FIVEFOLD_REPETITION": "fivefold repetition",
}

# rules.causa() devuelve etiquetas en español (las usa la evaluación); aquí se traducen solo para mostrar
CAUSAS_EN = {
    "notación inválida": "invalid notation",
    "ambigua (dos piezas pueden ir ahí)": "ambiguous (two pieces can go there)",
    "enroque imposible": "castling isn't possible",
    "esa pieza ya no existe": "that piece no longer exists",
    "casilla ocupada por pieza propia": "square occupied by its own piece",
    "dejaría a su rey en jaque": "would leave its king in check",
    "ninguna pieza de ese tipo alcanza la casilla": "no piece of that type can reach the square",
    "formato: número sin jugada": "format: move number with no move",
    "captura falsa": "marked as a capture, but nothing is captured",
    "captura sin x": "a capture written without the x",
    "mate falso": "marked as mate, but it isn't",
    "jaque falso": "marked as check, but it isn't",
}


def traducir_analisis(analisis):
    """Copia del análisis con la causa del error traducida al inglés."""
    if not analisis:
        return analisis
    libre = dict(analisis["libre"])
    if libre.get("causa"):
        libre["causa"] = CAUSAS_EN.get(libre["causa"], libre["causa"])
    return {**analisis, "libre": libre}


def create_app(motor):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    lock = threading.Lock()                       # una sola partida, protegida entre peticiones
    partida = {"board": chess.Board(), "humano": chess.WHITE, "analisis": None}

    # ------------------------------------------------------------ helpers
    def opciones():
        datos = request.get_json(silent=True) or {}
        modo = datos.get("modo", "mejor")
        if modo not in ("mejor", "muestrear"):
            modo = "mejor"
        try:
            temperatura = float(datos.get("temperatura", 1.0))
        except (TypeError, ValueError):
            temperatura = 1.0
        return datos, modo, min(max(temperatura, 0.05), 3.0)

    def turno_del_modelo(modo, temperatura):
        board = partida["board"]
        if not board.is_game_over() and board.turn != partida["humano"]:
            analisis = motor.analizar(board, modo=modo, temperatura=temperatura)
            board.push_uci(analisis["uci"])
            partida["analisis"] = analisis

    def estado():
        board = partida["board"]
        replay, jugadas = chess.Board(), []
        for move in board.move_stack:
            jugadas.append(replay.san(move))
            replay.push(move)

        fin = board.is_game_over()
        rey = board.king(board.turn) if board.is_check() else None
        motivo = None
        if fin:
            nombre = board.outcome().termination.name
            motivo = MOTIVOS.get(nombre, nombre.lower())

        return jsonify({
            "piezas": {chess.square_name(sq): p.symbol() for sq, p in board.piece_map().items()},
            "turno": "w" if board.turn == chess.WHITE else "b",
            "humano": "w" if partida["humano"] == chess.WHITE else "b",
            "legales": [m.uci() for m in board.legal_moves],
            "ultima": board.peek().uci() if board.move_stack else None,
            "jaque_en": chess.square_name(rey) if rey is not None else None,
            "jugadas": jugadas,
            "fin": fin,
            "resultado": board.result() if fin else None,
            "motivo": motivo,
            "analisis": traducir_analisis(partida["analisis"]),
            "modelo": os.path.basename(motor.nombre),
            "dispositivo": motor.device,
        })

    # ------------------------------------------------------------ rutas
    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/estado")
    def api_estado():
        with lock:
            return estado()

    @app.post("/api/nueva")
    def api_nueva():
        datos, modo, temperatura = opciones()
        with lock:
            partida["board"] = chess.Board()
            partida["humano"] = chess.BLACK if datos.get("color") == "b" else chess.WHITE
            partida["analisis"] = None
            turno_del_modelo(modo, temperatura)        # si juegas negras, el modelo abre
            return estado()

    @app.post("/api/jugar")
    def api_jugar():
        datos, modo, temperatura = opciones()
        with lock:
            board = partida["board"]
            if board.is_game_over() or board.turn != partida["humano"]:
                return jsonify({"error": "It isn’t your move."}), 409
            try:
                move = chess.Move.from_uci(str(datos.get("uci", "")))
            except ValueError:
                return jsonify({"error": "That move isn’t in a valid format."}), 400
            if move not in board.legal_moves:
                return jsonify({"error": "That move isn’t legal."}), 400
            board.push(move)
            turno_del_modelo(modo, temperatura)
            return estado()

    @app.post("/api/deshacer")
    def api_deshacer():
        _, modo, temperatura = opciones()
        with lock:
            board = partida["board"]
            if board.move_stack:
                board.pop()
                while board.move_stack and board.turn != partida["humano"]:
                    board.pop()                        # quita también la respuesta del modelo
            partida["analisis"] = None
            turno_del_modelo(modo, temperatura)        # si quedó su turno (p. ej. juegas negras)
            return estado()

    return app
