"""Pruebas del motor que no necesitan un modelo entrenado. Corre con:  pytest"""
import chess

from chess_gpt.engine import texto_partida


def test_texto_inicio():
    assert texto_partida(chess.Board()) == "\n1."


def test_texto_turno_negras():
    b = chess.Board()
    b.push_san("e4")
    assert texto_partida(b) == "\n1.e4 "


def test_texto_turno_blancas():
    b = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        b.push_san(san)
    assert texto_partida(b) == "\n1.e4 e5 2.Nf3 Nc6 3."


def test_texto_con_jaque_y_captura():
    b = chess.Board()
    for san in ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+"]:
        b.push_san(san)
    assert texto_partida(b) == "\n1.e4 d5 2.exd5 Qxd5 3.Nc3 Qe5+ 4."
