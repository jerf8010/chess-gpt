"""Pruebas de las reglas de evaluación. Corre con:  pytest"""
import chess

from chess_gpt.rules import (contar_plies, limpiar_token, partidas_completas,
                             primera_ilegal, primera_ilegal_detalle)

PARTIDA_NXC7 = ("1.e4 g6 2.d4 Bg7 3.f3 d5 4.exd5 c6 5.dxc6 Nxc6 "
                "6.Ne2 Nf6 7.Nbc3 Bf5 8.Nb5 O-O 9.Nxc7")


def test_mate_real_es_legal():
    assert primera_ilegal("1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7# 1-0") is None


def test_peon_hacia_atras():
    assert primera_ilegal("1.e4 d5 2.c4 d6") == 4


def test_alfil_imposible():
    assert primera_ilegal("1.e4 c6 2.d4 Nf6 3.Nc3 d6 4.Nf3 g6 5.h3 Bg7 6.Bxe7") == 11


def test_enroque_imposible():
    assert primera_ilegal("1.e4 e5 2.Ke2 Ke7 3.O-O") == 5


def test_token_numerico_sin_punto():
    assert primera_ilegal("1.e4 e5 2.Nf3 12 Nc6") == 4


def test_mate_falso():
    assert primera_ilegal("1.e4 e5 2.Nf3 d6 3.Bc4 Bg4 4.d3 Bf5# 0-1") == 8


def test_captura_falsa():
    assert primera_ilegal(PARTIDA_NXC7) == 17
    _, _, board, error = primera_ilegal_detalle(PARTIDA_NXC7)
    assert error == "captura falsa"
    assert board.turn == chess.WHITE        # el tablero es el de ANTES de 9.Nxc7


def test_captura_sin_x():
    assert primera_ilegal("1.e4 d5 2.Bb5+ c6 3.Bc6") == 5


def test_utilidades():
    assert limpiar_token("12.Nf3") == "Nf3"
    assert limpiar_token("12") == "12"
    assert contar_plies("1.e4 e5 2.Nf3 1-0") == 3


def test_partidas_completas():
    assert partidas_completas("\n1.e4 e5 2.Nf3 1-0\n1.d4 d5 2.c4 0-1\n1.e4 c5 2.N") == \
        ["1.e4 e5 2.Nf3 1-0", "1.d4 d5 2.c4 0-1"]
    assert partidas_completas("\n1.d4 e5 2.Nc3 d6 28.a4 Bf") == []
    assert partidas_completas("\n1.e4 e5 1-0\n") == ["1.e4 e5 1-0"]
