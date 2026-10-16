"""
chess_gpt.engine — Convierte el GPT de texto en un jugador de ajedrez.

Idea central: el modelo solo sabe continuar texto como '\\n1.e4 e5 2.'. El motor:
  1. Traduce la posición actual a ese formato (texto_partida).
  2. Puntúa TODAS las jugadas legales: P('Nf3 ' | texto) para cada una, en una sola pasada.
  3. Elige la más probable o muestrea entre ellas (nunca juega algo ilegal).
  4. Además, deja que el modelo escriba "libremente" para mostrar qué habría hecho sin ayuda.
"""
import chess
import torch
from torch.nn import functional as F

from .model import cargar_modelo
from .rules import causa

MAX_CARACTERES_JUGADA = 12      # 'exd8=Q+' tiene 7; esto sobra
TEMPERATURA_MINIMA = 0.05


def elegir_dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def texto_partida(board):
    """Texto en el formato del dataset, terminando justo donde va la siguiente jugada.

    inicio            → '\\n1.'
    tras 1.e4         → '\\n1.e4 '
    tras 1.e4 e5      → '\\n1.e4 e5 2.'
    """
    replay = chess.Board()
    partes = []
    for move in board.move_stack:
        san = replay.san(move)
        partes.append(f"{replay.fullmove_number}.{san}" if replay.turn == chess.WHITE else san)
        replay.push(move)
    texto = "\n" + " ".join(partes) + (" " if partes else "")
    if board.turn == chess.WHITE:
        texto += f"{board.fullmove_number}."
    return texto


class Motor:
    """Jugador de ajedrez basado en un checkpoint de chess_gpt."""

    def __init__(self, ruta_checkpoint, device=None):
        self.device = device or elegir_dispositivo()
        self.model, chars = cargar_modelo(ruta_checkpoint, self.device)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))
        self.block_size = self.model.block_size
        self.nombre = ruta_checkpoint

    def encode(self, texto):
        return [self.stoi[c] for c in texto]

    # ------------------------------------------------------------------ puntuar
    @torch.no_grad()
    def puntuar(self, board, temperatura=1.0):
        """Probabilidad de cada jugada legal según el modelo, de mayor a menor.
        Regresa una lista de dicts {'san', 'uci', 'prob', 'logp'}."""
        legales = list(board.legal_moves)
        sans = [board.san(m) for m in legales]
        candidatos = [self.encode(s + " ") for s in sans]       # el espacio marca el fin de la jugada

        max_cand = max(len(c) for c in candidatos)
        prefijo = self.encode(texto_partida(board))[-(self.block_size - max_cand):]
        P = len(prefijo)

        # una fila por jugada candidata: prefijo + candidata (relleno a la derecha)
        largo = P + max_cand
        x = torch.zeros((len(candidatos), largo), dtype=torch.long)
        for i, c in enumerate(candidatos):
            x[i, :P + len(c)] = torch.tensor(prefijo + c)

        logits, _ = self.model(x.to(self.device))
        logp = F.log_softmax(logits.float(), dim=-1).cpu()        # (n_jugadas, largo, vocab)

        # log P(candidata) = suma de log P(cada carácter | todo lo anterior)
        puntajes = []
        for i, c in enumerate(candidatos):
            posiciones = torch.arange(P - 1, P - 1 + len(c))      # el logit en t predice el carácter t+1
            puntajes.append(logp[i, posiciones, torch.tensor(c)].sum().item())

        t = max(temperatura, TEMPERATURA_MINIMA)
        probs = F.softmax(torch.tensor(puntajes) / t, dim=0).tolist()
        filas = [
            {"san": s, "uci": m.uci(), "prob": p, "logp": lp}
            for s, m, p, lp in zip(sans, legales, probs, puntajes)
        ]
        return sorted(filas, key=lambda f: f["prob"], reverse=True)

    # ------------------------------------------------------------------ libre
    @torch.no_grad()
    def escribir_libre(self, board, temperatura=1.0):
        """Deja que el modelo escriba la jugada carácter por carácter, sin restricciones."""
        ctx = torch.tensor([self.encode(texto_partida(board))], device=self.device)
        t = max(temperatura, TEMPERATURA_MINIMA)
        escrito = []
        for _ in range(MAX_CARACTERES_JUGADA):
            logits, _ = self.model(ctx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :] / t, dim=-1)
            siguiente = torch.multinomial(probs, num_samples=1)
            caracter = self.itos[siguiente.item()]
            if caracter in (" ", "\n"):
                break
            escrito.append(caracter)
            ctx = torch.cat((ctx, siguiente), dim=1)
        san = "".join(escrito)

        try:
            board.parse_san(san)
            return {"san": san, "legal": True, "causa": None}
        except ValueError as error:
            return {"san": san, "legal": False, "causa": causa(san, board, error)}

    # ------------------------------------------------------------------ jugar
    def analizar(self, board, modo="mejor", temperatura=1.0, top=5):
        """Elige la jugada del modelo. modo: 'mejor' (la más probable) o 'muestrear'."""
        libre = self.escribir_libre(board, temperatura)
        filas = self.puntuar(board, temperatura)
        if modo == "muestrear":
            idx = torch.multinomial(torch.tensor([f["prob"] for f in filas]), 1).item()
        else:
            idx = 0
        elegida = filas[idx]
        return {
            "jugada": elegida["san"],
            "uci": elegida["uci"],
            "prob": elegida["prob"],
            "ranking": idx + 1,
            "n_legales": len(filas),
            "top": [{"san": f["san"], "prob": f["prob"]} for f in filas[:top]],
            "libre": libre,
            "modo": modo,
        }
