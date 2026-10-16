"""
scripts/play.py — Juega contra tu modelo en el navegador.

Uso (desde la raíz del proyecto):
    python scripts/play.py
    python scripts/play.py --checkpoint models/chess_gpt_exp3.pt --port 8001
"""
import argparse
import os

from chess_gpt.engine import Motor
from chess_gpt.web import create_app


def main():
    parser = argparse.ArgumentParser(description="Play chess against chess_gpt in your browser.")
    parser.add_argument("--checkpoint", default="models/chess_gpt_exp5.pt", help="path to the .pt model")
    parser.add_argument("--port", type=int, default=8000, help="local port (on macOS avoid 5000: AirPlay uses it)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Couldn’t find {args.checkpoint}. Are you in the project root?")

    print(f"Loading {args.checkpoint} …")
    motor = Motor(args.checkpoint)
    print(f"Model ready on {motor.device}. Open  http://127.0.0.1:{args.port}  in your browser (Ctrl+C to quit).")
    create_app(motor).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
