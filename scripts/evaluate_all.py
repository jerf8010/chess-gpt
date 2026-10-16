"""
scripts/evaluate_all.py — Evalúa todos los checkpoints con las MISMAS reglas y semillas,
guarda los resultados en resultados.json e imprime una tabla comparativa.

Uso (desde la raíz del proyecto):  python scripts/evaluate_all.py

Los resultados se guardan después de cada (experimento, semilla). Si lo vuelves a correr,
salta lo que ya está calculado. Para forzar todo de nuevo, borra resultados.json
o cambia VERSION_EVALUADOR.
"""
import json
import os
from collections import Counter

import numpy as np
import torch

from chess_gpt.model import cargar_modelo
from chess_gpt.rules import partidas_completas, primera_ilegal_detalle, contar_plies, causa

# ======================= CONFIGURACIÓN =======================
EXPERIMENTOS = {
    "exp0": "./models/chess_gpt.pt",
    "exp1": "./models/chess_gpt_exp1.pt",
    "exp2": "./models/chess_gpt_exp2.pt",
    "exp3": "./models/chess_gpt_exp3.pt",
    "exp4": "./models/chess_gpt_exp4.pt",
    "exp5": "./models/chess_gpt_exp5.pt",
}
SEMILLAS = [0, 1, 2]
N_SECUENCIAS = 100
MAX_NUEVOS = 1000
PARTIDA_LARGA = 20                  # una partida legal "cuenta" si tiene al menos estas medias jugadas
ARCHIVO = "./resultados.json"
VERSION_EVALUADOR = "v3-capturas"   # cámbiala si modificas las reglas de evaluacion.py


def elegir_dispositivo():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generar_partidas(model, chars, semilla, device):
    itos = dict(enumerate(chars))
    torch.manual_seed(semilla)
    contexto = torch.zeros((N_SECUENCIAS, 1), dtype=torch.long, device=device)   # '\n' en cada fila
    salida = model.generate(contexto, MAX_NUEVOS)
    partidas = []
    for fila in salida:
        texto = "".join(itos[i] for i in fila.tolist())
        partidas.extend(partidas_completas(texto))
    return partidas


def medir(partidas):
    errores = [primera_ilegal_detalle(p) for p in partidas]
    con_error = [e for e in errores if e is not None]
    plies = [e[0] for e in con_error]
    largos_legales = [contar_plies(p) for p, e in zip(partidas, errores) if e is None]
    causas = Counter(causa(jugada, board, err) for _, jugada, board, err in con_error)
    return {
        "n_partidas": len(partidas),
        "promedio": float(np.mean(plies)) if plies else None,
        "mediana": float(np.median(plies)) if plies else None,
        "maximo": int(max(plies)) if plies else None,
        "legales": len(largos_legales),
        "legales_largas": sum(1 for n in largos_legales if n >= PARTIDA_LARGA),
        "causas": dict(causas),
    }


def cargar_resultados():
    if os.path.exists(ARCHIVO):
        with open(ARCHIVO, encoding="utf-8") as f:
            datos = json.load(f)
        if datos.get("version") == VERSION_EVALUADOR:
            return datos
        print(f"⚠️  {ARCHIVO} es de otra versión del evaluador: lo recalculo todo")
    return {"version": VERSION_EVALUADOR, "experimentos": {}}


def guardar_resultados(datos):
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)


def resumir(por_semilla):
    """Combina los resultados de varias semillas."""
    promedios = [r["promedio"] for r in por_semilla if r["promedio"] is not None]
    medianas = [r["mediana"] for r in por_semilla if r["mediana"] is not None]
    causas = Counter()
    for r in por_semilla:
        causas.update(r["causas"])
    return {
        "n_partidas": sum(r["n_partidas"] for r in por_semilla),
        "promedio": np.mean(promedios),
        "promedio_std": np.std(promedios),
        "mediana": np.mean(medianas),
        "maximo": max(r["maximo"] for r in por_semilla if r["maximo"] is not None),
        "legales": sum(r["legales"] for r in por_semilla),
        "legales_largas": sum(r["legales_largas"] for r in por_semilla),
        "causas": causas,
    }


def main():
    device = elegir_dispositivo()
    print(f"usando: {device}\n")
    datos = cargar_resultados()

    for nombre, ruta in EXPERIMENTOS.items():
        if not os.path.exists(ruta):
            print(f"⚠️  {nombre}: no encontré {ruta} — lo salto")
            continue
        guardados = datos["experimentos"].setdefault(nombre, {})
        pendientes = [s for s in SEMILLAS if str(s) not in guardados]
        if not pendientes:
            print(f"{nombre}: ya evaluado ✓")
            continue

        model, chars = cargar_modelo(ruta, device)
        for semilla in pendientes:
            partidas = generar_partidas(model, chars, semilla, device)
            r = medir(partidas)
            guardados[str(semilla)] = r
            guardar_resultados(datos)                  # guardar en cada paso: si algo truena, no se pierde
            print(f"{nombre} · seed {semilla}: {r['n_partidas']} partidas, "
                  f"promedio {r['promedio']:.2f}, mediana {r['mediana']:.1f}")
        del model
        if device == "mps":
            torch.mps.empty_cache()

    # ======================= TABLA =======================
    print("\n" + "=" * 96)
    print(f"{'exp':5s} {'partidas':>8s} {'promedio':>15s} {'mediana':>8s} {'máx':>5s} "
          f"{'legales':>8s} {'≥' + str(PARTIDA_LARGA) + ' plies':>10s}   causa principal")
    print("-" * 96)
    for nombre in EXPERIMENTOS:
        por_semilla = [datos["experimentos"].get(nombre, {}).get(str(s)) for s in SEMILLAS]
        por_semilla = [r for r in por_semilla if r]
        if not por_semilla:
            continue
        t = resumir(por_semilla)
        total_err = sum(t["causas"].values())
        principal, k = t["causas"].most_common(1)[0] if total_err else ("—", 0)
        print(f"{nombre:5s} {t['n_partidas']:8d} {t['promedio']:8.2f} ± {t['promedio_std']:4.2f} "
              f"{t['mediana']:8.1f} {t['maximo']:5d} {t['legales']:8d} {t['legales_largas']:10d}   "
              f"{principal} ({100 * k / max(total_err, 1):.0f}%)")
    print("=" * 96)
    print(f"promedio ± desviación entre {len(SEMILLAS)} semillas · resultados en {ARCHIVO}")


if __name__ == "__main__":
    main()
