// Interfaz del tablero: dibuja el estado que manda el servidor y envía tus jugadas.
// Toda la lógica de ajedrez vive en Python (python-chess); aquí solo hay presentación.

const COLUMNAS = "abcdefgh";
const GLIFOS = { K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟" };

let estado = null;      // último estado recibido del servidor
let seleccion = null;   // casilla de origen elegida, p. ej. "e2"
let ocupado = false;    // esperando respuesta del servidor

const $ = (sel) => document.querySelector(sel);

// ------------------------------------------------------------------ servidor
async function pedir(ruta, cuerpo) {
  const opciones = cuerpo === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) };
  const r = await fetch(ruta, opciones);
  const datos = await r.json();
  if (!r.ok) throw new Error(datos.error || "Server error");
  return datos;
}

function preferencias() {
  return { modo: $("#modo").value, temperatura: parseFloat($("#temp").value) };
}

async function accion(ruta, extra = {}) {
  if (ocupado) return;
  ocupado = true;
  seleccion = null;
  mostrarError(null);
  pintar(true);
  try {
    estado = await pedir(ruta, { ...preferencias(), ...extra });
  } catch (e) {
    mostrarError(e.message);
  } finally {
    ocupado = false;
    pintar(false);
  }
}

// ------------------------------------------------------------------ interacción
function esMiTurno() {
  return estado && !estado.fin && !ocupado && estado.turno === estado.humano;
}

function alTocarCasilla(casilla) {
  if (!esMiTurno()) return;

  if (seleccion) {
    const opciones = estado.legales.filter((m) => m.slice(0, 2) === seleccion && m.slice(2, 4) === casilla);
    if (opciones.length) {
      // coronación: si hay varias (q, r, b, n), se elige dama
      const uci = opciones.find((m) => m.length === 4 || m.endsWith("q"));
      accion("/api/jugar", { uci });
      return;
    }
  }
  const tieneJugadas = estado.legales.some((m) => m.slice(0, 2) === casilla);
  seleccion = tieneJugadas && casilla !== seleccion ? casilla : null;
  pintar(false);
}

// ------------------------------------------------------------------ dibujo
function pintar(pensando) {
  if (!estado) return;
  pintarTablero();
  pintarEstado(pensando);
  pintarAnalisis();
  pintarJugadas();
  document.querySelectorAll("button").forEach((b) => (b.disabled = ocupado));
  $("#modelo").textContent = `${estado.modelo} · ${estado.dispositivo}`;
}

function pintarTablero() {
  const tablero = $("#tablero");
  tablero.innerHTML = "";
  const blancasAbajo = estado.humano === "w";
  const destinos = seleccion
    ? estado.legales.filter((m) => m.slice(0, 2) === seleccion).map((m) => m.slice(2, 4))
    : [];
  const ultima = estado.ultima ? [estado.ultima.slice(0, 2), estado.ultima.slice(2, 4)] : [];

  for (let fila = 0; fila < 8; fila++) {
    for (let col = 0; col < 8; col++) {
      const rank = blancasAbajo ? 8 - fila : fila + 1;
      const file = blancasAbajo ? col : 7 - col;
      const casilla = COLUMNAS[file] + rank;
      const pieza = estado.piezas[casilla];

      const el = document.createElement("button");
      el.type = "button";
      el.className = "casilla " + ((file + rank) % 2 === 1 ? "oscura" : "clara");
      if (ultima.includes(casilla)) el.classList.add("ultima");
      if (casilla === seleccion) el.classList.add("sel");
      if (casilla === estado.jaque_en) el.classList.add("jaque");
      if (destinos.includes(casilla)) el.classList.add("destino", ...(pieza ? ["ocupada"] : []));
      el.setAttribute("aria-label", casilla + (pieza ? " " + nombrePieza(pieza) : ""));
      el.addEventListener("click", () => alTocarCasilla(casilla));

      if (pieza) {
        const span = document.createElement("span");
        span.className = "pieza " + (pieza === pieza.toUpperCase() ? "blanca" : "negra");
        span.textContent = GLIFOS[pieza.toUpperCase()];
        el.appendChild(span);
      }
      if (col === 0) el.appendChild(coordenada("fila-n", rank));
      if (fila === 7) el.appendChild(coordenada("col-l", COLUMNAS[file]));
      tablero.appendChild(el);
    }
  }
}

function coordenada(clase, texto) {
  const s = document.createElement("span");
  s.className = "coord " + clase;
  s.textContent = texto;
  return s;
}

function nombrePieza(p) {
  const nombres = { K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight", P: "pawn" };
  return `${p === p.toUpperCase() ? "white" : "black"} ${nombres[p.toUpperCase()]}`;
}

function pintarEstado(pensando) {
  const el = $("#estado");
  el.classList.toggle("pensando", pensando);
  if (pensando) {
    el.textContent = "The model is thinking…";
  } else if (estado.fin) {
    const ganador = { "1-0": "White wins", "0-1": "Black wins", "1/2-1/2": "Draw" }[estado.resultado];
    el.textContent = `${ganador} by ${estado.motivo} (${estado.resultado})`;
  } else if (estado.turno === estado.humano) {
    el.textContent = estado.jaque_en ? "Your move — you’re in check!" : "Your move";
  } else {
    el.textContent = "Model’s move";
  }
}

function pintarAnalisis() {
  const a = estado.analisis;
  const caja = $("#analisis");
  if (!a) {
    caja.innerHTML = '<p class="vacio">Once the model moves, you’ll see its top 5 choices here and what it would have written on its own.</p>';
    return;
  }
  const maximo = Math.max(...a.top.map((t) => t.prob));
  const barras = a.top.map((t) => `
      <div class="barra ${t.san === a.jugada ? "elegida" : ""}">
        <span>${t.san}</span>
        <span class="pista"><span class="relleno" style="width:${(100 * t.prob) / maximo}%"></span></span>
        <span class="pct">${(100 * t.prob).toFixed(1)}%</span>
      </div>`).join("");

  const libre = a.libre.legal
    ? `<span class="ok">legal ✓</span>`
    : `<span class="mal">illegal ✗ — ${a.libre.causa}</span>`;

  caja.innerHTML = `
    <div class="barras">${barras}</div>
    <p class="nota">Played <strong>${a.jugada}</strong> (#${a.ranking} of ${a.n_legales} legal moves, ${(100 * a.prob).toFixed(1)}%)</p>
    <p class="libre">On its own it would have written <code>${a.libre.san || "(nothing)"}</code> → ${libre}</p>`;
}

function pintarJugadas() {
  const lista = $("#jugadas");
  lista.innerHTML = "";
  const j = estado.jugadas;
  for (let i = 0; i < j.length; i += 2) {
    const li = document.createElement("li");
    const blanca = document.createElement("span");
    blanca.textContent = j[i];
    li.appendChild(blanca);
    if (j[i + 1]) {
      li.append(" ");
      const negra = document.createElement("span");
      negra.textContent = j[i + 1];
      li.appendChild(negra);
    }
    if (i + 2 >= j.length) li.lastChild.classList.add("ult");
    lista.appendChild(li);
  }
  lista.scrollTop = lista.scrollHeight;
}

function mostrarError(mensaje) {
  const el = $("#error");
  el.hidden = !mensaje;
  el.textContent = mensaje || "";
}

// ------------------------------------------------------------------ inicio
$("#nueva-w").addEventListener("click", () => accion("/api/nueva", { color: "w" }));
$("#nueva-b").addEventListener("click", () => accion("/api/nueva", { color: "b" }));
$("#deshacer").addEventListener("click", () => accion("/api/deshacer"));
$("#temp").addEventListener("input", (e) => ($("#temp-val").textContent = parseFloat(e.target.value).toFixed(1)));

pedir("/api/estado")
  .then((datos) => { estado = datos; pintar(false); })
  .catch((e) => mostrarError("Couldn’t reach the server: " + e.message));
