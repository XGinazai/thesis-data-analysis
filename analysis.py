"""
🎯 Formato de logs: 
[2025-12-30T00:12:53.921546] OCR: initialize - finished
[timestamp ISO 8601] componente: evento - detalles

🎯 Formato del csv (output):
| id_prueba | timestamp | modo | tipo de evento | escenario | distancia_m | iluminacion |
objeto_verdad | objeto_predicho | confianza | t_total_ms | acierto | notas |

💾 Origen de los datos:
id_prueba -> auto incremental
timestamp -> tiempo en formato ISO 8601
modo -> MOBILE o LOCAL
tipo de evento -> asignado manualmente según el evento ocurrido
escenario -> 1 - 4
distancia_m -> distancia en metros
iluminacion -> LED o NATURAL
objecto_verdad -> etiqueta del objeto real (si aplica)
objeto_predicho -> etiqueta del objeto predicho por el modelo
confianza -> confianza del modelo en la predicción (0.0 - 1.0) (si aplica)
t_total_ms -> tiempo total de procesamiento en milisegundos ($TA - TB = duración de B$)
acierto -> 1 si la predicción es correcta, 0 si es incorrecta (si aplica al evento)

🎯 Ejemplo de flujo regular:
[2025-12-29T18:29:25.428165] ESP32: capture - starting
[2025-12-29T18:29:25.646733] ESP32: capture - finished (17391 bytes)

[2025-12-29T18:29:25.653748] SCENE: detectScene - starting
[2025-12-29T18:29:26.045672] SCENE: preprocessing - resizing to 520

[2025-12-29T18:29:26.338148] SCENE: running inference
[2025-12-29T18:29:32.684727] SCENE: finished inference

[2025-12-29T18:29:32.752895] SCENE: classFrequency - error: no known object/class was found
[2025-12-29T18:29:32.765650] SCENE: result - unknown with confidence 0.000

[2025-12-29T18:29:32.767799] SCENE: naturalDescription - No puedo identificar nada en concreto.

[2025-12-29T18:29:32.771370] TTS: converting output to speech - "No puedo identificar nada en concreto."
[2025-12-29T18:29:32.786957] TTS: converted output to speech

🎯 approach mas sencillo:
tiempo del evento actual - tiempo del evento anterior = duración del evento anterior
$TA - TB = duración de B$
"""
from __future__ import annotations

import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


def format_duration_ms(ms: int) -> str:
	"""Convierte milisegundos a 'HH:MM:SS.mmm'"""
	if ms is None:
		return ""
	seconds, milli = divmod(int(ms), 1000)
	mins, sec = divmod(seconds, 60)
	hours, mins = divmod(mins, 60)
	return f"{hours:02d}:{mins:02d}:{sec:02d}.{milli:03d}"


LINE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s*(?P<component>[^:]+):\s*(?P<event>[^-]+?)(?:-\s*(?P<details>.*))?$")
CONF_RE = re.compile(r"with confidence\s+(?P<conf>[0-9]*\.?[0-9]+)", re.IGNORECASE)
DIST_RE = re.compile(r"(?:distance|distancia)\D*(?P<dist>[0-9]+(?:\.[0-9]+)?)\s*m", re.IGNORECASE)


def parse_log_lines(lines: List[str]) -> List[Dict]:
	"""Parsea las líneas del log y devuelve una lista de eventos estructurados."""
	events = []
	for ln in lines:
		ln = ln.strip()
		if not ln:
			continue
		m = LINE_RE.match(ln)
		if not m:
			# línea no conforme; guardar como nota
			events.append({
				"ts": None,
				"component": None,
				"event": None,
				"details": ln,
				"raw": ln,
			})
			continue
		ts_s = m.group("ts")
		try:
			ts = datetime.fromisoformat(ts_s)
		except Exception:
			ts = None
		details_group = m.group("details") or ""
		print(f"Event: {m.group('event').strip()}, Details: {details_group.strip()}")
		events.append({
			"ts": ts,
			"component": m.group("component").strip(),
			"event": m.group("event").strip(),
			"details": details_group.strip(),
			"raw": ln,
		})
	return events

def find_ground_truth(lines: List[str]) -> Optional[str]:
	for ln in lines:
		if "ground truth" in ln.lower() or "objeto verdad" in ln.lower() or "objeto_verdad" in ln.lower():
			# extraer última palabra como etiqueta probable
			parts = ln.split()
			if parts:
				return parts[-1].strip('"')
	return None


def compute_inference_duration(events: List[Dict]) -> None:
	"""Asigna en cada evento la duración del evento previo según TA - TB = duración de B.

	Además, si se encuentra un par explícito 'running inference' -> 'finished inference',
	se asigna esa duración al evento 'running inference'. La función modifica la lista
	`events` in-place y no devuelve valor.
	"""
	# Inicializar
	for ev in events:
		ev.pop("duration_ms", None)
		ev.pop("duration_hms", None)

	if not events:
		return None

	# Primera regla solicitada: asignar duración por evento como current_ts - previous_ts
	# Para el primer evento asignar 0 ms.
	first_ts = events[0].get("ts")
	if first_ts:
		events[0]["duration_ms"] = 0
		events[0]["duration_hms"] = format_duration_ms(0)
	else:
		events[0]["duration_ms"] = None
		events[0]["duration_hms"] = None

	for i in range(1, len(events)):
		prev = events[i - 1]
		cur = events[i]
		ts_prev = prev.get("ts")
		ts_cur = cur.get("ts")
		# Asignar la duración al evento actual como (current - previous)
		if ts_prev and ts_cur:
			try:
				ms = int((ts_cur - ts_prev).total_seconds() * 1000)
				cur["duration_ms"] = ms
				cur["duration_hms"] = format_duration_ms(ms)
			except Exception:
				cur["duration_ms"] = None
				cur["duration_hms"] = None
		else:
			cur["duration_ms"] = None
			cur["duration_hms"] = None

	return None


def process_file(path: Path, escenario: int, modo: str = "MOBILE") -> List[Dict]:
	text = path.read_text(encoding="utf-8", errors="replace")
	lines = text.splitlines()
	events = parse_log_lines(lines)

	ilum = "LED"
	truth = find_ground_truth(lines)
	# compute_inference_duration ahora asigna duration_ms en cada evento (in-place)
	compute_inference_duration(events)

	rows = []
	# Buscar eventos 'result' como instancias de predicción
	for idx, ev in enumerate(events):
		if not ev.get("event"):
			continue
		# if ev.get("event").strip().lower() == "result":
		if True:
			ts = ev.get("ts")
			details = ev.get("details", "")
			# extraer label y confianza
			conf_m = CONF_RE.search(details)
			conf = None
			if conf_m:
				try:
					conf = float(conf_m.group("conf"))
				except Exception:
					conf = None
			# label: texto antes de 'with confidence' o el primer token
			label = details
			if "with confidence" in details.lower():
				label = re.split(r"with confidence", details, flags=re.IGNORECASE)[0].strip()
			# limpiar label palabras como 'unknown'
			label = label.strip()

			note = ""
			# si el detalle contiene 'error' agregar nota
			if "error" in details.lower():
				note = details

			row = {
				"timestamp": ts.isoformat() if ts else "",
				"modo": modo,
				"tipo de evento": ev.get("event").strip(),
				"escenario": escenario,
				"distancia_m": "",
				"iluminacion": ilum or "",
				"objeto_verdad": truth or "",
				"objeto_predicho": label,
				"confianza": f"{conf:.3f}" if conf is not None else "",
				"t_total_ms": ev.get("duration_hms") if ev.get("duration_hms") is not None else "",
				"acierto": (1 if truth and label and truth.lower() == label.lower() else 0) if truth else "",
				"notas": note,
			}
			rows.append(row)
	# si no se encontraron 'result' crear una entrada resumida
	if not rows:
		# usar último timestamp del archivo
		last_ts = None
		for ev in reversed(events):
			if ev.get("ts"):
				last_ts = ev.get("ts")
				break
		# tomar la última duración calculada (si existe)
		last_dh = ""
		for ev in reversed(events):
			if ev.get("duration_hms"):
				last_dh = ev.get("duration_hms")
				break
		rows.append({
			"timestamp": last_ts.isoformat() if last_ts else "",
			"modo": modo,
			"tipo de evento": "no_result",
			"escenario": escenario,
			"distancia_m": "",
			"iluminacion": ilum or "",
			"objeto_verdad": truth or "",
			"objeto_predicho": "",
			"confianza": "",
			"t_total_ms": last_dh or "",
			"acierto": "",
			"notas": "no result lines found",
		})
	return rows


def main(root: Optional[Path] = None) -> None:
	if root is None:
		root = Path(__file__).parent

	out_path = root / "analysis.csv"
	# header for CSV (exclude id_prueba here; it's added separately)
	header = [
		"timestamp",
		"modo",
		"tipo de evento",
		"escenario",
		"distancia_m",
		"iluminacion",
		"objeto_verdad",
		"objeto_predicho",
		"confianza",
		"t_total_ms",
		"acierto",
		"notas",
	]

	id_counter = 1
	rows_to_write = []

	for i in range(1, 5):
		candidate_dirs = [root / f"escenario {i}", root / "logs" / f"escenario {i}"]
		for dirp in candidate_dirs:
			if not dirp.exists() or not dirp.is_dir():
				continue
			for f in sorted(dirp.iterdir()):
				if not f.is_file():
					continue
				if not f.name.startswith("MOBILE"):
					continue
			try:
				file_rows = process_file(f, escenario=i, modo="MOBILE")
			except Exception as e:
				file_rows = [
					{
						"timestamp": "",
						"modo": "MOBILE",
						"tipo de evento": "parse_error",
						"escenario": i,
						"distancia_m": "",
						"iluminacion": "",
						"objeto_verdad": "",
						"objeto_predicho": "",
						"confianza": "",
						"t_total_ms": "",
						"acierto": "",
						"notas": f"parse error: {e}",
					}
				]
			for r in file_rows:
				r_out = {
					"id_prueba": id_counter,
					**r,
				}
				rows_to_write.append(r_out)
				id_counter += 1

	# construir un DataFrame y ordenar por timestamp (más antiguo -> más nuevo)
	df = pd.DataFrame(rows_to_write)
	cols = ["id_prueba"] + header
	# parsear timestamp y ordenar (NaT al final)
	if "timestamp" in df.columns:
		df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
		df = df.sort_values(by=["timestamp_parsed"], na_position="last", kind="mergesort")
		df = df.reset_index(drop=True)
		df["id_prueba"] = df.index + 1
		df = df.drop(columns=["timestamp_parsed"])
	else:
		df = df.reset_index(drop=True)
		df["id_prueba"] = df.index + 1

	# asegurar el orden de columnas al escribir
	cols = [c for c in cols if c in df.columns]

	try:
		target = out_path
		df.to_csv(target, columns=cols, index=False, encoding="utf-8")
	except PermissionError:
		fallback = root / "analysis_out.csv"
		df.to_csv(fallback, columns=cols, index=False, encoding="utf-8")
		target = fallback

	print(f"Wrote {len(df)} rows to {target}")


if __name__ == "__main__":
	main()