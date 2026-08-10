# Plan: Multi-Kriterien Pre-Filter fuer Quad-Domain-Partition

## Ziel
Den aktuellen `MeshCheck` (nur Flaechenvergleich) durch einen hierarchischen
Validator ersetzen, der **vor** dem teuren `Transfinite_Interpolation`-Schritt
auf der Block-Topologie laeuft. Verwendet Metriken aus
`literature/quad_domain_partition.pdf` (Kowalski et al. 2015).

## Architektur
```
MeshGenerator -> FrameField -> StreamlineGenerator -> StreamlineSimplificator
                                                                 |
                                                  [NEU: QuadPartitionValidator]
                                                                 |
                                                                 v
                                                  (valide? -> weiter)
                                                                 |
                                                                 v
                                                    Transfinite_Interpolation
                                                                 |
                                                                 v
                                                        quad_mesh -> MeshCheck
```

## Neue Datei: `tools/quad_partition_validator.py`
Klasse `QuadPartitionValidator(blocked_mesh, tri_mesh, frame_field, tol_config=None)`.

### Public API (Python-Stubs)
```python
class QuadPartitionValidator:
    def __init__(self, blocked_mesh: Data, tri_mesh: Data,
                 frame_field: torch.Tensor,
                 tol_config: Optional[dict] = None,
                 strict: bool = False): ...
    def is_valid(self) -> bool: ...
    def quality_score(self) -> Dict[str, float]: ...
    def passes_soft_thresholds(self, tol: dict) -> bool: ...
    def diagnostics(self) -> List[str]: ...
```
- `is_valid() -> bool`              # Hard-Checks (boolesches AND)
- `quality_score() -> dict`         # Soft-Metriken
- `passes_soft_thresholds(tol)`     # optional, separat nutzbar
- `diagnostics() -> List[str]`      # menschenlesbarer Failure-Report

### Hard-Checks (boolesche Filter)
1. **Topologie**: Euler-Charakteristik des Block-Meshes muss `V - E + F == 0`
   sein (planare Domain mit einem Loch = Airfoil). *Hinweis*: Wiederverwende
   bestehende Logik aus `tools/singularity_detector.py` (Euler-Charakteristik
   & Poincaré-Index), statt diese neu zu implementieren.
2. **Block-Valenz**: alle inneren Vertices haben Valenz 4 (oder 3 an
   Singularitaeten, 2 an Boundary-Corners). Nutze dafür die Kanten-Adjazenz
   aus `blocked_mesh.edge_index`.
3. **Element-Validitaet**: kein Quad mit `scaled_jacobian < 0`
   (invertiert) oder `min_angle < 5 deg` / `max_angle > 175 deg` oder
   **Edge-Length-Ratio** (max_k |e_k| / min_k |e_k|) > 10.
4. **Boundary-Match**: Vergleiche Boundary-Knoten des Tri-Meshes mit dem
   Boundary-Polygon des Block-Meshes mittels **Hausdorff-Distanz** oder
   **Punkt-auf-Polygon-Test** (Toleranz 1e-6). Das Block-Mesh muss alle
   Boundary-Konturen (Rechteck + Airfoil) korrekt umschliessen.

### Soft-Metriken (geloggt + Score)
1. **Scaled Jacobian** pro Quad, an allen 4 Ecken:
   `J_k = det( [p_{k+1}-p_{k-1}, p_{k+2}-p_k}] ) / (|a|*|b|)`
   -> min, mean ueber alle Quads
2. **Interior Angles** pro Quad, vektorisiert via `atan2(cross, dot)`
   -> globales min, max
3. **Edge-Length-Ratio** pro Quad (max / min) -> globales max
4. **Singularity Efficiency** = `Smin / n_actual`
   (siehe Kowalski Gl. 20)

### Default-Toleranzen (Literatur)
```python
DEFAULT_TOLERANCES = {
    "scaled_jacobian_min":  0.3,    # Cubit acceptable
    "scaled_jacobian_mean": 0.85,   # Kowalski mean ~0.99
    "min_interior_angle":   45.0,   # Cubit acceptable (45-90 ideal)
    "max_interior_angle":   135.0,  # Cubit acceptable (90-135 ideal)
    "edge_length_ratio_max": 10.0,  # Verhindert extrem gestreckte Quads
    "singularity_efficiency_min": 0.5,
}
```
*Hinweis*: Toleranzen sind als `dict` konstruktor-seitig ueberschreibbar.
Empfohlen: In `data_generator.py` / `test.py` via `argparse` oder einer
optionalen `.json`-Config laden, um Hyperparameter-Tuning zu erleichtern.

## Smin-Berechnung (Kowalski Gl. 20)
```
Smin = |theta1(g0h) + theta2(g0h) - sum_i(theta1(gih) + theta2(gih))|
```
- `g0h`: piecewise-linear outer boundary (Rechteck)
- `gih`: inner boundaries (NACA-Splines als Edge-Liste)
- `theta1`: Integral von `dphi > 0` entlang der Kurve, gewichtet mit
  Edge-Laenge
- `theta2`: dasselbe fuer `dphi < 0`
- **Kritisch**: `dphi` ist *nicht* direkt aus `mesh.frame_field` interpolierbar,
  weil das Cross-Field 4-fach-ambigu ist (Period-Jumps an Kanten).
  Es muss zuerst ein **sprung-freier (smooth) Pfad** entlang der Boundary
  berechnet werden (Branch-Alignment / unwrapping). Verwende dafuer die
  bestehende Logik aus `frame_field.py` (z.B. `get_smooth_representation()`
  oder aequivalente Unwrapping-Methode), bevor die Winkel-Differenzen
  `dphi` entlang der Boundary integriert werden.

Inputs dafuer existieren bereits:
- `tri_mesh.x[:, 0:2]` + `tri_mesh.edge_attr == 1` (Boundary-Edges)
- `tri_mesh.frame_field` (cross-field Vektoren)
- `tri_mesh.singularities_coords` (Innen-Singularitaeten)

## Integration & Graceful-Degradation
- `data_generator.py:29` -> Validator-Aufruf **vor** Transfinite:
  `validator = QuadPartitionValidator(blocked_mesh, tri_mesh, frame_field, strict=False)`
  - **Phase 1 (Einfuehrung)**: `strict=False` (Default). Hard-Checks
    loggen Warnungen via `diagnostics()`, produzieren aber **keinen**
    Datenverlust. Soft-Metriken werden geloggt, um die Ablehnungsrate
    zu kalibrieren.
  - **Phase 2 (nach Kalibrierung)**: `strict=True` aktivierbar per
    `argparse`-Flag oder Config. Dann: `if not validator.is_valid(): return None`
- `test.py:73` analog, dort ggf. `strict=True` erwuenscht
- `quadmesh_generator.py` optional gleiche Integration
- `MeshCheck` (Post-Hoc) bleibt als Sanity-Check erhalten
- `tools/__init__.py`: Import hinzufuegen

## Was NICHT geaendert wird
- `check_mesh.py` (Post-Hoc-Flaechencheck bleibt)
- `frame_field.py`, `singularity_detector.py` (Read-Only Datenquellen)
- `transfinite_interpolation.py` (erhaelt saubere Inputs)

## Geaenderte Dateien
| Datei | Aenderung |
|-------|-----------|
| `tools/quad_partition_validator.py` | NEU - gesamte Logik |
| `tools/__init__.py` | Import der neuen Klasse |
| `data_generator.py` | Validator-Aufruf + Diagnose-Logging |
| `test.py` | gleiche Integration |
| `quadmesh_generator.py` | optional analog |
| `tests/test_quad_partition_validator.py` | NEU - Unit-Tests fuer Validator |

## Unit-Tests (neu)
Erstelle `tests/test_quad_partition_validator.py` mit folgenden Faellen:
- **Perfektes Rechteck-Quad-Mesh**: alle Hard-Checks passen, Soft-Metriken
  innerhalb akzeptabler Range.
- **Invertiertes Quad**: `is_valid()` muss `False` liefern (scaled_jacobian < 0).
- **Falsche Euler-Charakteristik**: z.B. zusaetzlicher isolierter Knoten ->
  `is_valid()` muss `False` liefern.
- **Extremes Aspect-Ratio-Quad**: Edge-Length-Ratio > 10 -> `is_valid()`
  muss `False` liefern.
- **Boundary-Mismatch**: Boundary-Knoten verschoben -> `is_valid()` muss
  `False` liefern.
- **Soft-Threshold-Logging**: bei `strict=False` muss `diagnostics()`
  Warnungen enthalten, aber `is_valid()` dennoch `True` liefern (solange
  keine Hard-Fail vorliegt).

## Verifikations-Strategie
- Hard-Checks (Topologie, Inversion) boolesch filtern
- Soft-Metriken geloggt, optional als Threshold-Filter nutzbar
- `diagnostics()` gibt bei Failure konkrete Info: betroffene Knoten-IDs,
  konkrete Metrik-Werte

## Offene Punkte (vor Implementierung zu klaeren)
1. Soll `quality_score()` als Tensor-Attribute am `blocked_mesh`
   persistiert werden?
2. Soll eine Per-Quad-Heatmap-Visualisierung mitimplementiert werden,
   oder nur numerische Reports?
3. `strict`-Mode: Default `False` im Generator akzeptabel, oder soll
   von Anfang an `True` sein?
4. Soll die Smin-Berechnung einen Fallback haben, falls das
   Branch-Alignment in `frame_field.py` nicht direkt exportierbar ist
   (z.B. separate Unwrapping-Funktion im Validator)?
