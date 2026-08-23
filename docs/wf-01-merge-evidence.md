# WF-01 Evidenzdokument: Diff-Analyse main vs blade-known-reinsertion

**Scope:** Structured analysis only. NO merge/port decision.
**Refs:** `main=72a0044`, `blade-known-reinsertion=790b195`, merge-base=`458a4b0`.
**Date:** 2026-08-23

---

## 1. Diff-Statistik

### Gesamt

| Metrik | Wert |
|--------|------|
| Commits auf `blade-known-reinsertion` seit Merge-Base | 103 |
| Commits auf `main` seit Merge-Base | 2 |
| Geaenderte Dateien | 866 |
| Zeilen hinzugefuegt | 51,303 |
| Zeilen entfernt | 2,357 |
| Netto-Zuwachs | +48,946 |

### Aufschluesselung nach Dateityp

| Kategorie | Dateien | Insertions | Deletions |
|-----------|---------|------------|-----------|
| Python-Quellcode | 71 | 12,298 | 1,944 |
| Nicht-Code (Runs, Figures, Logs, Docs, DBs, Lockfiles, Meshes) | 795 | 39,005 | 413 |

### Verzeichnis-Schwerpunkte (`dirstat=files,0`)

- `runs/` (Optuna-Sweep-Ergebnisse, Configs, Metrics): ~70% der Dateien
- `figures/` (Visualisierungen, E2E-Galerien): ~3.5%
- `slurm_logs/` (Cluster-Job-Outputs): ~3.2%
- `docs/ho_quad_transformer/` (Dokumentation): ~0.9%
- Python-Quellcode (Wurzelverzeichnis): ~8%

### Kernaussage

Der grosste Teil des Diff (91.7% der Dateien, 76% der Zeilen) sind Artefakte: Trainings-Runs, Bilder, Logs, Sweep-DBs. Der eigentliche Code-Unterschied betraegt ~71 Python-Dateien mit ~12k Insertions / ~2k Deletions.

---

## 2. Feature-Bloecke auf `blade-known-reinsertion`

Die 103 Commits lassen sich in thematische Bloecke gliedern. `feat:`-Commits sind die Traeger der neuen Funktionalitaet.

### 2.1 Plan-B E2E-Pipeline (Kernfeature)

| Commit | Message | Bedeutung |
|--------|---------|-----------|
| `db1574c` | feat: quadratic Coons reconstruction + known-blade re-injection | Erste Plan-B-Geometrie: Coons-Patches mit bekanntem Blatt-Profil |
| `7de1eea` | feat: two-stage tokenizer stage 3 (half-edge HO edge geometry) | Stufe-3-Tokenizer: High-Order Kantengeometrie |
| `93df623` | feat: plan-B two-stage transformer prototypes (all 3 stages) | Alle 3 Transformer-Stufen (Vertex, Face, Geometry) |
| `829da3a` | feat: plan-B end-to-end chaining (point cloud -> full HO-quad mesh) | `chain_e2e.py`: Durchgaengige Kette Point-Cloud -> HO-Quad-Mesh |
| `74d986e` | feat: variable face-count generation in plan B (n-conditioning) | n-Conditioning: variable Face-Anzahl |
| `b491726` | feat: 6-face transfinite subdivision augmentation | 6-Face-Transfinite-Unterteilung fuer Augmentierung |
| `38e5497` | feat: chain_e2e --load-s1 to reuse a pretrained stage-1 model | Warm-Start: S1-Checkpoint wiederverwendbar |
| `790b195` | feat: per-stage checkpointing + --lr in chain_e2e | Per-Stage Checkpoints, LR-Steuerung |

**Beteiligte Dateien:**
`chain_e2e.py`, `reconstruct_domain.py`, `prototype_twostage.py`, `geom_head_prototype.py`, `vertex_head_prototype.py`, `pointer_head_prototype.py`, `tokenizer_domain.py`, `train_domain.py`, `train_pointer.py`

### 2.2 SLURM-Cluster-Integration

| Commit | Message | Bedeutung |
|--------|---------|-----------|
| `fc42110` | feat: bwUniCluster 3.0 SLURM job for Plan-B training | Erster SLURM-Job fuer Plan-B |
| `33c3f00` | feat: uv sync bootstrap in Plan-B SLURM job | `uv sync` im SLURM-Job |
| `67af1f5` | feat: standalone stage-1 SLURM job with per-epoch best-val save | Separater S1-Job mit Best-Val-Save |
| `f570abc` | feat: larger default batch (256) + lr passthrough in S1 SLURM job | Batch-Groesse 256, LR-Weitergabe |
| `8820736` | feat: --init warm-start to continue S1 training from a checkpoint | S1-Warm-Start aus Checkpoint |
| `d8f65df` | feat: LIMIT env in Plan-B job to shrink example build for short tests | LIMIT-Umgebungsvariable fuer kurze Tests |

**Beteiligte Dateien:**
`train_planb.slurm`, `train_s1.slurm`, `probe_nodes.slurm`, `sweep.slurm`

### 2.3 Datensatz und Domain-Pipeline

| Commit | Message | Bedeutung |
|--------|---------|-----------|
| `eda21cc` | feat: 10k dataset extraction + training launcher + HO-quad study | 10k-Datensatz-Extraktion |
| `7f8760a` | feat: point-cloud label conditioning for MeshtronDomain | Point-Cloud Label Conditioning |
| `cb55709` | feat: domain partition pipeline with curved edges | Domain-Partition mit gekruemmten Kanten |

**Beteiligte Dateien:**
`extract_10k.py`, `domain_extractor.py`, `domain_embedding.py`, `domain_trainer.py`, `meshtron_domain.py`, `run_domain_10k.py`

### 2.4 Build-System und Reproduzierbarkeit

| Commit | Message | Bedeutung |
|--------|---------|-----------|
| `a49ee69` | build: uv project setup (pyproject + lock) for reproducible env | `pyproject.toml` + `uv.lock` |
| `275b702` | fix: pin uv cache to workspace fs to avoid hardlink-copy fallback | UV-Cache-Fix |

**Beteiligte Dateien:**
`pyproject.toml`, `uv.lock`

### 2.5 Visualisierung und Dokumentation

- `docs/ho_quad_transformer/README.md` (neu)
- `viz_*.py` (mehr als 10 Visualisierungsskripte)
- `meshtron-domain-obsidian.md`, `handoff.md`, `overview.md`, `progress.md`, `plan.md`
- `figures/e2e/e2e_gallery_*.png` (E2E-Galerien)

---

## 3. Commits nur auf `main`

`main` hat genau 2 Commits seit dem Merge-Base `458a4b0`. Beide sind Bugfixes auf Dateien, die auf `blade-known-reinsertion` in anderer Form existieren.

| Commit | Message | Dateien | Insertions | Deletions |
|--------|---------|---------|------------|-----------|
| `e30489a` | bug fixes | `attention.py`, `hourglass_transformer.py`, `trainer.py` | 7 | 14 |
| `72a0044` | fix: reconcile active quadmesh files for train.py and transformer.py | `train.py`, `transformer.py` | 187 | 201 |

### Analyse

- `e30489a` (2026-05-07): Kleine Korrekturen in Attention und Trainer. Wahrscheinlich auf `main` direkt gefixt, weil der Branch zu diesem Zeitpunkt schon stark divergiert hatte.
- `72a0044` (2026-08-12): Groessere Reconcile-Arbeit an `train.py` und `transformer.py`. Dieser Commit existiert vermutlich, weil `main` die alten Versionen dieser Dateien beibehalten hat, waehrend `blade-known-reinsertion` sie durch `train_domain.py`, `tokenizer_domain.py` etc. ersetzt hat. Der Commit versucht, `main` auf einen lauffaehigen Stand zu bringen, ohne die Branch-Architektur zu aendern.

**Wichtige Beobachtung:** Keiner dieser 2 Commits fuegt neue Features hinzu. Beide sind reaktive Fixes auf einem abgezweigten Stand.

---

## 4. Risiko-Assessment: Merge vs Port

### 4.1 Merge (Rebase) des kompletten Branch

**Vorteile:**
- Historie bleibt vollstaendig erhalten (103 Commits, Training-Provenienz).
- Alle Artefakte (Checkpoints, Galerien, Sweep-Ergebnisse) bleiben referenzierbar.
- Kein manuelles File-Picking noetig.

**Risiken:**
- 795 Nicht-Code-Dateien (Runs, Logs, Bilder, DBs) wuerden in `main` landen. Das vergroessert das Repo massiv.
- `main` hat 2 Commits, die nicht im Branch sind. Die wuerden wahrscheinlich Konflikte erzeugen oder ueberschrieben werden.
- Die Branch-Struktur ist nicht linear: mehrere Merge-Commits (`56488a4`, `ab945a0`, `9a723f5`, `4fa31dd`) zeigen, dass `blade-known-reinsertion` interne Sub-Branches gemerged hat.
- `train.py` und `transformer.py` auf `main` sind von `72a0044` modifiziert, waehrend der Branch diese Dateien durch neue ersetzt hat. Ein Merge wuerde hier wahrscheinlich einen Konflikt erzeugen.

**Schwere:** Mittel-hoch. Nicht unloesbar, aber erfordert sorgfaeltige Konfliktloesung und moeglicherweise ein `git merge --squash` oder `git rebase -i`.

### 4.2 Port ausgewaehlter Module in `main`

**Vorteile:**
- Nur relevanter Code wird uebernommen (~15-20 Python-Dateien statt 866 Dateien).
- `main` bleibt schlank. Artefakte koennen separat archiviert werden.
- Die 2 `main`-spezifischen Fixes (`e30489a`, `72a0044`) bleiben erhalten.

**Risiken:**
- Verlust der Commit-Historie fuer die portierten Dateien. Provenienz geht verloren.
- Abhaengigkeiten zwischen portierten und nicht-portierten Dateien muessen manuell aufgeloest werden.
- `uv.lock` und `pyproject.toml` haben moeglicherweise Konflikte mit der aktuellen `main`-Umgebung.
- Die 3-Stufen-Trainingpipeline (`chain_e2e.py`) hat komplexe Inter-Stage-Abhaengigkeiten. Ein unvollstaendiger Port wuerde sie brechen.

**Schwere:** Mittel. Erfordert manuelles File-Mapping und Integrationstests.

### 4.3 Zusammenfassung Risiko-Matrix

| Strategie | Aufwand | Risiko | Historie | Groesse |
|-----------|---------|--------|----------|---------|
| Voll-Merge | Hoch | Hoch (Konflikte, Bloat) | Voll erhalten | +~50 MB Artefakte |
| Squash-Merge | Mittel | Mittel | Verlust der 103 Commits | +~50 MB Artefakte |
| Selektiver Port | Mittel-hoch | Mittel | Verlust | Minimal |
| Neuer Branch von main + Cherry-Pick | Hoch | Hoch | Fragmentiert | Minimal |

---

## 5. Offene Fragen

1. **Artefakte:** Sollen die ~700 Run-Verzeichnisse, Sweep-DBs und Galerie-Bilder mit in `main` oder in ein separates Archiv-Repo?
2. **main-Fixes:** Die 2 Commits auf `main` (`e30489a`, `72a0044`) beheben Bugs in `attention.py`, `hourglass_transformer.py`, `trainer.py`, `train.py`, `transformer.py`. Sind diese Fixes in `blade-known-reinsertion` bereits enthalten (durch andere Commits) oder fehlen sie dort?
3. **train.py vs train_domain.py:** `main` hat `train.py` als primaeres Trainings-Entrypoint. `blade-known-reinsertion` hat `train_domain.py`, `train_pointer.py`, `train_s1.slurm`, `train_planb.slurm`. Welcher Entrypoint soll canonical werden?
4. **Checkpoints:** Existieren die reported Checkpoints (S1, S2, S3) physisch im Repo oder nur auf dem Cluster? Wenn sie nicht im Repo sind, ist der Branch ohne Cluster-Zugang nicht voll reproduzierbar.
5. **E2E-Kette:** `chain_e2e.py` setzt voraus, dass alle 3 Stufen nacheinander trainiert werden. Ist diese Abhaengigkeit fuer das Ziel-Workflow notwendig, oder reicht ein Subset (z.B. nur S1 + S3)?
6. **Wayfinder-Ticket WF-02:** Die Reproduzierbarkeit der Plan-B-Trainingseinheiten ist noch nicht bestaetigt. Ohne WF-02 ist jede Integrationsentscheidung spekulativ.
7. **dtOO/OpenFOAM-Adapter:** `reconstruct_domain.py` produziert HO-Quad-Meshes. Gibt es bereits einen Export-Pfad zu Gmsh/dtOO auf dem Branch, oder ist das separat zu entwickeln?

---

*Dieses Dokument ist reine Evidenz. Entscheidung in separater Wayfinder-Session (WF-01).*
