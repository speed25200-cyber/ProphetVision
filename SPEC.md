# SPEC — ProphetVision v2 (Gravity Auto Roulette)

Objectif : logiciel de prédiction de zone d'atterrissage EN TEMPS RÉEL sur le flux
Gravity Auto Roulette (BeterLive), avec boucle d'auto-apprentissage par lecture
automatique des résultats affichés à l'écran. Hérite du moteur physique v1
(physics.py, predict.py, scatter.py — NE PAS CASSER, tests existants verts).

## Faits mesurés sur la vidéo réelle (ne pas ré-deriver aveuglément)

- Vidéo : 1206×910, conteneur 60 fps AVEC images dupliquées (taux réel unique
  plus faible — détecter et dédupliquer). Fenêtres dispo dans
  `/mnt/agents/output/prophetvision/videos/` :
  `hq_060_100.mp4` (t abs 60→100 s), `hq_100_135.mp4` (t abs 100→135 s).
- Cycle d'un spin ≈ 40 s : paris ~20 s (compte à rebours vert à droite),
  lancement bille ~3 s AVANT la fin du compte à rebours, plan plongée peu après,
  chute ~10-12 s après lancement, résultat affiché (marqueur rouge sur la poche
  + texte « N player(s) won »), puis cycle suivant.
- Plans caméra : OBLIQUE (ellipse, paris) ↔ PLONGÉE (cercle, spin) ; coupures
  franches (diff inter-frame > 8 en moyenne normalisée 300×226).
- PLONGÉE (mesuré) : centre=(610,419), R=345 px. Anneau bille 0.86–1.00 R.
  Anneau rotor (zéro vert) 0.55±0.10 R — tracking absolu du zéro FONCTIONNE
  (v1), rotor ≈ 58°/s. Bille : azimut DÉCROISSANT, ~800°/s à l'apparition du
  plan plongée, ~470°/s 4 s plus tard.
- Piège : réflections statiques (ex. spot az=41°, r=302) — un détecteur couleur
  naïf se verrouille dessus. Solution validée : unwrap polaire de l'anneau +
  soustraction du fond MÉDIAN temporel (`explore/track_median.py`) — la bille
  rapide ressort en blob compact ; segments fragmentés → Kalman obligatoire.
- RAM 4 Go : JAMAIS toutes les frames en mémoire (v1 read_video a été OOM-kill).
  Traitement en streaming obligatoire.
- Vérité terrain à l'écran : bandeau supérieur = historique (dernier numéro à
  GAUCHE, chiffres blancs/rouges sur pastilles sombres, y≈20-60, x≈20-1150) ;
  overlay résultat (marqueur rouge « 25 » au-dessus de la poche gagnante).

## NOTES v2.1 — réalité mesurée (prioritaires sur le reste du SPEC)

1. **Tracking oblique ABANDONNÉ** (2 méthodes testées : frame-diff et
   unwrap-médian elliptique → uniquement des spots statiques + animation
   cartes multiplicateurs ; la bille rapide n'est pas extractible en oblique
   avec la CV classique). Prédiction basée UNIQUEMENT sur le plan plongée.
2. **Le plan plongée ne montre que la queue du spin** : bille ~110°/s →
   ~55°/s sur ~4 s (t_abs 81,5→85,7), chute quand ω_bille ≈ ω_rotor (~55-58°/s).
3. **`omega_drop` DOIT être appris en ligne** : sur cette roue ≈ 50-60°/s
   (PAS le défaut v1 640°/s qui rend t_drop absurde). Idem `fall_time` /
   `fall_travel_deg` ≈ petits (bille et rotor quasi à la même vitesse → la
   bille tombe quasi à la verticale : la poche d'impact ≈ azimut de chute
   projeté sur la phase rotor à cet instant). Calibration auto : à chaque
   fin de piste (drop détecté), omega_drop ← ω mesurée en fin de piste.
4. **Avance de prédiction réaliste** sur ce flux : ~2-3 s avant la chute
   (fit dès ≥1,2 s d'arc après l'entrée en plongée). L'honnêteté du rapport
   prime : afficher l'avance réelle mesurée.
5. Vérité mesurée par l'agent balltrack sur plunge_A.mp4 (approuvée) :
   segment bille [0,3–4,2 s], arc ~370°, v ∈ [70,180]°/s.

## Modules (interfaces contractuelles)

### A. `prophetvision/streaming.py`
```python
class FrameSource:
    def __init__(self, path: str): ...        # streaming, une frame à la fois
    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]: ...
        # (frame_index, t_seconds, frame_bgr) — t depuis le début du flux
    @property
    def fps(self) -> float: ...
def dedup_times(frames_iter, diff_thresh: float = 0.08) -> Iterator[tuple[float, np.ndarray]]:
    """Ne laisse passer que les frames réellement nouvelles (norme L1 sur
    300×226 gray), avec temps CORRIGÉ (comptage des doublons)."""
class ShotSegmenter:
    """Classifie chaque frame : 'oblique' | 'plunge' | 'other'.
    Règle : coupe si diff L1 > seuil ; plunge si un cercle Hough de R∈[250,420]
    centré image est détecté (avec hysteresis), sinon oblique si ellipse large,
    sinon other."""
    def process(self, frame) -> str: ...
```
Tests : segmentation de hq_060_100 → oblique jusqu'à ~21,5 s puis plunge ;
dedup trouve les doublons.

### B. `prophetvision/calibration.py`
```python
@dataclass
class PlungeCal:  cx: float; cy: float; R: float
    @classmethod
    def detect(cls, frame) -> "PlungeCal | None"   # Hough, None si échec
@dataclass
class ObliqueCal:  cx: float; cy: float; a: float; b: float; angle: float
    """Ellipse du rebord en vue oblique."""
    @classmethod
    def detect(cls, frame) -> "ObliqueCal | None"
    def to_image(self, azimuth_deg: float, r_frac: float) -> tuple[float, float]
    def to_azimuth(self, x: float, y: float) -> float
    """Mapping ellipse→cercle canonique (dénormalisation x/a, y/b tournée)."""
```

### C. `prophetvision/balltrack.py` (cœur révolutionnaire)
```python
@dataclass
class BallSample:  t: float; azimuth: float; strength: float
class PolarMedianTracker:
    """Unwrap polaire anneau [r_in,r_out]×R (720 bins) + fond médian glissant
    (fenêtre ~45 frames) + détection blob compact + filtre de Kalman
    état=(azimuth, omega, alpha) en azimut DÉROULÉ, gating innovation 3σ,
    interpolation parabolique sub-bin. Rejette les blobs statiques (vitesse
    incompatible avec une bille) et comble les dropouts ≤ 0,5 s."""
    def __init__(self, cal: PlungeCal | ObliqueCal, fps: float,
                 r_in=0.86, r_out=1.00, n_bins=720): ...
    def process(self, t: float, frame) -> None: ...
    def track(self) -> tuple[np.ndarray, np.ndarray]:  # (t, theta_unwrapped_deg)
        """Azimuts déroulés continus ; dropouts comblés par prédiction Kalman
        UNIQUEMENT si ≤0,5 s, sinon la série est coupée en segments."""
class BallTracker:   # façade multi-plans
    """Utilise PolarMedianTracker sur 'plunge' ; sur 'oblique', même algo via
    ObliqueCal (best effort). Recalib à chaque entrée de plan."""
```
Exigence mesurable (RÉALITÉ mesurée sur plunge_A.mp4 = hq_060_100 tronqué
21,5–39,5 s) : la bille n'est dans l'anneau rebord que de t≈0,3 à t≈4,2 s du
clip (chute à t_abs≈85,7 s ; le plan plongée ne montre que la QUEUE du spin,
la phase rapide ~800°/s ayant lieu en vue oblique t_abs≈74–81). Seuils :
≥90 % des frames du segment bille [0,3–4,2 s] échantillonnées ; arc ≥ 300° ;
vitesse en début de segment ∈ [70, 180]°/s ; monotonicité ≥ 98 % ; < 2 % de
sauts anormaux ; perte de piste propre après la chute (aucun fantôme sur les
spots statiques) ; précision sub-bin ≤ 0,5° (critique : fit sur arc court).

### D. `prophetvision/chronology.py`
```python
@dataclass
class SpinEvent:  kind: str; t: float; detail: dict
    # kind ∈ 'countdown' (detail: seconds_left), 'bets_close' ('No more bets'
    # détecté), 'launch' (1er échantillon bille >300°/s), 'drop' (bille quitte
    # l'anneau rebord), 'result' (detail: number)
class Chronology:
    """Détecte : compte à rebours (OCR chiffre dans le cercle vert à droite,
    template matching sur masks), 'No more bets' (détection texte blanc centre
    haut par énergie de contours), résultat (overlay marqueur rouge + lecture
    du chiffre), bannière d'historique (OCR chiffres 1-2 digits par pastille)."""
    def process(self, t: float, frame) -> list[SpinEvent]: ...
```
OCR digits : construire templates par cross-correlation normalisée sur crops
synthétiques (police sans-serif gras) OU nearest-neighbour sur pixels
seuillés ; valider sur frames réelles extraites (`explore/frames/`).

### E. `prophetvision/rotortrack.py`
Reprendre la logique v1 (zéro vert absolu + corrélation FFT repli) en version
STREAMING (process(t, frame)), recalib par plan, tolérance aux coupures
(réinit douce, pas de saut d'azimut).

### F. `prophetvision/live.py` (orchestrateur temps réel)
```python
class LiveEngine:
    """Pipeline complet : FrameSource → dedup → ShotSegmenter → calibration →
    BallTracker + RotorTracker (sur plunge) → dès que ≥1,2 s d'arc bille :
    fit BallDecayModel + RotorModel → LandingZonePredictor.predict() →
    événement 'prediction' (zone, proba, t_drop estimé, confiance) rafraîchi
    ~2×/s. Chronology alimente les événements et, à 'result', appelle
    observe_outcome() automatiquement (auto-apprentissage du scatter et de
    omega_drop)."""
    def run(self, video_path: str, on_event: callable) -> "SessionReport"
@dataclass
class SessionReport:
    spins: list[dict]   # par spin : events, track stats, predictions (t, zone,
                        # zone_p, top), result OCR, hit (zone contient résultat)
    def summary(self) -> dict
```
Contrainte perf : ≥ 30 frames/s en 1206×910 sur CPU 4 cœurs (unwrap polaire
vectorisé numpy, pas de Python par pixel). Mode `realtime=True` simule le
direct (ne lit pas plus vite que le temps réel).

### G. `prophetvision/dashboard.py`
```python
def render_spin_report(report: SessionReport, video: str, out_html: str,
                       frames_dir: str) -> None
```
HTML auto-contenu par session : pour chaque spin — frame annotée (cercle zone
prédite + bille trackée), heatmap polaire temps×angle, courbes ω(t) bille et
rotor, roue avec probabilités par poche (couleur), zone prédite, chronologie
(countdown/launch/close/drop/result), verdict hit/miss, confiance.
Style : fond sombre élégant (pas de dégradés flashy), typographie nette.

### H. Intégration `cli.py`
Nouvelles commandes :
- `prophetvision live VIDEO [--realtime] [--report out.html]` : session complète.
- `prophetvision history VIDEO` : dump OCR de la bannière (validation).
Toutes doivent STREAMER (jamais read_video sur les vraies vidéos).

## Tests obligatoires (par module, dans tests/)
- streaming : dedup + segmentation sur clip réel A (oblique→plunge à ~21,5 s).
- balltrack : exigence mesurable du §C sur plunge_A.mp4.
- chronology : countdown OCR ≥ 8/10 frames correctes sur t=72..84 ; résultat
  25 détecté sur t=95..99 ; bannière lue = [27,34,29,15,7,...] cohérente.
- live : session sur les 2 fenêtres → ≥1 spin avec prédiction émise ≥4 s AVANT
  la chute estimée de la bille + rapport JSON sauvegardé.
  NOTE CHRONOLOGIE MESURÉE (Gravity Auto Roulette, vidéo Video-v1) : cycle
  ≈47 s ; compte à rebours ~20 s ; « No more bets » ~5 s APRÈS la fin du
  compte à rebours ; lancement bille ~8-9 s APRÈS bets_close ; chute ~12 s
  après lancement. La bille N'EXISTE donc pas pendant la fenêtre de mise :
  le moteur prédit AU PLUS TÔT (dès ≥1,2 s d'arc), mesure l'avance réelle
  par rapport à la chute, et le rapport DOIT afficher honnêtement le gap
  launch−bets_close (verdict de faisabilité par flux). La précision du
  moteur est validée a posteriori : zone prédite vs résultat OCR réel.
- régression : les 13 tests v1 existants restent verts.

## Livrable
Code dans le package `prophetvision/`, tests verts, `verifier` v2 exécuté,
rapport HTML de démo sur les fenêtres réelles, README mis à jour (honnêteté :
prédiction = distribution de probabilité, pas de certitude ; avertissement
légal conservé).
