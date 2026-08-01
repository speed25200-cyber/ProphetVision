# ProphetVision

Prédiction de la zone d'atterrissage d'une bille à partir d'une vidéo, par
vision par ordinateur + modèle physique déterministe + apprentissage en ligne
de la dispersion.

L'idée clé : tant que la bille tourne sur le rebord, sa dynamique est
**déterministe** et obéit à une décélération `dω/dt = −(c₀ + c₂ω²)`
(frottement + traînée aérodynamique), qui admet une **solution analytique
exacte**. Quelques dixièmes de seconde de suivi vidéo suffisent donc à
ajuster le modèle et à extrapoler *tout le reste de la trajectoire* — en
particulier l'instant et l'azimut où la bille quitte le rebord. La phase
chaotique qui suit (déflecteurs, rebonds) est traitée en probabiliste : une
distribution circulaire d'écart de poches, apprise spin après spin.

## Architecture

```
vidéo ─► calibration (détection roue, Hough)     tracking.Calibration
      ─► suivi bille  (diff. d'images dans l'anneau du rebord, déroulé d'angle)
      ─► suivi rotor  (azimut absolu du repère vert, sub-degré ;
                       repli : corrélation de phase FFT sur l'anneau)
      ─► ajustement physique                     physics.BallDecayModel
           régression initiale sur la dérivée, puis raffinement
           Levenberg-Marquardt directement sur la trajectoire analytique
      ─► prédiction chute + point d'impact       predict.LandingZonePredictor
      ─► convolution avec la dispersion apprise  scatter.ScatterModel
      ─► P(poche finale) → meilleure zone de k poches + confiance
```

## Résultats (validation synthétique de bout en bout)

Le générateur `synthetic.py` simule des spins avec une physique connue
(bruit de rendu et rebonds chaotiques inclus) et rend une vraie vidéo ; le
pipeline complet — détection, suivi, ajustement, prédiction — tourne dessus
sans rien connaître de la vérité terrain.

- Instant de chute prédit **2,5 s à l'avance** avec ~10 ms d'erreur.
- Point d'impact sur le rotor prédit à **~2 poches près** (médiane).
- Vitesse du rotor mesurée à **0,01 °/s près** (suivi absolu du zéro vert).
- Après 8 spins d'apprentissage : zone de 9 poches correcte avec **76 %**
  de masse de probabilité (le hasard pur : 24 %).

```
python -m pytest tests/          # 11 tests, dont 3 de bout en bout
```

## Résultat sur la vidéo réelle fournie (Lightning Roulette en ligne)

La vidéo analysée (3 min 57, 954×720) est un **enregistrement d'écran d'une
roulette en ligne en direct** (Lightning Roulette). Le pipeline a été exécuté
dessus. Verdict mesuré, reproductible via `prophetvision audit`.

La bille **est** bien présente et rapide, et elle est traçable pendant
plusieurs secondes dans les plans plongeants les plus longs — un suivi dédié
(signature jaune crème, recherche locale par continuité) la suit sur 345°
d'arc. Ce qui bloque n'est donc pas la détection, mais la **chronologie**.

Chronologie mesurée, sur deux tours indépendants :

| | Tour A | Tour B |
|---|---|---|
| Fin du compte à rebours | ~71 s | ~119 s |
| « No more bets » affiché | 72 s | — |
| Bille observable à partir de | **84 s** | **129,7 s** |
| Retard après fermeture | **+12 s** | **+10,8 s** |

**La bille n'est lancée qu'après la fermeture des paris.** Pendant toute la
fenêtre de mise, la caméra montre un plan oblique de la roue qui tourne à
vide. Vérifié de quatre façons indépendantes : rehaussement de contraste,
carte d'écart-type temporel (activité uniquement au moyeu, aucun anneau
orbital), carte max−médiane, et analyse fréquentielle par pixel sur la bande
1–6 Hz (le pic vient des rayons du moyeu, pas d'une orbite). Il n'existe donc
*aucune information sur la bille* au moment où l'on peut miser.

Et même en ignorant ce point, l'arc observable reste insuffisant : 345° au
lieu des ~720° (deux tours) nécessaires pour séparer le frottement `c₀` de la
traînée `c₂`. Sur un arc aussi court les deux paramètres sont non
identifiables ; l'ajustement refait avec 0,5° de bruit de mesure donne une
incertitude d'atterrissage de **±12 poches sur 37** — soit l'équivalent du
hasard.

### Contre-vérification en pleine résolution (hypothèse « bille très rapide »)

L'hypothèse qu'une bille rapide, étalée par le flou de mouvement, ait été
effacée par la compression 720p a été testée en ré-extrayant les fenêtres
critiques **à la résolution d'origine (1206×910, CRF 14)** via le workflow
`package-video.yml` (branche `video-hq`). Résultats :

- Cartes espace-temps le long de l'ellipse de la vue latérale : les seules
  stries sont la texture du rotor (~60°/s en azimut d'ellipse) et les
  graphismes des multiplicateurs. Aucune strie rapide en sens opposé.
- Détecteur de traînées (une bille à 2-5 tr/s laisse un étalement tangentiel
  de 50-100 px par image à cette résolution) : les seules traînées détectées
  sont les bras de la tourelle centrale et deux reflets fixes qui
  scintillent toujours aux mêmes coordonnées. Aucune traînée ne balaie la
  piste.
- Lecture directe des images : pendant les paris les numéros du rotor sont
  nets (rotor lent) ; après fermeture ils sont flous de mouvement — la roue
  est accélérée à ce moment-là. C'est ce « spin-up », spectaculaire à
  l'écran, qui donne l'impression visuelle d'un objet très rapide sur la
  demi-ellipse. La mesure montre que c'est la roue elle-même, pas une bille.

La bille n'apparaît qu'ensuite, lancée pendant l'affichage des
multiplicateurs, 10 à 13 s après la fermeture des paris.

S'y ajoute un problème d'échantillonnage : le conteneur annonce 60 fps mais
seules ~24–49 images/s portent une information nouvelle (images dupliquées par
la chaîne de diffusion).

**Conclusion honnête : la prédiction est structurellement impossible sur ce
type de flux**, et ce n'est pas une limite de l'algorithme. On ne peut pas
prédire à partir d'une bille qui n'a pas encore été lancée. Aucun logiciel,
quel qu'il soit, ne peut contourner ce point — c'est précisément ainsi que ces
jeux sont conçus. Toute application qui prétendrait le faire sur ce flux
donnerait des sorties sans lien avec le résultat.

Le module `feasibility.py` existe pour dire cela franchement plutôt que de
retourner un chiffre rassurant que les données ne soutiennent pas :

```bash
prophetvision audit ma_video.mp4 --start 129.3 --end 137
```

### Prédiction réelle démontrée sur la vidéo (module `realstream`)

Sur les tours où la bille est observable en vol, le pipeline complet a été
exécuté **avec coupure temporelle** : ajustement du modèle uniquement sur les
données antérieures à la coupure, extrapolation, puis comparaison à ce qui
s'est réellement passé.

**Spin A** (pleine résolution, arc de 291°, résidu d'ajustement 0,97°) :

| | Prédit (à 84,8 s) | Observé (à 86,0 s) |
|---|---|---|
| Azimut au premier contact | 243,9° | 251,1° |
| Poche d'impact | idx 23,6 | idx 24,3 |
| **Erreur** | **0,74 poche, 1,2 s à l'avance** | |

La chaîne complète est vérifiée : le rebond se termine à l'index 7,6 ≈ la
poche du **25**, qui est le numéro gagnant officiel du tour (drapeau à
l'écran + tête de l'historique au tour suivant). L'offset de rebond mesuré
(−17 poches) alimente le modèle de dispersion.

**Spin C** (720p, trou de détection de 1,5 s → ajustement réduit à 159°
d'arc) : erreur de 6,4 poches. C'est exactement le cas que les seuils de
`feasibility.py` signalent : sous ~2 tours d'arc observé, l'extrapolation se
dégrade. Le numéro final estimé par le suivi de la bille posée (poche 9)
est confirmé visuellement à l'image.

Dynamique mesurée de cette roue : rotor +66,7°/s (rms 2,4°), bille lancée à
contre-sens ~−130°/s décélérant à ~−55°/s au contact, spirale de descente
~1,3 s.

### Où la méthode fonctionne réellement

Sur une roue **physique** filmée par une caméra **fixe en plongée**, avec la
bille visible pendant plusieurs tours de rebord (≥ 1,5 s et ≥ 720° d'arc), les
préconditions sont remplies et les performances mesurées sur banc synthétique
s'appliquent. C'est le régime pour lequel le moteur physique a été conçu et
validé.

## Utilisation

```bash
pip install -e .

# Démo auto-validée (aucune vidéo requise) :
prophetvision demo --seed 7 --lead 2.5 --train 8 --save spin.avi

# Sur une vraie vidéo — TOUJOURS auditer d'abord :
prophetvision audit      ma_video.mp4 --start 0 --end 20
prophetvision calibrate  ma_video.mp4
prophetvision analyze    ma_video.mp4 --cutoff 4.0 --zone-width 9 \
    --scatter-file scatter.json
# Après le spin, renseignez le résultat réel pour apprendre la dispersion :
prophetvision analyze    ma_video.mp4 --cutoff 4.0 \
    --scatter-file scatter.json --outcome 20
```

`--cutoff N` limite l'analyse aux N premières secondes : la prédiction est
faite *avant* la chute de la bille, comme en conditions réelles.

Options utiles : `--wheel american`, `--center cx,cy --radius r` (calibration
manuelle si la détection automatique échoue), `--zone-width k`.

### Session temps réel (v2, Gravity Auto Roulette)

```bash
# Session complète en streaming (jamais la vidéo entière en mémoire) :
prophetvision live hq_060_100.mp4 --report session.html --json session.json
prophetvision live flux.mp4 --realtime --verbose   # simule le direct
prophetvision history hq_060_100.mp4               # bannière d'historique OCR
```

Le moteur (`live.py`) chaîne : lecture streaming dédupliquée (~50 fps
réels) → segmentation des plans → calibration plongée → suivi bille
(unwrap polaire + Kalman) et rotor (zéro vert absolu) → dès 1,2 s d'arc de
piste, prédiction de zone rafraîchie ~2×/s (chute estimée par **cinématique
linéaire locale** — sur la queue de spin mesurée, ω ≈ 110→55 °/s est
quasi-linéaire) → lecture OCR du résultat → auto-apprentissage du scatter
de rebond ET de la vitesse de chute (ω_drop ≈ 55-65 °/s sur cette roue, pas
le défaut v1). Le rapport HTML (`dashboard.py`) affiche par spin : roue
avec probabilités par poche, zone prédite (9 poches) et zone adaptative
(dimensionnée à ~67 % de masse), courbes ω(t), chronologie, verdicts
hit/miss bruts et corrigés, frames annotées.

**Honnêteté (mesuré sur le flux BeterLive)** : la bille est lancée ~10 s
APRÈS la fermeture des paris — la prédiction n'est donc pas jouable en mise
sur ce flux ; le rapport affiche ce gap et l'avance réelle
prédiction → chute (~2 s) plutôt qu'une promesse non vérifiable.

## Adapter à votre vidéo

1. La caméra doit voir la roue à peu près de dessus, stable.
2. `WheelConfig` expose les rayons relatifs (anneau de la bille, anneau des
   poches) si votre roue diffère des valeurs par défaut.
3. Le sens de numérotation (`pocket_order`) est l'ordre horaire vu de
   dessus ; les biais angulaires constants restants sont automatiquement
   absorbés par la distribution de dispersion apprise.
4. Plus vous renseignez de résultats réels (`--outcome`), plus la zone se
   resserre.

## Limites et avertissement

**Sur les casinos en ligne.** Le flux analysé ici est un jeu d'argent réel. En
plus de l'impossibilité technique démontrée ci-dessus, utiliser un logiciel de
prédiction contre un opérateur de jeu viole ses conditions d'utilisation
(compte et gains susceptibles d'être annulés), et l'emploi d'un dispositif de
prédiction est une infraction pénale dans de nombreuses juridictions. Ce dépôt
n'est pas fait pour ça et ne fonctionnerait pas pour ça.


La phase de rebond est physiquement chaotique : aucune méthode ne peut donner
la poche exacte à coup sûr — l'objectif honnête est une *distribution* de
probabilité resserrée sur une zone. Ce projet est fourni à des fins d'étude
de la physique et de la vision par ordinateur. L'utilisation d'un dispositif
de prédiction dans un établissement de jeu réel est illégale dans de
nombreuses juridictions : vérifiez votre droit local avant tout usage, et
n'utilisez pas cet outil pour tricher.
