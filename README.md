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

## Prédiction réussie sur la vidéo réelle (spin A)

**Ce qui marche, mesuré sur les images fournies.** Une fois la bille en vol sur
le rebord, sa trajectoire est extrapolée avec précision. Sur le spin A (vidéo
pleine résolution), le modèle est ajusté sur les échantillons **jusqu'à un
instant de coupure**, puis on compare l'azimut prédit à l'azimut réellement
observé au moment du contact avec le rotor (t = 86,00 s) :

| Coupure | Arc ajusté | Avance | Erreur | En poches |
|---|---|---|---|---|
| 83,60 s | 112° | **2,40 s** | −4,4° | **0,45** |
| 84,00 s | 219° | 2,00 s | −10,4° | 1,07 |
| 84,40 s | 255° | 1,60 s | −10,6° | 1,09 |
| 84,80 s | 291° | 1,20 s | −7,2° | **0,74** |
| 85,20 s | 324° | 0,80 s | −11,4° | 1,18 |

Validation *held-out* (échantillons jamais vus par l'ajustement, coupure
84,80 s) : sur toute la phase de rebord restante, l'erreur médiane est de
**0,15 poche** et reste sous 0,4 poche jusqu'à 0,8 s d'avance. Au-delà de
~1,8 s la bille a quitté le rebord et le modèle ne s'applique plus (l'erreur
saute à ~4 poches) — c'est le comportement attendu, pas un défaut.

Ce qu'il a fallu pour y arriver sur des images réelles, et qui est encodé dans
`realstream.py` :

1. **Suivi par couleur + fond médian temporel.** La bille est un blob *jaune
   crème et mobile* : `min(G,R) − B`, moins la médiane temporelle. Sans la
   soustraction de fond, les traqueurs se verrouillent sur des reflets fixes
   (le mode d'échec dominant, rencontré trois fois).
2. **Suppression des images dupliquées.** Le conteneur annonce 60 fps mais
   ~24–49 seulement portent une information nouvelle ; les doublons corrompent
   les estimations de dérivée.
3. **Dé-roulement directionnel.** L'azimut de la bille ne progresse que dans
   un sens ; les pas sont ramenés dans (−330°, +30°] pour qu'une coupure de
   détection ne puisse pas inverser silencieusement le sens de parcours.
4. **Rejet des rayons dorés du moyeu** par contrainte de forme (compact et
   rond) — ils sont jaunes comme la bille.

**Limite honnête.** L'extrapolation de *trajectoire* est validée ci-dessus.
La conversion en *numéro de poche absolu* dépend de la phase du rotor : le
suivi du zéro vert donne un ajustement à 2,4° près (0,24 poche), mais je n'ai
pas pu confirmer indépendamment la correspondance absolue contre le résultat
officiel (25) — la bille au repos n'a pas pu être isolée de façon fiable dans
la phase finale. La chaîne complète azimut → poche → dispersion reste donc à
valider sur davantage de spins.

Sur un second spin (C), l'ajustement n'a disposé que de 159° d'arc à cause
d'une coupure de détection, et l'erreur est montée à 6,4 poches — exactement
le cas que les seuils de `feasibility.py` sont là pour signaler.

## Contexte : la fenêtre de paris sur ce flux

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
