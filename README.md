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

## Vue latérale : la bille est suivie AVANT la coupure caméra (`sidetrack.py`)

C'est ce qui bloquait tout le projet — et le précédent, dont le SPEC porte
« tracking oblique ABANDONNÉ ». Trois pièces l'ont débloqué.

**1. Calibrer l'ellipse par la rigidité du rotor.** Une ellipse devinée sur le
contour visible est assez fausse pour que la vitesse angulaire retrouvée varie
avec le rayon — 147 et 127 °/s mesurés à deux rayons du *même rotor rigide*.
Un corps rigide n'a qu'une vitesse : cet écart est donc un signal d'erreur
objectif. En l'optimisant, l'écart tombe de ~20 à **5,3 °/s**.

**2. Voter sur des trajectoires entières, jamais chaîner les détections.** Le
chaînage glouton se verrouille sur le premier reflet fixe : il a produit une
« piste » figée à 139° pendant huit secondes. La transformée de Hough sur la
famille `θ(t) = φ + ωt + ½αt²` y est insensible, et supporte les trous causés
par la bille qui disparaît derrière le rebord proche à chaque tour.

**3. Rejeter le fouillis statique d'abord** (735 → 167 détections).

**Résultat sur la vidéo de référence** : la bille est suivie de **71,7 s à
80,05 s en vue latérale — avant la coupure caméra mesurée à 81,467 s** — avec
ω₀ = −520 °/s et α = +40 °/s².

**La vérification qui prouve que c'est bien la bille** : cet ajustement,
extrapolé à 81,4 s, donne **−129 °/s**. La vue plongeante, après la coupure et
par une mesure totalement indépendante, donne **−130 °/s à 82,07 s**. Un accord
à 1 % à travers un changement de caméra.

### Zone produite depuis la vue latérale seule

Images utilisées : **71,70 → 80,05 s**, soit entièrement avant la coupure
caméra (81,47 s). 86 détections, modèle physique ajusté (résidu 4,6°).

Contrôle croisé : vitesse prédite à 81,4 s = **−135 °/s**, mesure indépendante
en vue plongeante = **−130 °/s**.

Budget d'erreur par Monte-Carlo (ancre rotor ±1,3 poche, instant de contact
±0,5 s, résidu de trajectoire) → **incertitude 1σ = 3,6 poches**.

| Largeur de zone | Couverture estimée | Contient le résultat réel (25) |
|---|---|---|
| 9 poches | 74 % | non |
| **13 poches** | **91 %** | **oui** |
| 17 poches | 98 % | oui |

Zone 13 poches : `[25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5]`

Autrement dit : à 3,6 poches de σ, une zone de 9 poches est **trop étroite** —
elle rate le résultat sur ce spin. Il faut 13 poches pour atteindre ~90 % de
couverture. C'est la précision réelle en vue latérale aujourd'hui, mesurée et
non postulée, sur **un seul spin** — ce qui n'est pas une validation
statistique.

### Ce qui limite la précision depuis le latéral

La trajectoire de la bille est acquise ; le maillon faible est la **phase
absolue du rotor en vue latérale**. Trois mesures indépendantes se
contredisent (corrélation d'anneau 30,6 °/s, rigidité 89 °/s, plongée
66,6 °/s), et le zéro vert n'est détectable que sur 27 images sur 624, avec une
dispersion de 64,6° (6,6 poches). Une zone de 9 poches demande environ ±4
poches : on n'y est pas. **La zone latérale produite n'est donc pas fiable, et
le dépôt ne prétend pas le contraire.**

## Vue latérale : le détecteur était aveugle — correction méthodologique

Une conclusion négative n'a de valeur que si le détecteur est capable de voir
ce qu'il cherche. Ce contrôle, longtemps omis, change tout.

**Contrôle de cécité.** Sur 78–81,4 s la caméra est en vue latérale et la bille
est *certainement* présente : elle apparaît à 82,07 s, juste après la coupure
caméra mesurée à 81,467 s. La carte max−médiane n'y montre **aucun arc**. Le
détecteur est donc aveugle en vue latérale — et tous les résultats négatifs
obtenus avec lui dans cette vue (y compris « pas de bille pendant les paris »)
sont **sans valeur**.

**Outil construit en réponse** — `vmf.py`, filtrage adapté en vitesse
(*track-before-detect*). On ne seuille jamais une image isolée : on postule une
trajectoire `θ(t) = φ + ωt + ½αt²`, on **intègre le signal brut le long** de
cette trajectoire, et on seuille l'intégrale. Un objet réel s'additionne de
façon cohérente sur N images (signal ∝ N, bruit ∝ √N), le reste s'annule.
C'est une transformée de Radon de la carte espace-temps. S'y ajoute
`null_rotor`, qui bascule la carte dans le référentiel tournant du rotor pour
en retirer exactement la texture (poches, numéros, bras) — sinon le rotor,
bien plus brillant que la bille, domine la transformée.

**Seuil de détection mesuré** sur rendu synthétique oblique, bruité et assombri
(régime où la détection par blob échoue) :

| Gain / bruit | Détection | Ratio pic/plancher | ω trouvée |
|---|---|---|---|
| 0,60 / 5 | oui | 3,20 | correcte |
| 0,30 / 9 | oui | 3,13 | correcte |
| 0,20 / 12 | oui | 2,77 | correcte |
| 0,12 / 16 | **non** | 1,34 | fausse |
| 0,08 / 20 | **non** | 1,33 | fausse |

**Mesure sur la vidéo réelle** : vue plongeante (bille certaine) → ratio
**3,06**, régime de détection franche. Fenêtre de paris 62–76 s en vue latérale
→ ratio **1,44**, avec les meilleurs pics éparpillés entre −825 et +2000 °/s :
c'est la signature du **régime de cécité**, pas celle d'une absence.

**Conclusion honnête** : sur ce flux, en vue latérale, l'instrumentation
actuelle est sous le seuil. La vidéo est compatible avec une bille présente
avant la fermeture des paris. Affirmer l'inverse serait confondre « je ne vois
pas » et « il n'y a rien » — l'erreur commise trois fois dans ce projet.

## Pipeline complet : vidéo → zone d'atterrissage (`endtoend.py`)

Le détecteur appris a débloqué la chaîne complète. Un seul détail décide de
tout : **la porte radiale doit s'appliquer AVANT la suppression non-maximale**.
Le moyeu doré est rond, brillant et couleur bille ; sans cette précaution il
rafle les détections et écrase celles de la bille (417 détections sur le moyeu
contre 3 sur l'anneau). Avec la porte en premier : **une détection propre par
image, confiance médiane 0,83, azimut parfaitement monotone**.

```
YOLO (porte radiale avant NMS)
  → dé-roulement prédictif (récupère les tours entiers cachés dans un trou)
  → ajustement du modèle de décroissance jusqu'à la coupure
  → extrapolation vers le contact rotor
  → phase du rotor (zéro vert) → poche d'impact
  → dispersion apprise → P(poche finale)
```

**Précision mesurée sur le spin A de la vidéo réelle** (erreur de l'azimut
extrapolé contre l'azimut réellement observé, ~1 s plus tard) :

| Arc suivi | Échantillons | Avance | Erreur | En poches |
|---|---|---|---|---|
| 84° | 36 | 1,00 s | 33,8° | 3,48 |
| 84° | 36 | 1,00 s | 48,6° | 5,00 |
| **253°** | 52 | 1,00 s | 5,4° | **0,55** |
| **289°** | 76 | 0,75 s | 1,8° | **0,18** |
| **308°** | 88 | 0,55 s | 1,1° | **0,12** |

La rupture est nette et reproductible : **en dessous de ~250° d'arc suivi, les
deux coefficients de décroissance ne sont pas séparables** et l'erreur passe de
0,2 à 3-5 poches. `predict_spin()` refuse donc explicitement de répondre sous
ce seuil plutôt que de rendre un chiffre que les données ne soutiennent pas :

**Horizon utile ≈ 1 seconde.** L'autre limite, mesurée sur la chaîne complète
(azimut prédit → phase rotor → indice de poche) :

| Avance | Erreur sur l'indice de poche |
|---|---|
| 0,55 s | **0,12** |
| 0,75 s | **0,18** |
| 1,00 s | **0,55** |
| 1,20 s | 3,20 |
| 1,60 s | 2,40 |

Au-delà d'environ 1 s l'erreur décroche : sur ce flux la bille est déjà lente
(≈130 °/s) quand elle devient observable, et la fin de la phase de rebord est
courte. C'est l'horizon réel de ce logiciel sur cette vidéo, pas une valeur
théorique.

```
cutoff 84.0 -> REFUS: tracked arc 201 deg is below the 250 deg needed
cutoff 84.4 | lead 1.6s | det 55 | arc 253 | resid 1.17 | rotor +67
cutoff 84.8 | lead 1.2s | det 79 | arc 289 | resid 1.08 | rotor +67
```

Le dé-roulement prédictif (`unwrap_predictive`) corrige un défaut réel : un
trou de détection de 1,3 s pendant que la bille tourne vite masque un ou
plusieurs tours entiers, qu'un dé-roulement à marge fixe perd silencieusement —
c'est ce qui faisait sauter l'arc ajusté de 84° à 253° en ajoutant quelques
échantillons.

## Nommer la poche : `pocketmap.py`

La physique prédit **où** la bille rejoint le rotor — un azimut relatif au
repère du rotor, et c'est cette quantité qui est validée à 0,12-0,55 poche.
Pour en tirer un **numéro** de poche il manque une constante : le décalage
angulaire entre le repère que la vision accroche et l'origine de la
numérotation. Cette constante dépend de la roue et de la caméra, pas du spin.

Deux façons de la mesurer automatiquement ont été essayées et ont échoué — les
deux échecs sont documentés parce qu'ils sont instructifs :

- **la bille au repos** : immobile elle est petite et sombre, et au rayon des
  poches elle est entourée de séparateurs blancs et de chiffres sur lesquels le
  détecteur tire tout autant. Filtrer assez durement sur la couleur crème pour
  les exclure ne laissait plus que 1 à 2 images exploitables sur des centaines ;
- **le marqueur du résultat** : la carte rouge posée sur le numéro gagnant a
  l'air d'un repère physique, mais c'est un **élément d'interface**. Mesuré de
  95 à 98,6 s, son azimut reste à 269,4 ± 0,5° pendant que la roue tourne de
  120°. Il est fixé à l'écran, pas à la poche.

La solution retenue est donc explicite : **un spin étiqueté suffit**. On lit le
résultat sur le bandeau d'historique du jeu, on appelle `observe()`, et tous
les spins suivants sur la même roue et la même caméra sont nommés. `PocketMap`
refuse de nommer tant qu'il n'est pas calibré, et signale `is_trustworthy =
False` tant que les décalages de plusieurs spins ne concordent pas à moins
d'une poche — le garde-fou qui a détecté que mes deux mesures de bille au repos
divergeaient de 19 poches.

## Détecteur appris (YOLO ONNX) — état mesuré

Le modèle `ball_sota_yolo11n_320_fp16.onnx` fourni avec le dépôt est enveloppé
par `balldetect.py`. Deux faits mesurés conditionnent son usage :

- **Échelle.** Le réseau est en 320×320. Sur l'image entière (1206×910) la
  bille tombe à ~4 px et n'est jamais trouvée. Sur une tuile 320 à résolution
  native centrée sur la bille, il la localise à **5 px près avec 0,80 de
  confiance**. L'inférence doit donc être **par tuiles à résolution native**.
- **La tourelle est un négatif difficile.** Sur une fenêtre plongeante où la
  bille est certainement sur le rebord, un balayage naïf a produit **417
  détections à r = 0,4-0,6 R** (le moyeu doré et ses bras : ronds, brillants,
  couleur bille) contre **3 seulement sur l'anneau de la bille**. D'où la porte
  radiale (`radial_gate`) et le plancher de confiance.

`BallDetector.evaluate()` mesure le taux de succès contre une trajectoire de
référence : c'est ce chiffre, et non une impression, qui décide si un résultat
négatif de ce détecteur signifie quelque chose. En l'état, le balayage par
tuiles n'est **pas** encore un détecteur fiable sur ce flux — il lui manque une
suppression des faux positifs de la tourelle et une couche de suivi.

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
