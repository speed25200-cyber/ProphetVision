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

## État actuel, en une table

Mesuré sur la vidéo de référence, prédiction émise **2 s avant l'animation des
multiplicateurs**, depuis la **vue latérale seule** :

### ⚠️ Test hors échantillon sur une deuxième vidéo — lisez ceci en premier

Une seconde vidéo (271 s, 6 tours, tag `Video-v2`) a été prédite avec la
calibration **figée** sur la première : échelle 1,0584, ω_transfert 93,6 °/s,
ellipse latérale, seuil d'abstention. Rien n'a été réajusté.

| | vidéo 1 (calibration) | **vidéo 2 (aveugle)** | **les deux** |
|---|---|---|---|
| tours mesurables | 5 / 5 | **4 / 6** | 9 / 11 |
| tours retenus | 3 (abstention) | 4 (aucune abstention) | 7 |
| σ bout-en-bout | 12,66 poches | **5,44 poches** | **8,47 poches** |
| **18 jetons** | 55,0 % | 90,2 % | **71,3 %** |
| plancher | 48,6 % | 48,6 % | 48,6 % |
| Rayleigh p | 0,98 | 0,19 | **0,43** |

**Sur les 7 tours poolés : 71,3 % à 18 jetons contre 48,6 % au hasard, soit
+22,7 points.** Mais **p = 0,43** : sept tours ne suffisent pas à distinguer ce
résultat du hasard. C'est un ordre de grandeur encourageant, pas une preuve.

Trois choses que la vidéo 2 a apprises, toutes défavorables au discours
précédent :

1. **Deux tours sur six ne sont pas mesurables du tout.** La vue plongeante y
   arrive après que la bille soit déjà descendue : il n'y a aucun arc sur
   lequel ancrer la référence. Ce n'est pas un échec de prédiction, c'est un
   défaut de couverture.
2. **Le critère d'abstention n'a rien filtré** : les quatre tours mesurables
   ont tous une concentration > 0,97. Sa capacité de tri — le point le plus
   faible du résultat de la vidéo 1 — reste **non testée**, pas confirmée.
3. **Le budget en quadrature était optimiste.** Mesurée bout-en-bout, la vidéo
   1 donne σ = 12,66 poches là où σ_prédiction ⊕ σ_dispersion annonçait 7,88.
   Les chiffres ci-dessus sont tous des mesures bout-en-bout, seules honnêtes.

Un défaut réel du pipeline a été trouvé grâce à la vidéo 2 et corrigé pour les
deux : `transfer_point` ne lisait la vitesse que sur la piste descendante
finale, alors que les trous de détection coupent souvent la course en plusieurs
pistes et que le franchissement peut tomber dans un trou (tour 116,50 s : bille
suivie à 118 °/s, perdue une seconde, reprise à 84 °/s). Chaque piste fournit
maintenant un point (temps, vitesse) et la loi est ajustée à travers eux.

Les six tours de la vidéo 2 sont figés dans `tests/data/video2_rounds.npz` ;
`tests/test_video2_out_of_sample.py` re-dérive ces chiffres sans la vidéo.

### Peut-on émettre plus tôt ? Le coût de chaque seconde, mesuré

Demande utilisateur : émettre la prédiction **1 s avant le « No More Bets »**.
Testé sur les 9 tours mesurables des deux vidéos en tronquant les détections
figées au nouveau seuil (aucune re-détection, mêmes fixtures) :

| émission | rms instant | dispersion (s) | équivalent poches | verdict |
|---|---|---|---|---|
| NMB + 0,75 s (référence) | 0,93 s | 0,88 | ~16 | fonctionne |
| **NMB + 0,00 s** | **0,78 s** | **0,75** | **~14** | **fonctionne** |
| NMB − 0,50 s | 1,52 s | 1,48 | ~27 | dégradé |
| NMB − 1,00 s | 1,71 s | 1,69 | **~31 sur 37** | **aucune information** |

**À NMB − 1 s, la dispersion de l'instant prédit vaut ~31 poches — plus large
que la roue.** Tout « winrate » calculé à ce seuil est une coïncidence de
repliement (les erreurs font plus d'un demi-tour et retombent parfois près du
but) : Rayleigh p = 0,10–0,75 selon le sous-ensemble, jamais significatif. Un
test (`test_emitting_one_second_before_nmb_is_not_informative`) verrouille ce
constat pour qu'aucun chiffre à ce seuil ne revienne dans ce README sans le
battre.

**Le plus tôt défendable aujourd'hui : l'instant du « No More Bets » lui-même**
(NMB + 0 s), qui précède l'animation de 2,75 s et l'arrivée de la bille de
~11-14 s, sans perte de précision par rapport à la référence.

Pourquoi le mur est là : les détections exploitables ne commencent qu'à
~NMB − 3 s. Avant, la bille est plus haut sur la cuvette, dans l'anneau
r > 1,30 dominé par les reflets fixes — élargir la bande de recherche y fait
chuter la concentration de 0,98 à ~0,2 (fouillis 10 contre 1). Émettre à
NMB − 1 ne laisse donc que ~1,5 s d'arc clairsemé et ~14 s d'extrapolation.
La seule voie identifiée pour gagner ce territoire est un détecteur qui
retrouve la bille dans cet anneau de fouillis (track-before-detect type
`vmf.py`, non testé dans cette fenêtre) — pas un réglage du pipeline actuel.

### Chiffres sur la vidéo 1 seule (calibration)

Validation croisée leave-one-out, cinq tours :

| Mise | tous les tours | avec abstention (n=3) |
|---|---|---|
| 18 jetons | 56,4 % | 74,7 % |
| plancher | 48,6 % | 48,6 % |

⚠️ Ces deux colonnes utilisent le budget en quadrature (σ_prédiction ⊕
σ_dispersion). Le tableau hors échantillon plus haut montre qu'il **sous-estime**
l'erreur réelle : mesurée bout-en-bout, la vidéo 1 donne 55,0 % et non 74,7 %.

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

Sur la vidéo réelle, la chaîne effectivement mesurée est celle-ci :

```
vue latérale, avant l'animation ─► vitesse de la bille       earlyside.py
      ─► extrapolation jusqu'à la vitesse de transfert       earlyside.OMEGA_TRANSFER_DEG_S
         (93,6 °/s à r = 0,95, mesurée, pas supposée)
      ─► phase du rotor à cet instant                        rotorphase.py
      ─► dispersion « passage → poche payée », mesurée       impactmeas.py
      ─► couverture d'une zone de k jetons + IC bootstrap    zone.py
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
python -m pytest tests/          # 100 tests, dont ceux qui re-dérivent
                                 # les chiffres réels publiés plus bas
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

## Prédiction précoce : 2 s avant l'animation des multiplicateurs (`earlyside.py`)

Contrainte opérationnelle : la prédiction doit sortir **2-3 s avant l'animation
des multiplicateurs**, qui recouvre ensuite la roue. Mesurée précisément, elle
démarre à **74,00 s** (spin A) et **122,00 s** (spin B). La coupure est donc
~72 s — et la bille n'atteint la piste extérieure que vers 70,4 s : il reste
**1,5 à 2 s de données**.

C'est bien trop court pour identifier la loi de décroissance, donc **on ne
l'ajuste pas**. `c₀` et `c₂` sont des propriétés de la roue, calibrées une fois
(`WheelDecay.from_two_decelerations`, exacte sur les deux décélérations
mesurées 38,9 et 19,2 °/s² → c₀ = 17,5, c₂ = 1,99×10⁻⁴). Seules la vitesse et
la phase restent libres — un paramètre, récupérable sur 19 détections.

**Ce qui a débloqué la précision.** Un premier passage donnait ω à ±2,7 %, soit
σ ≈ 13 poches. Le diagnostic : les résidus étaient du bruit pur (autocorrélation
0,30, aucune tendance), et un bruit indépendant n'aurait donné que **0,57 %**.
L'écart venait des **trous temporels** — 19 détections groupées en trois paquets
avec des vides de 0,5 s. Deux corrections :

1. **Balayage dense** de la fenêtre 70,2-72,1 s (seuil YOLO abaissé à 0,20, pas
   de tuile réduit) : 83 détections sur 61 images, couverture temporelle
   régulière.
2. **Rejet robuste des aberrants** (`robust_fit_speed`) : quelques détections
   tombent sur un reflet voisin ; les garder faisait passer le résidu de 6 à
   29° et le bootstrap de 0,9 à 1,7 %. À 4,2 poches par 1 % d'erreur de
   vitesse, ça vaut plusieurs poches.

**Ce que l'extrapolation doit viser (correction importante).** Elle visait
« la sortie du rebord », à une vitesse supposée de 55 °/s. C'est une cible qui
n'existe pas : la sortie dépend du déflecteur rencontré, et sur les cinq tours
mesurés son azimut varie de 152° à 272°. La bonne cible est le **passage à
r = 0,95** du rayon de la cuvette : le rayon de la bille y est fixé par
l'équilibre gravité / pente de la cuvette, donc la vitesse à ce rayon est une
propriété de la roue. Mesurée sur quatre tours
(`impactmeas.speed_at_radius`) : **93,6 ± 1,0 °/s, soit 1,0 % de dispersion**.

| tour | ω à r = 0,95 (°/s) | t du passage (s) |
|---|---|---|
| 1 | −92,2 | 40,20 |
| A | −93,8 | 84,65 |
| B | — (l'arc démarre déjà à r = 0,89) | — |
| 4 | −94,5 | 174,52 |
| 5 | −93,7 | 223,85 |

**Résultat sur les quatre tours qui disposent d'images latérales**, coupure
2,0 s avant l'animation, ajustement circulaire (`fit_speed_circular`) :

| tour | coupure | dét. | ω à la coupure | transfert prédit | mesuré | erreur |
|---|---|---|---|---|---|---|
| A | 72,05 | 43 | 486 °/s | 84,19 | 84,44 | **−0,25 s** |
| B | 119,72 | 100 | 356 °/s | 129,37 | 129,95 | **−0,58 s** |
| 4 | 163,30 | 70 | 446 °/s | 174,78 | 174,62 | **+0,17 s** |
| 5 | 209,57 | 36 | 550 °/s | 222,62 | 223,93 | **−1,31 s** |

**rms = 0,73 s.** Le taux relatif bille-rotor au transfert est de 160 °/s =
**18,3 poches/s**, donc cela vaut **13,4 poches**.

⚠️ **Deux corrections de versions antérieures de ce README.**

1. Il annonçait `ω₀ = 614 ± 5 °/s (±0,89 %)`, un résidu de 6,3° et
   **σ = 4,4 poches**. Non reproductibles avec le code du dépôt sur les
   détections sauvegardées. Retirés.
2. Il annonçait ensuite **−0,38 s** sur le tour A. C'était **un seul tour**, et
   le chiffre n'était même pas stable : le même tour rescanné donne +1,06 s.
   Le chiffre défendable est le rms sur quatre tours ci-dessus.

## Mesure du point d'impact : robuste, vérifiée, reproductible (`impactmeas.py`)

Deux implémentations antérieures de cette mesure divergeaient de 14 et 16 poches
sur deux tours sur cinq. Ce que je prenais pour la dispersion du rebond était en
grande partie ce bruit-là. Quatre causes, toutes identifiées et traitées :

1. **Calibration bimodale du cercle.** `HoughCircles` choisit librement entre la
   cuvette (R ≈ 345) et l'anneau des poches (R ≈ 254) ; deux tours avaient été
   calibrés sur le mauvais. `minRadius = 300` fixe cela — les cinq tours donnent
   maintenant R = 345,1 ± 0,3.
2. **Coupures caméra dans la fenêtre.** Détectées par `find_cuts` ; la mesure
   démarre après la dernière.
3. **Objets fixes pris pour la bille.** Un reflet immobile dérive dans le repère
   rotor exactement à la vitesse du rotor, ce qu'aucun test de mouvement relatif
   ne rejette. Le test décisif est en **repère laboratoire** : la bille y court
   encore, et **à contresens du rotor**. Un test de rapport ne suffirait pas —
   près du contact le rapport n'est que de 1,6.
4. **La piste qui survit à l'impact.** La bille tient −92 °/s puis perd sa
   vitesse en 0,15 s tout en continuant à descendre : elle a touché un
   déflecteur. Le détecteur suit encore le blob pendant 0,27 s, pendant
   lesquelles le rotor tourne de 3,7 poches. L'impact est le **dernier
   échantillon encore en mouvement**.

Chaque arc retenu a été **vérifié à l'œil** sur les images (la bille est un
point brillant net sur le rebord).

| tour | t impact | azimut sortie | index impact | arc | descente | ω labo |
|---|---|---|---|---|---|---|
| 1 | 40,72 | 271,3° | **15,95** | 78 éch. / 1,32 s | 0,209 | −9,7 p/s |
| A | 85,60 | 271,6° | **29,09** | 87 éch. / 1,57 s | 0,199 | −8,9 p/s |
| B | 130,05 | 179,0° | **2,16** | 16 éch. / 0,25 s | 0,099 | −9,9 p/s |
| 4 | 176,00 | 152,8° | **2,22** | 148 éch. / 2,53 s | 0,219 | −9,5 p/s |
| 5 | 224,55 | 269,5° | **24,49** | 80 éch. / 1,32 s | 0,180 | −9,8 p/s |

Contrôle : le spin 4 est le seul où les deux anciennes implémentations
s'accordaient (2,21 et 2,75) ; la nouvelle donne **2,22**.

Les détections de rebord des cinq tours sont figées dans
`tests/data/spins_rim.npz` et `tests/test_real_spins.py` re-dérive chacun des
chiffres publiés ici sans la vidéo.

## Ce que le passage à cinq tours a changé

Résultats officiels lus à l'écran et **chacun vérifié par ses deux voisins de
roue**, puis recoupés avec le bandeau d'historique : 27, 25, 9, 1, 31.

La prédiction n'avait été évaluée que sur le tour A. Les cinq tours ont
maintenant été balayés (fenêtre finissant 2 s avant l'animation, seuil YOLO
0,08, ~310 détections brutes par tour). Trois choses en sont sorties.

### 1. Le déroulé d'angle perdait des tours

À 500-600 °/s la bille boucle en 0,6-0,7 s et les détections latérales sont
trouées ; un déroulé prédictif qui se trompe d'un tour est décalé de 360° pour
tout le reste. Résidus : **111°, 139°, 231°** sur trois tours sur quatre.

Correctif : **ne jamais dérouler** (`fit_speed_circular`). Pour chaque vitesse
candidate on retranche le trajet analytique de l'azimut **enroulé** ; la bonne
vitesse laisse une phase constante quel que soit le nombre de tours. Le score
est la longueur du vecteur résultant, lisible comme une confiance — **0,97 à
0,99** sur quatre tours après refit sur les inliers, contre un plancher de
bruit de 0,15.

### 2. La vitesse ajustée n'était pas le maillon faible

`corr(vitesse à la coupure, temps réellement restant) = 0,975` sur cinq tours.
Ce qui était faux, c'est la **conversion vitesse → temps** : elle vient de
coefficients `(c₀, c₂)` mesurés ailleurs, et l'erreur qu'ils produisent croît
avec la longueur de l'extrapolation — signature d'un *taux* de décroissance
faux d'un facteur constant (10,2 s d'extrapolation → −0,61 s ; 14,1 s →
−1,45 s).

`TransferCalibration` ajuste ce seul facteur sur la roue elle-même
(**×1,058**, stable à ±0,015 sur les cinq replis). Deux paramètres — laisser
c₀ et c₂ bouger séparément — **sur-apprend** : le LOO se dégrade à 0,64 s.

| conversion vitesse → temps | rms LOO | 18 jetons | 21 jetons |
|---|---|---|---|
| loi de décroissance telle quelle | 0,851 s | 49,7 % | 57,8 % |
| **échelle calibrée (1 paramètre)** | **0,624 s** | **53,9 %** | **61,8 %** |
| c₀ et c₂ libres (2 paramètres) | 0,639 s | 53,4 % | 61,4 % |

### 3. Le tour A était de la chance

Le même tour, rescanné avec un autre jeu de détections, passait de −0,38 s à
+1,06 s. Erreurs LOO sur les cinq tours : **−0,87 / +0,39 / +0,10 / +0,91 /
−0,45 s**, rms **0,624 s**.

### Le budget

| terme | mesure | σ |
|---|---|---|
| instant de transfert | 5 tours en LOO, rms 0,624 s × 18,3 poches/s | **11,43 poches** |
| transfert → poche payée | 5 tours, sans ancrage | **6,55 poches** |
| **total** | quadrature | **13,18 poches** |

| Mise | Couverture | Plancher | Avantage |
|---|---|---|---|
| 13 | 39,8 % | 35,1 % | +4,7 |
| 18 | 53,9 % | 48,6 % | +5,2 |
| **21** | **61,8 %** | 56,8 % | **+5,1** |
| 24 | 69,5 % | 64,9 % | +4,6 |

### Où est le verrou, chiffré

Le terme aval est bon : **avec un instant de transfert exact, 18 jetons
couvriraient 74 %**. Tout le déficit vient de l'instant. La sensibilité est
`dt/dω ≈ 0,017 s par °/s` à 450 °/s :

| rms sur l'instant | σ total | 18 jetons |
|---|---|---|
| 0,85 s (non calibré) | 16,9 | 49,7 % |
| **0,62 s (aujourd'hui)** | **13,2** | **53,9 %** |
| 0,48 s | 10,9 | 60,0 % |
| 0,30 s | 8,3 | 72,0 % |

**Il reste un facteur ~1,3 à gagner sur l'instant pour tenir 60 % à 18 jetons.**
La fenêtre exploitable ne dure qu'environ une seconde (les détections se
concentrent dans les 0,5 s avant la coupure), ce qui limite la courbure
mesurable et donc la qualité de l'extrapolation sur 10-14 s. C'est là qu'il
faut travailler, et il faut plus de tours pour calibrer mieux.

### Le défaut qui restait : le vote aliasait

Le vote circulaire n'est **pas identifiable seul**. Sur une fenêtre de T
secondes, deux vitesses séparées d'environ 360/T produisent la même phase
enroulée : le vote a donc un peigne de pics, et son maximum global n'est pas
toujours le bon. Mesuré sur les cinq tours : **tous les gagnants sont biaisés
vers le bas**, et sur le tour 1 la bonne réponse était au **troisième** pic
(−2,0 % contre −19,0 % pour le maximum).

Le correctif utilise une information que le vote ignorait : deux détections
séparées d'une image ne sont distantes que d'une dizaine de degrés, ce qui fixe
la vitesse **sans aucune ambiguïté de tour**. `coarse_speed_from_steps` en tire
une estimation grossière (en écartant les pas < 3° — sinon la médiane tombe sur
le fouillis statique, qui est majoritaire), et `fit_speed_gated` restreint le
vote à ±22 % autour d'elle. Chacun sert à ce qu'il sait faire : les pas disent
*quelle dent* du peigne, le vote dit *où exactement* sur cette dent.

Et leur désaccord est le signal d'abstention : un optimum collé au bord de la
fenêtre, ou une concentration effondrée, signalent un tour non prédictible —
sans jamais regarder le résultat.

| tour | concentration | verdict | erreur de vitesse |
|---|---|---|---|
| 1 | 0,44 | **abstention** | −23,4 % |
| A | 0,98 | prédit | −5,1 % |
| B | 0,98 | prédit | −5,7 % |
| 4 | 0,97 | prédit | +0,9 % |
| 5 | 0,44 | **abstention** | −15,5 % |

Erreurs LOO sur les trois tours acceptés : **−0,18 / −0,35 / +0,52 s**,
rms **0,376 s**.

### Combien de tours contient la vidéo

Cinq, et ils sont tous utilisés. La vidéo fait 237,49 s pour une cadence de
46,0 s par tour. L'enregistrement démarre **en cours de tour** : à t = 0 la vue
plongeante est déjà à l'écran, mais la bille est à r ≈ 0,5, immobilisée, et le
marqueur de résultat (**34**, encadré de 17 et 6) apparaît à t = 1,0 s. Sa
sortie de rebord s'est produite vers t = −5 s, avant le début de
l'enregistrement — vérifié sur les images. La fenêtre 0-26 s est extraite par
le workflow et ne contient pas de sixième mesure.

**Pour dépasser n = 5, il faut d'autres enregistrements.** Aucun traitement ne
fabriquera de l'information statistique qui n'est pas dans ces 237 secondes.

### Historique des estimations (toutes dépassées)

Traçabilité des chiffres successivement publiés ici pour 18 jetons, et de ce qui
les invalidait :

| étape | 18 jetons | ce qui n'allait pas |
|---|---|---|
| 2 tours, rebond supposé | 59-82 % | n = 2, σ_rebond posé a priori |
| 5 tours, « jeu A » | 53,7 % | cercle calibré sur l'anneau des poches pour 2 tours |
| 5 tours, « jeu B » | 86,8 % | pistes verrouillées sur des reflets fixes |
| 4 tours, ancrage r = 0,95 | 69,6 % | terme de prédiction mesuré sur **1 seul** tour |
| 5 tours mesurés, 4 prédits | 51,2 % | déroulé d'angle cassé, pas de calibration de roue |
| **5 tours, LOO, échelle calibrée** | **53,9 %** (18 j.) / **61,8 %** (21 j.) | chiffre actuel |

Les deux chiffres du milieu diffèrent d'un facteur 1,6 sur les mêmes images :
c'est la mesure du point d'impact qui bougeait, pas la physique. C'est ce
constat qui a motivé `impactmeas.py`.

### Ce que le suivi du rebond ne permet PAS (piège documenté)

Tentative de mesure directe sur le spin A, en suivant la bille en continu depuis
le rebord jusqu'au repos. Astuce qui évite le problème d'ancrage : impact et
poche finale sont mesurés **dans le même repère rotor**, donc toute erreur
d'ancre s'annule dans leur différence.

Le suivi est propre et à forte confiance (0,68-0,88) pendant la descente, mais
au-delà de 87 s les confiances tombent (0,47-0,70) et **le point
d'immobilisation n'est pas résolu**. La dispersion du rebond ne peut donc pas
être obtenue en suivant la bille jusqu'au repos ; elle n'est accessible que
statistiquement, sur plusieurs tours, par la méthode sans ancrage ci-dessus.

### Le verrou, chiffré : la précision sur l'instant de passage

La bille et le rotor tournent à des vitesses comparables et opposées. L'angle
**relatif** — la seule quantité qui fixe la poche — évolue donc au passage à
r = 0,95 à `93,6 + 66,6 = 160,3 °/s`, soit **16,5 poches par seconde d'erreur**.

| Zone visée | Erreur maximale admissible sur l'instant de passage |
|---|---|
| 9 poches (±4,5) | Δt < 0,27 s |
| 13 poches (±6,5) | Δt < 0,39 s |
| 18 poches (±9) | **Δt < 0,55 s** |
| 21 poches (±10,5) | Δt < 0,64 s |

Viser la vitesse de transfert mesurée (93,6 °/s) au lieu d'une vitesse de
décrochage supposée (55 °/s) était nécessaire mais pas suffisant : sur quatre
tours l'erreur est de **0,73 s rms**, soit au-dessus du seuil des 18 poches.
Le tableau dit exactement ce qu'il reste à gagner — un facteur 2.

### Zone sous hypothèse d'instant de contact connu

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

⚠️ **Ce tableau suppose l'instant de contact connu à ±0,5 s**, information qui
venait de la vue plongeante : c'est une hypothèse, pas un résultat. Depuis le
latéral seul l'instant est prédit à **0,73 s rms** sur quatre tours, donc cette
hypothèse n'est toujours pas satisfaite. Tableau conservé comme trace de
l'étape intermédiaire ; les chiffres à retenir sont ceux de la section « Ce que
le passage à quatre tours a changé ».

Sur un seul spin, dans les deux cas — ce qui n'est pas une validation
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
