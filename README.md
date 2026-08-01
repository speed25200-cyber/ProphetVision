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

## Utilisation

```bash
pip install -e .

# Démo auto-validée (aucune vidéo requise) :
prophetvision demo --seed 7 --lead 2.5 --train 8 --save spin.avi

# Sur une vraie vidéo (ex. téléchargée depuis Mega sur votre machine) :
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

La phase de rebond est physiquement chaotique : aucune méthode ne peut donner
la poche exacte à coup sûr — l'objectif honnête est une *distribution* de
probabilité resserrée sur une zone. Ce projet est fourni à des fins d'étude
de la physique et de la vision par ordinateur. L'utilisation d'un dispositif
de prédiction dans un établissement de jeu réel est illégale dans de
nombreuses juridictions : vérifiez votre droit local avant tout usage, et
n'utilisez pas cet outil pour tricher.
