# ProphetVision — notes de travail

## Contrainte utilisateur (à respecter dans toute la suite du projet)

**La prédiction doit être possible depuis la vue latérale (side view) seule,
AVANT l'apparition de la vue plongeante aérienne (top-down).**

## OBSERVATION UTILISATEUR — À TRAITER COMME UNE DONNÉE, PAS UNE HYPOTHÈSE

**L'utilisateur voit la bille À L'ŒIL NU sur la vue latérale AVANT le
« No More Bets ».** Elle est présente et rapide sur la demi-ellipse.

Toute conclusion « pas de bille avant NMB » produite par un détecteur est donc
un **échec du détecteur**, pas une propriété de la vidéo. Historique des échecs
(ne pas les refaire à l'identique) :
- différence d'images (moi) → capte les reflets fixes et le rotor accéléré ;
- luminance seule (moi) → se verrouille sur un reflet statique ;
- unwrap polaire elliptique + médiane (KIMI, branche v2-realtime) → idem,
  « tracking oblique ABANDONNÉ » dans son SPEC.md.

### STATUT ÉTABLI (2026-08-03) — l'absence n'est PAS démontrable ici

Contrôle de cécité exécuté : sur la fenêtre 78-81,4 s en vue latérale, où la
bille est **certainement** présente (elle apparaît à 82,07 s juste après la
coupure caméra à 81,467 s), la carte max−médiane ne montre **aucun arc**. Le
détecteur est donc aveugle en vue latérale, et tout résultat négatif obtenu
avec lui y est **sans valeur**.

Outil construit en réponse : `vmf.py` — filtrage adapté en vitesse
(track-before-detect, transformée de Radon espace-temps) + annulation du rotor.
Validé : détecte la bille sur le contrôle plongeant réel (SNR 9,86 = 3,06× le
plancher) et sur rendu synthétique oblique bruité où la détection par blob
échoue.

Seuil de détection mesuré sur synthétique oblique :
- ratio ≥ 2,8 → détection franche, ω correcte ;
- ratio ≈ 1,33 → **cécité**, ω fausse et éparpillée.

Mesure réelle : plongée = ratio 3,06 (détection) ; **paris 62-76 s en latéral =
ratio 1,44, ω éparpillées entre −825 et +2000 °/s → régime de CÉCITÉ**.
Conclusion : la vidéo est compatible avec une bille présente avant le NMB mais
sous le seuil de sensibilité en vue latérale. Ne jamais écrire « pas de bille
avant NMB » : c'est non démontré et probablement faux.

Piste géométrique restant à tester : **la piste de bille est
un cercle situé DANS UN PLAN PLUS HAUT que celui des poches** (la bille roule
sur la paroi interne du rebord). En vue oblique son ellipse projetée est donc
**décalée vers le haut** par rapport à l'ellipse du plan des poches. Chercher
la bille sur l'ellipse du plan des poches revient à chercher au mauvais
endroit. Il faut chercher l'ellipse de la piste (offset vertical à calibrer),
et/ou remonter le temps depuis une détection certaine en vue plongeante à
travers la coupure caméra.

Conséquences pour le pipeline :
- Le suivi bille/rotor doit fonctionner sur la géométrie elliptique de la
  vue oblique (dé-projection ellipse → angle roue réel).
- La fenêtre exploitable est celle où la bille est en vol pendant que la
  side view est affichée — sur le flux étudié, cela correspond à la phase
  d'affichage des multiplicateurs (la roue reste partiellement visible
  autour des cartes).
- La vue plongeante ne peut servir qu'à la vérification a posteriori du
  résultat, jamais à la prédiction.

## RÉSULTAT ÉTABLI (2026-08-04, v2) — 61,8 % sur 21 jetons, en LOO

Tous les chiffres en **validation croisée leave-one-out** (5 tours, calibration
de chaque tour ajustée sur les 4 autres). Les cinq tours sont maintenant
prédits, pas un seul.

| Mise | Couverture | Plancher | Avantage |
|---|---|---|---|
| 18 | 53,9 % | 48,6 % | +5,2 |
| **21** | **61,8 %** | 56,8 % | **+5,1** |
| 24 | 69,5 % | 64,9 % | +4,6 |

**L'objectif « ≥ 60 % » est tenu à partir de 21 jetons.** Mais ne jamais le
présenter sans la colonne « avantage » : élargir la zone monte le taux affiché
*et* le plancher ; les ~5 points d'avantage sont toute l'information extraite.

Trois choses ont produit ce résultat :

1. **Ne jamais dérouler l'angle** (`fit_speed_circular`). Le déroulé prédictif
   perdait des tours (résidus 111°, 139°, 231° sur 3 tours sur 4). On vote sur
   l'azimut **enroulé** ; concentration 0,97-0,99 après refit sur inliers.
2. **La vitesse ajustée n'est pas le problème** : `corr(ω_coupure, temps
   restant) = 0,975`. C'est la **conversion vitesse → temps** qui était fausse.
   `TransferCalibration` ajuste un seul facteur d'échelle sur la roue (×1,058,
   stable à ±0,015 sur les replis). Deux paramètres sur-apprennent (LOO 0,64 s).
3. Erreurs LOO : −0,87 / +0,39 / +0,10 / +0,91 / −0,45 s → **rms 0,624 s**.

Budget : prédiction **11,43** ⊕ dispersion aval **6,55** = **13,18 poches**.

**Verrou restant** : il faut 0,48 s rms pour tenir 60 % à *18* jetons (on est à
0,62). Avec un instant exact, 18 jetons donneraient 74 % — le rebond n'est pas
le problème. La fenêtre exploitable ne dure ~1 s (les détections se concentrent
dans les 0,5 s avant coupure), ce qui limite la courbure mesurable.

**La vidéo ne contient que 5 tours** et ils sont tous utilisés. Le 6e (résultat
34) s'est immobilisé avant t = 0 — vérifié à l'image.

Fixtures : `tests/data/spins_rim.npz` (rebord, 5 tours) et
`tests/data/side_dets.npz` (latéral, 5 tours). `tests/test_real_spins.py` et
`tests/test_prediction_on_all_spins.py` re-dérivent tous les chiffres publiés.

### Chiffres antérieurs retirés (ne pas les ressortir)

- **69,6 %** : terme de prédiction mesuré sur **un seul tour**, et instable
  (le même tour rescanné donnait +1,06 s au lieu de −0,38 s).
- **51,2 %** : déroulé d'angle cassé, aucune calibration de roue.
- **σ = 4,4 poches** : non reproductible avec le code du dépôt.

## Acquis qui tiennent toujours

- **Vitesse de transfert** `earlyside.OMEGA_TRANSFER_DEG_S = 93,6 °/s` au
  passage à r = 0,95 du rayon de cuvette, dispersion **1,0 %** sur quatre
  tours. Le rayon y est fixé par la géométrie de la cuvette, donc la vitesse y
  est une propriété de la roue. C'est la bonne cible d'extrapolation ; « la
  sortie du rebord » n'en est pas une (azimut de sortie 152°-272°).
- **Dispersion transfert → poche payée** : 6,55 poches sur les 5 tours, sans
  ancrage. Suffisamment bon pour 74 % sur 18 jetons si l'instant était exact.
- Forme close du modèle de décroissance (`time_to`, `speed_after`, `travel`) :
  exacte et vectorisée, `fit_speed` passe de 765 000 itérations Python par
  appel à une opération matricielle.

Mesure d'impact rendue reproductible (`impactmeas.py`) après une divergence de
14-16 poches entre deux implémentations. Quatre pièges, tous traités et
testés : calibration Hough bimodale (cuvette R≈345 vs anneau des poches R≈254 —
imposer `minRadius=300`), coupures caméra dans la fenêtre, reflets fixes (le
test décisif est en **repère laboratoire** : la bille court encore et à
contresens du rotor ; un test de rapport ne suffit pas, il n'est que de 1,6),
et la piste qui survit à l'impact (0,27 s de blob immobile = 3,7 poches de
biais). Les cinq arcs ont été **vérifiés à l'œil** sur les images.

Détections figées dans `tests/data/spins_rim.npz` ; `tests/test_real_spins.py`
re-dérive tous les chiffres publiés sans la vidéo.

## Faits établis sur la vidéo de référence (release Video-v1)

- Flux Lightning Roulette, 1206×910 natif, ~24-49 fps utiles (images
  dupliquées par la chaîne de diffusion).
- Chronologie par tour : paris → fermeture → spin-up du rotor (numéros
  flous en side view) → multiplicateurs (~4 s, lancement de la bille
  pendant cette phase) → vue plongeante (bille déjà en fin de course) →
  résultat.
- Pendant les paris, aucune bille sur la roue (vérifié en pleine
  résolution : cartes espace-temps elliptiques, détecteur de traînées,
  analyse spectrale). Le mouvement rapide perçu sur la demi-ellipse est le
  rotor accéléré après fermeture.
- Tour A : fermeture ~71 s, multiplicateurs 77-81,5 s, top-down 81-100 s,
  résultat 25. Tour B : fermeture ~119 s, multiplicateurs 126-129,5 s,
  top-down 129-143 s.
- Les clips pleine résolution des fenêtres critiques sont sur la branche
  `video-hq` (extraits par `.github/workflows/package-video.yml`).
