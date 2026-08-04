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

## RÉSULTAT ÉTABLI (2026-08-04) — 18 jetons, 51,2 %. OBJECTIF NON ATTEINT.

**Le 69,6 % annoncé la veille était faux** : son terme de prédiction reposait
sur **un seul tour**. Testé sur les quatre tours qui ont des images latérales,
ce terme passe de 6,3 à **13,4 poches** et absorbe tout l'avantage.

- Bug réel trouvé au passage : **le déroulé d'angle perdait des tours** (résidus
  111°, 139°, 231° sur 3 tours sur 4). Correctif : `fit_speed_circular` — ne
  jamais dérouler, voter sur l'azimut enroulé. rms 1,71 s → **0,73 s**.
- Le −0,38 s du tour A était de la chance : le même tour rescanné donne +1,06 s.
- Budget : prédiction **13,45** ⊕ dispersion aval **6,55** = **14,96 poches**
  → 18 jetons = **51,2 %** (plancher 48,6 %). Balayage de la vitesse cible :
  plat entre 49,7 et 51,9 % — ce n'est pas un paramètre mal réglé.
- **Le verrou est la vitesse à la coupure** (connue à 5-14 % près). Avec un
  instant de transfert exact, 18 jetons donneraient **74 %** : le rebond n'est
  pas le problème. Il faut diviser l'erreur de vitesse par 2 à 4.
- **La vidéo ne contient que 5 tours** et ils sont tous utilisés. Le 6e (résultat
  34) s'est immobilisé avant t = 0 — vérifié à l'image. Pour n > 5 il faut
  d'autres enregistrements.

Fixtures : `tests/data/spins_rim.npz` (rebord, 5 tours) et
`tests/data/side_dets.npz` (latéral, 4 tours). `tests/test_real_spins.py` et
`tests/test_prediction_on_all_spins.py` re-dérivent tous les chiffres publiés.

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
