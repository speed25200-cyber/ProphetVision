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

## RÉSULTAT ÉTABLI (2026-08-03) — 18 jetons, 69,6 % en estimation ponctuelle

Chaîne mesurée de bout en bout, prédiction depuis le latéral seul :

- **Ce qui a débloqué la prédiction** : ne plus extrapoler jusqu'à « la sortie
  du rebord » (vitesse supposée 55 °/s — cible inexistante, l'azimut de sortie
  varie de 152° à 272° sur les cinq tours) mais jusqu'à la **vitesse de
  transfert mesurée** : `earlyside.OMEGA_TRANSFER_DEG_S = 93,6 °/s` au passage
  à r = 0,95 du rayon de cuvette, dispersion **1,0 %** sur quatre tours. La
  bille y est à un rayon fixé par la géométrie de la cuvette, donc la vitesse
  y est une propriété de la roue. Erreur sur l'instant de passage :
  **0,38 s** (contre > 1 s avant), soit 6,3 poches à 16,5 poches/s.
- **Dispersion passage → poche payée** : 6,17 poches (4 tours, sans ancrage).
- **σ total = 8,79 poches → 18 jetons = 69,6 %** (IC 90 % 64,2-82,4 ;
  plancher hasard 48,6 %).
- **Non significatif** : Rayleigh p = 0,28 sur quatre tours. Ne jamais
  présenter ce 69,6 % comme démontré ; il faut ~20 tours.

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
