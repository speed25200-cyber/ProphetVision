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
