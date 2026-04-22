# Réalité Augmentée From Scratch — Semestres 1 (S1)
**Modules 01 à 04 — Bibliothèque mathématique NkMath**  
*KANMO NGAMINI BELLA RINDA · 4AIA · 22P008 . Semestre 1 : NkMath*


---

## Table des matières

- [Semaine 1 — Module 01 : IEEE 754 & Arithmétique Flottante](#semaine-1--module-01--ieee-754--arithmétique-flottante)
- [Semaine 2 — Module 02 : Vecteurs Vec2d, Vec3d, Vec4d](#semaine-2--module-02--vecteurs-vec2d-vec3d-vec4d)
- [Semaine 3 — Module 03 : Matrices Mat3d, Mat4d](#semaine-3--module-03--matrices-mat3d-mat4d)
- [Semaine 4 — Module 04 : Quaternions](#semaine-4--module-04--quaternions)

---

## Semaine 1 — Module 01 : IEEE 754 & Arithmétique Flottante

### Objectifs

Comprendre la représentation binaire des flottants selon le standard IEEE 754, identifier les valeurs spéciales (NaN, Inf, subnormaux), mesurer l'epsilon machine et les ULP, et implémenter des algorithmes numériquement stables (Kahan, Welford).

### Réponses aux questions — Semaine 1

#### Question 1 : Pourquoi `0.1f` n'est-il pas exact ?

`0.1` n'est pas représentable exactement en base 2, de la même façon que `1/3` n'est pas représentable exactement en base 10. La fraction décimale `0.1` correspond à `1/10`, et `10 = 2 × 5`. La puissance de 5 ne se divise pas exactement en base binaire, ce qui entraîne une période infinie en binaire : `0.1 = 0.000110011001100…₂`. Le float 32 bits ne dispose que de 23 bits de mantisse, donc la valeur est tronquée et arrondie, donnant environ `0.100000001490116…` au lieu de `0.1` exact.

#### Question 2 : Vérification de `1.0f` — signe=0, exposant=127, mantisse=0

`1.0f` en IEEE 754 se représente comme `(-1)^0 × 2^(127-127) × 1.0 = 1.0`. Le signe est 0 (positif), l'exposant biaisé est 127 (ce qui correspond à un exposant réel de 0), et la mantisse est 0 (le bit implicite `1.` devant la mantisse suffit à représenter la valeur 1). C'est donc la représentation canonique de l'entier 1 en virgule flottante.

#### Question 3 : Identifier `1.0f / 0.0f` — `+Inf`

En IEEE 754, la division d'un nombre positif fini par zéro ne lève pas d'exception par défaut : elle produit `+Inf`. Le pattern binaire correspondant est : signe=0, exposant=0xFF (tous les bits à 1), mantisse=0x000000 (tous à zéro). L'infini est une valeur spéciale réservée par le standard pour représenter un dépassement positif vers l'infini. On le détecte avec `std::isinf(x) && x > 0`.

#### Question 4 : Identifier `std::sqrt(-1.0f)` — NaN

La racine carrée d'un nombre négatif n'est pas définie dans les réels. IEEE 754 retourne `NaN` (Not a Number) dans ce cas. Le pattern binaire d'un NaN est : exposant=0xFF (tous les bits à 1), mantisse ≠ 0. NaN est **contagieux** : toute opération arithmétique impliquant un NaN produit un NaN. La propriété fondamentale est que `NaN != NaN` est **VRAI** — c'est le seul flottant pour lequel l'égalité avec lui-même échoue. La méthode correcte de détection est `std::isnan(x)`.

#### Question 5 : Comparer les bits de `-0.0f` avec `+0.0f`

`+0.0f` et `-0.0f` ont des représentations binaires différentes :
- `+0.0f` : signe=0, exposant=0x00, mantisse=0x000000 → bits = `0x00000000`
- `-0.0f` : signe=1, exposant=0x00, mantisse=0x000000 → bits = `0x80000000`

Malgré des bits différents, la comparaison `+0.0f == -0.0f` est **VRAIE** en C++ (IEEE 754 l'exige). Le zéro signé est un artefact utile : `1.0f / +0.0f = +Inf` alors que `1.0f / -0.0f = -Inf`.

#### Question 6 : `std::numeric_limits<float>::min()` — identifier un subnormal

`std::numeric_limits<float>::min()` retourne le plus petit float **normalisé positif** (`≈ 1.175e-38`), pas le plus petit subnormal. Un nombre subnormal (dénormalisé) a l'exposant biaisé à 0x00 et une mantisse non nulle — le bit implicite devient `0.` au lieu de `1.`, permettant de représenter des valeurs encore plus petites (jusqu'à `≈ 1.4e-45`). Le plus petit subnormal positif est `std::numeric_limits<float>::denorm_min()`. Les subnormals permettent un "underflow graduel" plutôt qu'un passage brutal à zéro, mais leur manipulation est plus lente sur certaines architectures.

---

## Semaine 2 — Module 02 : Vecteurs Vec2d, Vec3d, Vec4d

### Objectifs

Implémenter les types vectoriels fondamentaux du pipeline 3D avec les garanties de layout mémoire requises pour `glVertexAttribPointer`, les opérations géométriques (dot, cross, normalisation, projection), l'orthogonalisation de Gram-Schmidt, et les coordonnées homogènes.

### Travaux pratiques réalisés

#### TP1 — `Vec2d` complet — 20 tests

Implémentation complète de `Vec2d` avec tous les opérateurs arithmétiques, accès par index, normalisation et produits scalaire/vectoriel 2D.

Tests couverts :
- **Dot product** : vérification sur 6 cas dont `(1,0)·(0,1)=0`, `(3,4)·(3,4)=25`.
- **Cross2D** : `(1,0)×(0,1)=1`, `(0,1)×(1,0)=-1`, cas parallèle = 0.
- **Normalisation** : `||(3,4).Normalized()|| = 1.0` à `kEps` près, direction conservée.
- **Opérateur `[]`** : accès en lecture et modification directe de `x` et `y`.
- **`static_assert(sizeof(Vec2d)==16)`** : garantie layout mémoire pour upload GPU.

#### TP2 — `Vec3d` avec Gram-Schmidt

Implémentation de `Vec3d` avec cross product 3D, `Project`, `Reject` et orthogonalisation de Gram-Schmidt.

Tests couverts :
- Règle de la main droite : `Cross(i,j)=k`, `Cross(j,i)={0,0,-1}`.
- Base complète : `Cross(j,k)=i`, `Cross(k,i)=j`.
- Gram-Schmidt sur **10 triplets aléatoires** : vérification de l'orthonormalité (`|u|=1`, `u·v=0`, `u·w=0`, `v·w=0`).
- `Project(a,b) + Reject(a,b) == a` pour décomposition correcte.

#### TP3 — `Vec4d` et projection perspective simple

Projection des 8 coins d'un cube unitaire en 2D avec `fx=fy=500`, `cx=cy=256`, `z_cam=2.0`. Dessin des coins et des 12 arêtes dans une `NkImage` 512×512 exportée en PPM.

---

## Semaine 3 — Module 03 : Matrices Mat3d, Mat4d

### Objectifs

Implémenter les matrices 4×4 en convention **column-major** (compatible OpenGL), avec produit matriciel, inversion par Gauss-Jordan avec pivot partiel, rotation de Rodrigues, matrice View `LookAt`, et composition TRS.

### Travaux pratiques réalisés

#### TP1 — `Mat4d` et inverse — 24 tests

- **10 tests** `M × Identity() == M` sur matrices aléatoires.
- **10 tests** `M × M⁻¹ == Identity()` à `1e-10` près pour des matrices non singulières.
- **1 test** : l'inverse d'une matrice singulière (ligne dupliquée) retourne `false`.
- **3 tests** : `RotateAxis({0,1,0}, π/2) × {1,0,0,1} ≈ {0,0,-1,1}`.

#### TP2 — Rasteriseur logiciel + rotation du cube

Animation de 10 frames d'un cube en rotation autour de l'axe Y, avec pipeline complet :
- Matrice View générée par `LookAt(eye={0,1,3}, target={0,0,0}, up={0,1,0})`.
- Projection perspective `Perspective(fov=60°, aspect=1.0, near=0.1, far=100)`.
- Affichage des 12 arêtes par `DrawLine` dans `NkImage` 512×512.
- Export PPM pour chaque frame (`frame_TP8_0.ppm` à `frame_TP8_9.ppm`).

#### TP3 — TRS et décomposition — 20 triplets

- Construction de `M = TRS(T, R, S)` pour 20 triplets aléatoires.
- Décomposition inverse par `DecomposeTRS(M, T2, R2, S2)`.
- Vérification de T et S à `kEps` près ; tolérance plus large (5.0) pour R en raison des ambiguïtés d'angles.

---

## Semaine 4 — Module 04 : Quaternions

### Objectifs

Comprendre et implémenter les quaternions unitaires comme représentation des rotations 3D, résoudre le problème du gimbal lock des angles d'Euler, implémenter SLERP pour l'interpolation sphérique, et assurer l'aller-retour Quat ↔ Mat3 via la méthode de Shepperd.

### Travaux pratiques réalisés

#### TP1 — Quaternions complets

**Test 41 — Rotation par quaternion :**  
`Rotate(FromAxisAngle({0,1,0}, π/2), {1,0,0}) ≈ {0,0,-1}` vérifié à `kEps` près sur les 3 composantes.

**Test 42 — Aller-retour Quat → Mat3 → Quat (50 quaternions aléatoires) :**  
Pour chaque quaternion normalisé aléatoire `q1`, conversion en `Mat3` via `ToMat3`, puis reconversion en quaternion via `FromMat3` (méthode de Shepperd). Vérification `ApproxQuat(q1, q2, 1e-4)`. Note : les quaternions `q` et `-q` représentent la même rotation — la tolérance gère l'ambiguïté de signe.

**Test 43 — `Quat × Quat.Inverse() == Identité` (50 quaternions) :**  
Pour tout quaternion unitaire normalisé, `q × q⁻¹` doit être l'identité `{w=1, x=0, y=0, z=0}`, vérifié à `1e-4` près.

#### TP2 — Animation SLERP vs LERP

Animation de 60 frames entre `FromAxisAngle({0,1,0}, 0)` et `FromAxisAngle({0,1,0}, π)` :

**SLERP (frames `Slerp__frame_TP11_0.ppm` à `..._59.ppm`) :**  
Interpolation sphérique sur la sphère des quaternions unitaires. Le chemin suivi est le plus court sur S³, produisant une vitesse angulaire **uniforme** : le cube tourne à vitesse constante entre les deux orientations.

**LERP (frames `Lerp__frame_TP11_0.ppm` à `..._59.ppm`) :**  
Interpolation linéaire naïve des composantes du quaternion, suivie d'une normalisation. Le chemin résultant n'est pas géodésique : la vitesse angulaire est **non uniforme** (plus lente au milieu de l'animation, plus rapide aux extrémités), ce qui produit un mouvement visuellement non naturel.

Les deux séquences utilisent le même pipeline de rasterisation (View + Perspective + `DrawLine`) que le TP de la Semaine 3.

---

## Structure du projet

```
Nkentseu/Application/Sandbox/Sandbox/

├── include/
    ├── Float.h          # IEEE 754, Kahan, Welford, epsilon machine
    ├── Vec2d.h          # Vecteur 2D (dot, cross2D, lerp, static_assert)
    ├── Vec3d.h          # Vecteur 3D (cross, project, reject, Gram-Schmidt)
    ├── Vec4d.h          # Coordonnées homogènes, déhomogénéisation
    ├── Mat3d.h          # Matrice 3×3, transpose, déterminant
    ├── Mat4d.h          # Matrice 4×4 column-major, inverse, LookAt, TRS
    ├── Quat.h           # Quaternions, SLERP, méthode de Shepperd
    └── NKImage.h        # NkImage RGBA, PPM I/O, DrawLine, SetPixel

├── tests/ ALL_TP.cpp           # Tous les tests unitaires S1–S4
```

## Dépendances

- **C++17** (STL uniquement)
- `Unitest/Unitest.h` et `Unitest/TestMacro.h` — framework de tests interne
- `NKLogger/NkLog.h` — logger interne
- Aucune bibliothèque tierce (pas d'OpenCV, GLM, Eigen, stb_image)
