# 🧠 Interprétabilité Mécaniste des LLMs
### Appliquée à GPT-2 Small — Les 3 Outils Fondamentaux

Tutoriel pratique d'interprétabilité mécaniste appliquée à **GPT-2 Small**, implémentant 3 outils fondamentaux pour comprendre ce qui se passe à l'intérieur d'un LLM.

## 🎯 Objectif

Les LLMs sont des boîtes noires : on voit l'entrée et la sortie, mais pas les mécanismes internes. Ce notebook "ouvre" cette boîte avec 3 outils :

| Outil | Question centrale |
|---|---|
| **Classifier Probes** | Où est encodée une information dans les couches ? |
| **Activation Patching** | Quel composant est causalement responsable d'un comportement ? |
| **Sparse Auto-Encoders (SAEs)** | Comment démêler la polysémantie des neurones ? |

## 🏗️ Modèle utilisé : GPT-2 Small

- 12 couches Transformer, 12 attention heads par couche, 768 dimensions par token, 117M paramètres
- Taille idéale : analysable sans GPU puissant, bien documenté dans la littérature de mech interp
- Chargé via **TransformerLens** (Neel Nanda, DeepMind), qui donne accès facile aux activations internes, permet d'utiliser des hooks pour intercepter/modifier les activations, d'analyser les attention heads individuellement, et de faire du patching.

## 📂 Structure du notebook

**Section 0 — Imports et configuration**
Installation de `transformer_lens`, `transformers`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `tqdm`.

**Section 1 — Chargement de GPT-2 et exploration**
- Chargement de GPT-2 Small via `HookedTransformer.from_pretrained('gpt2')`.
- Exploration de l'architecture (tokenisation → embedding → 12 blocs Transformer avec attention + MLP → unembed → prédiction).
- Exploration des activations internes avec `run_with_cache` sur un prompt d'exemple ("The Eiffel Tower is located in Paris...").

**Section 2 — Classifier Probes**
- Principe : entraîner un classifieur simple (régression logistique/SVM) sur les vecteurs internes d'une couche pour vérifier si une information y est encodée. Précision élevée = information présente à cette couche ; précision proche du hasard = absente.
- Expérience : détection **NOM vs VERBE** pour des mots ambigus selon le contexte (ex. *run*, *light*, *water*), avec un jeu de phrases construit pour l'entraînement.
- Extraction du residual stream couche par couche (`extract_residual_stream`), entraînement des probes avec validation croisée stratifiée (5 folds, `StratifiedKFold`), puis visualisation de la précision par couche.

**Section 3 — Activation Patching**
- Principe : prouver la causalité d'un composant en copiant les activations d'un prompt "clean" dans un prompt "corrupted" et en mesurant si cela restaure la prédiction d'origine.
- Expérience : tâche **IOI (Indirect Object Identification)**, étudiée par Wang et al. 2022 (*Interpretability in the Wild*) — le modèle doit identifier le destinataire indirect dans une phrase (ex. "John gave a drink to ___" → "Mary").
- Patching exhaustif couche par couche / token par token, mesure de l'effet sur le `logit_diff`, puis visualisation en heatmap.

**Section 4 — Sparse Auto-Encoders (SAEs)**
- Problème : la **polysémantie** — un même neurone du LLM peut s'activer pour des concepts sans rapport (superposition de concepts due au nombre limité de neurones).
- Solution : projeter les activations denses (768 dims) vers un espace beaucoup plus large mais creux (sparse, 4096+ dims), avec seulement quelques neurones actifs par token.
- Implémentation d'un **SAE TopK** (sparsité garantie par sélection des K activations les plus fortes), architecture encodeur/décodeur avec perte de reconstruction + pénalité L1.
- Collecte des activations de GPT-2 (couche 6, milieu du réseau) sur des données Wikipedia, entraînement du SAE, puis analyse des features apprises (tokens qui activent le plus chaque neurone du SAE = son "concept").

## ⚙️ Dépendances

```bash
pip install transformer_lens transformers torch scikit-learn matplotlib seaborn numpy tqdm
```

- **transformer_lens** : accès aux activations internes de GPT-2, hooks, patching
- **transformers** : modèle GPT-2 et tokenizer
- **scikit-learn** : Classifier Probes (régression logistique, SVM, validation croisée)
- **torch** : calculs tensoriels et implémentation du SAE
- **matplotlib / seaborn** : visualisations (courbes de précision, heatmaps)

GPU recommandé pour l'entraînement du SAE (Section 4), mais le reste du notebook tourne sur CPU.

## ▶️ Utilisation

1. Exécuter les cellules dans l'ordre (Section 0 → 4) ; compatible Google Colab.
2. Section 2 nécessite le jeu de phrases annotées (NOM/VERBE) fourni dans le notebook — extensible avec d'autres exemples.
3. Section 3 utilise une paire de prompts CLEAN/CORRUPTED modifiable pour tester d'autres tâches causales.
4. Section 4 est la plus coûteuse (entraînement du SAE sur des données Wikipedia) ; `SAE_LAYER` et les hyperparamètres (taille cachée, K, λ) sont ajustables.

## 📚 Référence

Wang, K., et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*.
