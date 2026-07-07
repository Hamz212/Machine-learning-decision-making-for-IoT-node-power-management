# Benchmark des méthodes de décision — IoT Node Power Manager

## Structure

```
benchmark/
├── environment.py              # Simulateur du node
├── run_benchmark.py            # Script principal
├── plot_results.py             # Génération du graphique
├── decision/                   # Une classe par méthode
│   ├── __init__.py             # IMPORTANT : requis pour Python
│   ├── base.py                 # Interface commune
│   ├── fsm_baseline.py
│   ├── fsm_optimized.py
│   ├── random_policy.py
│   ├── q_learning.py
│   ├── mlp_distilled.py
│   ├── decision_tree.py
│   └── fuzzy_logic.py
└── results/                    # Créé automatiquement
```

## Installation

```bash
pip3 install numpy matplotlib
```

Si tu as l'erreur "externally managed environment" :
```bash
pip3 install --break-system-packages numpy matplotlib
```

## Utilisation

```bash
cd benchmark
python3 run_benchmark.py     # ~2 minutes
python3 plot_results.py       # ~2 secondes
```

Les résultats seront dans :
- `results/benchmark_results.csv`
- `results/benchmark_comparison.png`

## En cas d'erreur

**`ModuleNotFoundError: No module named 'decision'`**

Vérifie que le fichier `decision/__init__.py` existe. Si non, crée-le :
```bash
touch decision/__init__.py
```

Vérifie aussi que tu lances la commande depuis le dossier `benchmark/`, pas depuis un autre dossier.
