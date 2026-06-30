# Benchmark des méthodes de décision — IoT Node Power Manager

Compare 7 méthodes de prise de décision sur le même simulateur et le même scénario TVCC standard.

## Structure

```
benchmark/
├── environment.py              # Simulateur du node (calibré sur le papier I2MTC)
├── decision/
│   ├── base.py                 # Interface commune à toutes les méthodes
│   ├── fsm_baseline.py         # FSM originale (référence)
│   ├── fsm_optimized.py        # FSM + algorithme génétique sur les seuils
│   ├── random_policy.py        # Sanity check
│   ├── q_learning.py           # Q-learning tabulaire (RL)
│   ├── mlp_distilled.py        # MLP par distillation depuis Q-learning
│   ├── decision_tree.py        # Decision Tree appris depuis Q-learning
│   └── fuzzy_logic.py          # Logique floue avec règles manuelles
├── run_benchmark.py            # Lance tout, génère le CSV
├── plot_results.py             # Génère le graphique de comparaison
└── results/
    ├── benchmark_results.csv
    └── benchmark_comparison.png
```

## Utilisation

```bash
cd benchmark
python3 run_benchmark.py     # ~2 minutes
python3 plot_results.py       # ~2 secondes
```

## Méthodes

| Méthode | Type | Mémoire | Description |
|---|---|---|---|
| Random Policy | Aucun | 4 B | Sanity check |
| FSM Baseline | Rule-based | 6 B | Reproduit le papier I2MTC |
| FSM Optimized (GA) | Optimisation | 6 B | Mêmes règles, seuils appris par algorithme génétique |
| Fuzzy Logic | Rule-based | 48 B | Règles floues avec transitions douces |
| Q-learning tabulaire | RL | 108 B | Table d'états × actions apprise par renforcement |
| Decision Tree | Supervisé (depuis RL) | 522 B | Arbre appris en imitant le Q-learning |
| MLP distillé | Hybride (RL → supervisé) | 59 B | Réseau de neurones imitant le Q-learning |

## Métriques

- **Autonomie (jours)** : durée de survie du node
- **Failures** : nombre de power failures
- **Transmissions** : qualité de service
- **Dropped packets** : pertes de données
- **Decision time (µs)** : coût computationnel
- **Memory (B)** : empreinte de déploiement

## Ajouter une nouvelle méthode

1. Créer un fichier dans `decision/`
2. Hériter de `DecisionMethod`
3. Implémenter `decide(state)` et éventuellement `train(env_factory)`
4. L'ajouter dans la liste `methods` de `run_benchmark.py`

```python
from decision.base import DecisionMethod

class MyMethod(DecisionMethod):
    name = "Ma méthode"
    learning_type = "supervised"

    def decide(self, state):
        # state = {'theta': ..., 'v_batt': ..., 'v_sc': ..., 'packets_pending': ...}
        return 0  # 0=NORMAL, 1=LOW_TEMP, 2=DEEP_CONSERVE

    def memory_bytes(self):
        return 100
```

## Scénario actuel

- 1 semaine de cyclage thermique (TVCC standard)
- Profil sinusoïdal : cycle 8h ±50°C + cycle 1.5h ±20°C + bruit gaussien
- Plage thermique : −40°C à +85°C
- Évaluation moyennée sur 5 seeds différents

## Pour aller plus loin

- Étendre la durée d'épisode (60+ jours) pour différencier sur l'autonomie batterie
- Ajouter d'autres scénarios (transitoire rapide, plateau long)
- Étendre l'espace d'action (RF/sampling/power couplés)
- Calibrer le simulateur avec les données expérimentales du collègue
