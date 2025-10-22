# Dashboard Module

Ce module a été restructuré pour améliorer la maintenabilité et la lisibilité du code.

## Structure

```
dashboard/
├── __init__.py           # Point d'entrée du module
├── generator.py          # Classe principale DashboardGenerator
├── charts.py             # Fonctions de création de graphiques
├── statistics.py         # Fonctions de calcul statistique
├── html_builder.py       # Construction du document HTML
├── templates/
│   ├── styles.css        # Styles CSS du dashboard
│   └── scripts.js        # Code JavaScript (avec fix du filtre TSNE)
└── README.md            # Ce fichier

```

## Changements Principaux

### 1. Séparation des responsabilités
- **generator.py** (160 lignes) : Classe principale DashboardGenerator
- **charts.py** (350 lignes) : Toutes les fonctions de création de graphiques
- **statistics.py** (50 lignes) : Calculs statistiques
- **html_builder.py** (280 lignes) : Construction HTML
- **templates/styles.css** (200 lignes) : Tous les styles CSS
- **templates/scripts.js** (360 lignes) : Tout le JavaScript

### 2. Fix du filtre TSNE

Le problème du filtre de recherche de protéines a été corrigé dans `templates/scripts.js`.

**Problème identifié :**
Le code JavaScript essayait d'extraire les IDs de protéines à partir de texte HTML formaté (`<b>protein_id</b>`), alors que les données sont stockées directement dans `customdata[0]`.

**Solution :**
Les fonctions `searchProtein2D()` et `searchProtein3D()` ont été simplifiées pour utiliser directement :
```javascript
const proteinIds = trace.customdata.map(item => {
    const label = Array.isArray(item) ? item[0] : item;
    return String(label).toLowerCase();
});
```

### 3. Avantages de la restructuration

- **Maintenabilité** : Chaque fichier a une responsabilité claire
- **Réutilisabilité** : Les composants peuvent être utilisés indépendamment
- **Testabilité** : Chaque module peut être testé séparément
- **Lisibilité** : Fichiers plus petits, plus faciles à comprendre
- **CSS/JS séparés** : Plus facile à déboguer et modifier

## Usage

### Import depuis le module restructuré

```python
from dashboard import DashboardGenerator

# Créer un générateur
generator = DashboardGenerator()

# Générer un rapport
report_path = generator.generate_html_report(
    predictions_df=predictions,
    fig_tsne_2d=tsne_2d_fig,
    fig_tsne_3d=tsne_3d_fig,
    output_path="outputs/report.html"
)
```

### Backward compatibility

Le fichier `src/dashboard.py` sert de wrapper pour la compatibilité avec le code existant :

```python
from src.dashboard import DashboardGenerator, create_dashboard
```

## Statistiques

### Avant la restructuration
- **1 fichier** : dashboard.py (1491 lignes)
- Mélange de Python, CSS, et JavaScript
- Difficile à maintenir et déboguer

### Après la restructuration
- **7 fichiers** bien organisés
- Séparation claire Python/CSS/JavaScript
- Total ~1400 lignes (réduction grâce à l'élimination du code redondant)

## Tests

Pour tester que tout fonctionne :

```bash
# Activer l'environnement virtuel
source venv_legionella/bin/activate

# Tester l'import
python -c "from src.dashboard import DashboardGenerator; print('Success!')"

# Exécuter votre pipeline habituel
python main.py
```

## Fichier de backup

L'ancien fichier a été sauvegardé dans :
```
src/dashboard.py.backup
```

## Modifications du filtre TSNE

Le filtre TSNE fonctionne maintenant correctement en :
1. Accédant directement aux données dans `customdata[0]`
2. Convertissant en minuscules pour une recherche insensible à la casse
3. Utilisant `includes()` pour une recherche partielle
4. Surlignant en or (#FFD700) avec une taille augmentée

## Migration

Si vous avez du code qui importe depuis `dashboard.py`, aucune modification n'est nécessaire grâce au wrapper de compatibilité.

Si vous voulez utiliser les nouveaux modules directement :

```python
# Ancien
from src.dashboard import DashboardGenerator

# Nouveau (recommandé)
from src.dashboard import DashboardGenerator  # Marche toujours
# ou
from dashboard import DashboardGenerator      # Import direct du package
```
