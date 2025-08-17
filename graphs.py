import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Charger les données
chemin_fichier = 'combined_stock_data2.csv'
try:
    df = pd.read_csv(chemin_fichier)
    print(f"Fichier chargé avec succès: {df.shape[0]} lignes et {df.shape[1]} colonnes")
except Exception as e:
    print(f"Erreur lors du chargement du fichier: {e}")
    exit(1)

# Sélectionner uniquement les 4 colonnes requises
colonnes_requises = ['SEANCE', 'CODE', 'VALEUR', 'CLOTURE']
try:
    df_reduit = df[colonnes_requises].copy()
    print(f"Colonnes sélectionnées: {', '.join(colonnes_requises)}")
except KeyError as e:
    print(f"Erreur: Colonnes non trouvées dans le fichier. Colonnes disponibles: {', '.join(df.columns)}")
    exit(1)

# Afficher le nombre de valeurs nulles par colonne avant remplacement
print("\nNombre de valeurs nulles par colonne avant remplacement:")
print(df_reduit.isna().sum())

# Remplacer toutes les valeurs nulles par 0.0 dans tout le dataframe
df_reduit = df_reduit.fillna(0.0)

# Vérifier qu'il n'y a plus de valeurs nulles
print("\nNombre de valeurs nulles par colonne après remplacement:")
print(df_reduit.isna().sum())

# Convertir la colonne SEANCE en datetime pour pouvoir extraire l'année
try:
    df_reduit['SEANCE'] = pd.to_datetime(df_reduit['SEANCE'], format='%d/%m/%Y', errors='coerce')
    df_reduit['ANNEE'] = df_reduit['SEANCE'].dt.year

    # Éliminer les lignes où la conversion de date a échoué (devient NaT)
    df_reduit = df_reduit.dropna(subset=['ANNEE'])
    df_reduit['ANNEE'] = df_reduit['ANNEE'].astype(int)

    print(f"\nAnnées présentes dans le jeu de données: {sorted(df_reduit['ANNEE'].unique())}")
except Exception as e:
    print(f"Erreur lors de la conversion des dates: {e}")
    exit(1)

# Créer un dossier pour stocker les graphiques
dossier_resultats = "graphiques_par_annee"
Path(dossier_resultats).mkdir(exist_ok=True)

# Statistiques descriptives globales
print("\nStatistiques descriptives après remplacement des valeurs manquantes:")
print(df_reduit.describe().round(2))

# Générer des graphiques pour chaque année
annees = sorted(df_reduit['ANNEE'].unique())

# Statistiques par année
stats_par_annee = df_reduit.groupby('ANNEE')['CLOTURE'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).round(2)
print("\nStatistiques de CLOTURE par année:")
print(stats_par_annee)

# Graphique global de l'évolution des moyennes annuelles
plt.figure(figsize=(14, 8))
sns.lineplot(x='ANNEE', y='CLOTURE', data=df_reduit, estimator='mean', ci=None, marker='o', linewidth=2)
plt.title('Évolution de la valeur moyenne de clôture par année', fontsize=16)
plt.xlabel('Année', fontsize=14)
plt.ylabel('Valeur de clôture moyenne', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.savefig(os.path.join(dossier_resultats, 'evolution_moyenne_annuelle.png'), dpi=300, bbox_inches='tight')

# Création des graphiques pour chaque année
for annee in annees:
    df_annee = df_reduit[df_reduit['ANNEE'] == annee]

    if len(df_annee) == 0:
        print(f"Pas de données pour l'année {annee}")
        continue

    print(f"\nTraitement des graphiques pour l'année {annee} - {len(df_annee)} observations")

    # Créer une figure avec 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Analyse des valeurs de clôture pour l\'année {annee}', fontsize=20)

    # 1. Histogramme de distribution
    sns.histplot(df_annee['CLOTURE'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title(f'Distribution des valeurs de clôture en {annee}')
    axes[0, 0].set_xlabel('Valeur de clôture')
    axes[0, 0].set_ylabel('Fréquence')

    # 2. Boxplot
    sns.boxplot(y=df_annee['CLOTURE'], ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title(f'Boîte à moustaches des valeurs de clôture en {annee}')
    axes[0, 1].set_ylabel('Valeur de clôture')

    # 3. Top 10 des codes avec les clôtures les plus élevées (UTILISER CODE AU LIEU DE VALEUR)
    top_codes = df_annee.groupby('CODE')['CLOTURE'].mean().nlargest(10).sort_values(ascending=False)
    top_codes.plot(kind='bar', ax=axes[1, 0], color='coral')
    axes[1, 0].set_title(f'Top 10 des codes avec les clôtures moyennes les plus élevées en {annee}')
    axes[1, 0].set_xlabel('Code')
    axes[1, 0].set_ylabel('Clôture moyenne')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Tendance mensuelle si suffisamment de données
    try:
        df_annee['MOIS'] = df_annee['SEANCE'].dt.month
        mois_data = df_annee.groupby('MOIS')['CLOTURE'].mean()

        if len(mois_data) > 1:  # S'il y a des données pour plus d'un mois
            mois_data.plot(kind='line', marker='o', ax=axes[1, 1], color='purple')
            axes[1, 1].set_title(f'Évolution mensuelle moyenne des clôtures en {annee}')
            axes[1, 1].set_xlabel('Mois')
            axes[1, 1].set_ylabel('Clôture moyenne')
            axes[1, 1].set_xticks(range(1, 13))
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[1, 1].text(0.5, 0.5, 'Données insuffisantes pour une analyse mensuelle',
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=12, color='red')
            axes[1, 1].set_title('Analyse mensuelle impossible')
    except Exception as e:
        print(f"Erreur lors de la création du graphique mensuel pour {annee}: {e}")
        axes[1, 1].text(0.5, 0.5, f'Erreur: {str(e)}',
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=12, color='red')

    # Ajouter des statistiques textuelles dans le graphique (UTILISER CODE AU LIEU DE VALEUR)
    stats_text = (f"Nombre d'observations: {len(df_annee)}\n"
                 f"Moyenne: {df_annee['CLOTURE'].mean():.2f}\n"
                 f"Médiane: {df_annee['CLOTURE'].median():.2f}\n"
                 f"Écart-type: {df_annee['CLOTURE'].std():.2f}\n"
                 f"Min: {df_annee['CLOTURE'].min():.2f}\n"
                 f"Max: {df_annee['CLOTURE'].max():.2f}\n"
                 f"Codes uniques: {df_annee['CODE'].nunique()}")

    fig.text(0.02, 0.02, stats_text, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))

    # Ajuster la mise en page et enregistrer
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(dossier_resultats, f'analyse_{annee}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Créer un graphique comparatif des boxplots par année
plt.figure(figsize=(16, 10))
sns.boxplot(x='ANNEE', y='CLOTURE', data=df_reduit)
plt.title('Comparaison des distributions de clôture par année', fontsize=16)
plt.xlabel('Année', fontsize=14)
plt.ylabel('Valeur de clôture', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(dossier_resultats, 'comparaison_boxplots_annuels.png'), dpi=300, bbox_inches='tight')

# Créer un graphique des codes les plus fréquents dans l'ensemble de données
plt.figure(figsize=(16, 10))
top_frequents = df_reduit['CODE'].value_counts().head(20)
top_frequents.plot(kind='bar', color='teal')
plt.title('Top 20 des codes les plus fréquents dans le jeu de données', fontsize=16)
plt.xlabel('Code', fontsize=14)
plt.ylabel('Nombre d\'occurrences', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(dossier_resultats, 'codes_plus_frequents.png'), dpi=300, bbox_inches='tight')

print(f"\nLes graphiques ont été enregistrés dans le dossier '{dossier_resultats}'")

# Sauvegarder le DataFrame nettoyé avec les valeurs nulles remplacées par 0.0
df_reduit.to_csv('donnees_nettoyees.csv', index=False)
print("\nLes données nettoyées ont été enregistrées dans 'donnees_nettoyees.csv'")