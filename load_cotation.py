import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Afficher les informations sur les données manquantes avant le remplacement
print("\nDonnées manquantes avant remplacement:")
print(df_reduit.isnull().sum())
print(f"Pourcentage de valeurs manquantes dans CLOTURE: {df_reduit['CLOTURE'].isnull().mean() * 100:.2f}%")

# Remplacer les valeurs nulles par 0.00
df_reduit['CLOTURE'] = df_reduit['CLOTURE'].fillna(0.0)

# Vérifier qu'il n'y a plus de valeurs manquantes
print("\nDonnées manquantes après remplacement:")
print(df_reduit.isnull().sum())

# Statistiques descriptives après remplacement
print("\nStatistiques descriptives après remplacement des NaN par 0.00:")
print(df_reduit.describe().round(2))

# Métriques de distribution pour la colonne CLOTURE
print("\nMétriques de distribution pour CLOTURE:")
print(f"Moyenne: {df_reduit['CLOTURE'].mean():.2f}")
print(f"Médiane: {df_reduit['CLOTURE'].median():.2f}")
print(f"Écart-type: {df_reduit['CLOTURE'].std():.2f}")
print(f"Min: {df_reduit['CLOTURE'].min():.2f}")
print(f"Max: {df_reduit['CLOTURE'].max():.2f}")

# Visualisation de la distribution
plt.figure(figsize=(14, 8))

# Histogramme
plt.subplot(2, 2, 1)
sns.histplot(df_reduit['CLOTURE'], bins=30, kde=True)
plt.title('Distribution des valeurs de clôture (après remplacement des NaN)')
plt.xlabel('CLOTURE')
plt.ylabel('Fréquence')

# Boxplot
plt.subplot(2, 2, 2)
sns.boxplot(y=df_reduit['CLOTURE'])
plt.title('Boîte à moustaches des valeurs de clôture')
plt.ylabel('CLOTURE')

# Distribution par CODE (top 10)
plt.subplot(2, 2, 3)
df_reduit.groupby('CODE')['CLOTURE'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Moyenne de CLOTURE par CODE (top 10)')
plt.xlabel('CODE')
plt.ylabel('CLOTURE moyenne')

# Distribution par VALEUR (top 10)
plt.subplot(2, 2, 4)
df_reduit.groupby('VALEUR')['CLOTURE'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Moyenne de CLOTURE par VALEUR (top 10)')
plt.xlabel('VALEUR')
plt.ylabel('CLOTURE moyenne')

plt.tight_layout()
plt.savefig('distribution_cloture.png')
plt.show()

# Statistiques par année si SEANCE est une date
try:
    df_reduit['SEANCE'] = pd.to_datetime(df_reduit['SEANCE'], format='%d/%m/%Y', errors='coerce')
    stats_annuelles = df_reduit.groupby(df_reduit['SEANCE'].dt.year)['CLOTURE'].agg(['mean', 'min', 'max', 'count'])
    print("\nStatistiques par année:")
    print(stats_annuelles)
except Exception as e:
    print(f"\nErreur lors de l'analyse par année: {e}")

# Exporter le dataframe nettoyé
df_reduit.to_csv('donnees_nettoyees.csv', index=False)
print("\nDonnées nettoyées exportées vers 'donnees_nettoyees.csv'")