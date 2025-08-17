import os
import pandas as pd
import logging

# 📁 Dossier contenant tous les fichiers CSV et TXT
csv_root_folder = "cotations_csv"
# 📄 Fichier final combiné
combined_output_path = "combined_cotations_2008_2024.csv"

# 📝 Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 📦 Liste pour stocker tous les DataFrames valides
all_dfs = []

# 🔁 Lecture de chaque fichier
for root, _, files in os.walk(csv_root_folder):
    for file in files:
        if not (file.endswith(".csv") or file.endswith(".txt")):
            continue

        file_path = os.path.join(root, file)
        logger.info(f"📄 Lecture du fichier : {file_path}")

        df = None
        for encoding in ["utf-8", "iso-8859-1", "windows-1252"]:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if df is not None and df.shape[1] > 1:
                        break
                except Exception:
                    continue
            if df is not None and df.shape[1] > 1:
                break

        if df is None or df.shape[1] <= 1:
            logger.warning(f"❌ Lecture échouée pour : {file_path}")
            continue

        # 🔠 Normalisation des noms de colonnes
        df.columns = [col.strip().upper() for col in df.columns]

        # 🔁 Mapping souple des noms de colonnes
        date_col = next((col for col in df.columns if "DATE" in col or "SEANCE" in col), None)
        valeur_col = next((col for col in df.columns if "VALEUR" in col), None)

        if not date_col or not valeur_col:
            logger.warning(f"⚠️ Colonnes nécessaires absentes dans : {file_path}")
            continue

        df = df.rename(columns={date_col: "SEANCE", valeur_col: "Valeur"})

        df["SEANCE_ORIG"] = df["SEANCE"]
        df["SEANCE"] = pd.to_datetime(df["SEANCE"], errors="coerce")

        if df["SEANCE"].isna().any():
            df["SEANCE"] = df["SEANCE"].fillna(method="ffill").fillna(method="bfill")

        df["Année"] = df["SEANCE"].dt.year
        all_dfs.append(df)

# ✅ Fusion finale
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.sort_values("SEANCE", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv(combined_output_path, index=False)

    logger.info(f"✅ Fichier combiné sauvegardé : {combined_output_path}")
    logger.info(f"📊 Nombre total de lignes : {len(combined_df)}")
    logger.info(f"📅 Années présentes : {sorted(combined_df['Année'].dropna().unique())}")
else:
    logger.error("🚫 Aucun fichier valide n'a été traité.")
