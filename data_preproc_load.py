import os
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataConverter:
    """
    A class to convert stock data files from various formats (TXT, CSV)
    spanning different years into a unified CSV format.

    Handles format variations:
    - 2008-2017: Original TXT format with fixed columns
    - 2018-2021: Modified TXT format with additional fields
    - 2022-2024: CSV format with flexible column names
    """

    def __init__(self, data_directory: str = './data/', output_file: str = 'combined_stock_data2.csv'):
        self.data_directory = Path(data_directory)
        self.output_file = output_file
        self.standard_columns = [
            'SEANCE', 'GROUPE', 'CODE', 'VALEUR', 'OUVERTURE',
            'CLOTURE', 'PLUS_BAS', 'PLUS_HAUT', 'QUANTITE_NEGOCIEE',
            'NB_TRANSACTION', 'CAPITAUX'
        ]
        self.processed_files = []
        self.failed_files = []
        self.total_rows_processed = 0

    def _get_year_from_path(self, file_path: Path) -> Optional[int]:
        """Enhanced year detection with better accuracy."""
        path_str = str(file_path).lower()

        # First, look for 4-digit years in the path
        year_matches = re.findall(r'(20\d{2})', path_str)
        if year_matches:
            # Return the most recent/relevant year found
            years = [int(y) for y in year_matches]
            return max(years)  # In case multiple years, take the latest

        # Look for 2-digit years in filename specifically
        filename = file_path.name.lower()
        year_match = re.search(r'(?:^|[^\d])(\d{2})(?:[^\d]|$)', filename)
        if year_match:
            year = int(year_match.group(1))
            return 2000 + year if year <= 25 else 1900 + year

        return None

    def _safe_numeric_conversion(self, value: str, data_type: str = 'float') -> Union[float, int, None]:
        """Safely convert string values to numeric types."""
        if not value or value in ['0', '0.000', '0,000', '', 'N/A', 'NULL']:
            return None if data_type == 'float' else 0

        try:
            clean_value = value.replace(',', '.').strip()
            if data_type == 'int':
                return int(float(clean_value))
            return float(clean_value)
        except (ValueError, TypeError):
            return None if data_type == 'float' else 0

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Improved date normalization that handles multiple formats and preserves valid dates.
        Returns date in DD/MM/YYYY format or None if parsing fails.
        """
        if not date_str or pd.isna(date_str):
            return None

        date_str = str(date_str).strip()

        # Common date patterns to try
        date_patterns = [
            # DD/MM/YYYY or DD/MM/YY
            (r'^(\d{1,2})/(\d{1,2})/(\d{4})$', lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"),
            (r'^(\d{1,2})/(\d{1,2})/(\d{2})$',
             lambda m: self._expand_two_digit_year(int(m.group(1)), int(m.group(2)), int(m.group(3)))),

            # YYYY-MM-DD
            (r'^(\d{4})-(\d{1,2})-(\d{1,2})$', lambda m: f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"),

            # DD-MM-YYYY or DD-MM-YY
            (r'^(\d{1,2})-(\d{1,2})-(\d{4})$', lambda m: f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"),
            (r'^(\d{1,2})-(\d{1,2})-(\d{2})$',
             lambda m: self._expand_two_digit_year(int(m.group(1)), int(m.group(2)), int(m.group(3)))),

            # MM/DD/YYYY (US format)
            (r'^(\d{1,2})/(\d{1,2})/(\d{4})$',
             lambda m: f"{int(m.group(2)):02d}/{int(m.group(1)):02d}/{m.group(3)}" if int(
                 m.group(1)) > 12 else f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"),
        ]

        for pattern, formatter in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    result = formatter(match)
                    # Validate the result by trying to parse it
                    if result and self._validate_date(result):
                        return result
                except:
                    continue

        # If no pattern matches, try pandas parsing as last resort
        try:
            parsed = pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')
            if not pd.isna(parsed):
                return parsed.strftime('%d/%m/%Y')
        except:
            pass

        logger.debug(f"Could not parse date: {date_str}")
        return None

    def _expand_two_digit_year(self, day: int, month: int, year: int) -> str:
        """Expand 2-digit year to 4-digit year."""
        if year <= 25:
            full_year = 2000 + year
        else:
            full_year = 1900 + year
        return f"{day:02d}/{month:02d}/{full_year}"

    def _validate_date(self, date_str: str) -> bool:
        """Validate that a date string is actually a valid date."""
        try:
            pd.to_datetime(date_str, format='%d/%m/%Y', errors='raise')
            return True
        except:
            return False

    def _parse_txt_old_format(self, file_path: Path) -> List[Dict]:
        """Parse TXT files from 2008-2017 with the original format."""
        data_rows = []
        encoding_attempts = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        content = None
        for encoding in encoding_attempts:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                    content = file.read()
                break
            except Exception as e:
                continue

        if content is None:
            logger.error(f"Could not read file {file_path} with any encoding")
            return []

        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

        for line_num, line in enumerate(lines, 1):
            # Skip header lines and separators
            if any(skip_term in line.upper() for skip_term in ['SEANCE', '------', '<HR>', 'GROUPE']):
                continue

            # IMPROVED DATE REGEX FOR TXT FILES
            date_match = re.match(r'^(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))', line)
            if not date_match:
                continue

            seance_raw = date_match.group(1)
            # Normalize date using improved function
            seance = self._normalize_date(seance_raw)
            if not seance:
                logger.debug(f"Invalid date in line {line_num}: {seance_raw}")
                continue

            # Find where the other parts of the line start
            line_remainder = line[len(seance_raw):].strip()
            parts = line_remainder.split()

            if len(parts) < 10:
                continue

            try:
                groupe, code = parts[0], parts[1]

                # Find where numeric values start (look for decimal numbers)
                numeric_start_idx = None
                for i in range(2, len(parts)):
                    if re.match(r'^\d+([,\.]\d+)?$', parts[i]):
                        numeric_start_idx = i
                        break

                if numeric_start_idx is None:
                    continue

                # Extract company name (everything between code and first numeric value)
                valeur = ' '.join(parts[2:numeric_start_idx])
                numeric_parts = parts[numeric_start_idx:]

                # Ensure we have at least 7 numeric values
                if len(numeric_parts) >= 7:
                    row = {
                        'SEANCE': seance,
                        'GROUPE': groupe,
                        'CODE': code,
                        'VALEUR': valeur.strip(),
                        'OUVERTURE': self._safe_numeric_conversion(numeric_parts[0]),
                        'CLOTURE': self._safe_numeric_conversion(numeric_parts[1]),
                        'PLUS_BAS': self._safe_numeric_conversion(numeric_parts[2]),
                        'PLUS_HAUT': self._safe_numeric_conversion(numeric_parts[3]),
                        'QUANTITE_NEGOCIEE': self._safe_numeric_conversion(numeric_parts[4], 'int'),
                        'NB_TRANSACTION': self._safe_numeric_conversion(numeric_parts[5], 'int'),
                        'CAPITAUX': self._safe_numeric_conversion(numeric_parts[6])
                    }
                    data_rows.append(row)

            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing line {line_num} in {file_path.name}: {str(e)[:100]}")
                continue

        return data_rows

    def _parse_txt_2018_2021(self, file_path: Path) -> List[Dict]:
        """Parse TXT files from 2018-2021 with the modified format - FIXED VERSION."""
        data_rows = []
        encoding_attempts = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        content = None
        for encoding in encoding_attempts:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                    content = file.read()
                break
            except Exception:
                continue

        if content is None:
            logger.error(f"Could not read file {file_path} with any encoding")
            return []

        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

        logger.info(f"Processing {len(lines)} lines from {file_path.name}")
        processed_count = 0

        for line_num, line in enumerate(lines, 1):
            # Skip header lines and separators - be more specific
            if (line.startswith('SEANCE') or
                    line.startswith('------') or
                    '<HR>' in line.upper() or
                    line.startswith('Erreur') or
                    'GROUPE CODE' in line or
                    len(line.replace('-', '').replace(' ', '')) == 0):  # Empty or just dashes
                continue

            # Look for date pattern at start of line (DD/MM/YYYY or DD/MM/YY format)
            date_match = re.match(r'^(\d{1,2}/\d{1,2}/\d{2,4})', line)
            if not date_match:
                continue

            try:
                seance_raw = date_match.group(1)

                # Normalize date using improved function
                seance = self._normalize_date(seance_raw)
                if not seance:
                    logger.debug(f"Invalid date in line {line_num}: {seance_raw}")
                    continue

                # Remove the date from the beginning and process the rest
                line_remainder = line[len(seance_raw):].strip()

                # Split the remainder into parts
                parts = line_remainder.split()

                # Need at least: GROUPE, CODE, VALEUR, and some numeric data
                if len(parts) < 4:
                    continue

                # Extract GROUPE (should be first after date)
                groupe = parts[0]

                # Extract CODE (should be second after date)
                code = parts[1]

                # Find where numeric data starts by looking for decimal numbers
                numeric_start_idx = None
                for i in range(2, len(parts)):
                    # Look for patterns like: 7.810, 18.950, 0.630, etc.
                    if re.match(r'^\d+\.\d+$', parts[i]) or re.match(r'^\d+,\d+$', parts[i]):
                        numeric_start_idx = i
                        break
                    # Also check for whole numbers that could be prices/volumes
                    elif re.match(r'^\d+$', parts[i]) and len(parts[i]) <= 6:  # Reasonable price/volume range
                        numeric_start_idx = i
                        break

                if numeric_start_idx is None or numeric_start_idx <= 2:
                    logger.debug(f"No numeric data found in line {line_num}: {line[:100]}")
                    continue

                # Extract company name (everything between CODE and first numeric value)
                valeur = ' '.join(parts[2:numeric_start_idx]).strip()

                # Extract numeric parts and clean them
                numeric_parts = [part.replace(',', '.') for part in parts[numeric_start_idx:]]

                # Need at least 7 numeric values: OUVERTURE, CLOTURE, PLUS_BAS, PLUS_HAUT, QUANTITE, NB_TRANS, CAPITAUX
                if len(numeric_parts) < 7:
                    logger.debug(f"Insufficient numeric data in line {line_num}: found {len(numeric_parts)}, need 7+")
                    continue

                # Parse the numeric values based on the 2018-2021 format
                ouverture = self._safe_numeric_conversion(numeric_parts[0])
                cloture = self._safe_numeric_conversion(numeric_parts[1])
                plus_bas = self._safe_numeric_conversion(numeric_parts[2])
                plus_haut = self._safe_numeric_conversion(numeric_parts[3])
                quantite_negociee = self._safe_numeric_conversion(numeric_parts[4], 'int')
                nb_transaction = self._safe_numeric_conversion(numeric_parts[5], 'int')
                capitaux = self._safe_numeric_conversion(numeric_parts[6]) if len(numeric_parts) > 6 else None

                # Skip entries where all price values are 0 or None
                if all(val in [0, None] for val in [ouverture, cloture, plus_bas, plus_haut]):
                    continue

                # Only add row if we have meaningful data
                if (any([ouverture, cloture, plus_bas, plus_haut]) or
                        (quantite_negociee and quantite_negociee > 0) or
                        (nb_transaction and nb_transaction > 0)):
                    row = {
                        'SEANCE': seance,
                        'GROUPE': groupe,
                        'CODE': code,
                        'VALEUR': valeur,
                        'OUVERTURE': ouverture,
                        'CLOTURE': cloture,
                        'PLUS_BAS': plus_bas,
                        'PLUS_HAUT': plus_haut,
                        'QUANTITE_NEGOCIEE': quantite_negociee or 0,
                        'NB_TRANSACTION': nb_transaction or 0,
                        'CAPITAUX': capitaux
                    }
                    data_rows.append(row)
                    processed_count += 1

            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing line {line_num} in {file_path.name}: {str(e)}")
                continue

        logger.info(f"Successfully parsed {processed_count} records from {file_path.name}")
        return data_rows

    def _process_csv_file(self, file_path: Path) -> List[Dict]:
        """Process CSV files from 2022-2024 with flexible column mapping."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        df = None
        for encoding in ['utf-8', 'ISO-8859-1', 'latin1', 'windows-1252']:
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    if df is not None and len(df.columns) > 1:
                        break
                except Exception as e:
                    continue  # skip to next encoding/sep if there's an error
            if df is not None and len(df.columns) > 1:
                break

        if df is None or len(df.columns) <= 1:
            logger.error(f"Could not read {file_path.name} with any encoding/separator combination")
            return []

        # AGGRESSIVE COLUMN NAME CLEANING FOR CSV FILES
        df.columns = df.columns.str.lower().str.replace('\xa0', ' ').str.strip()
        logger.info(f"CSV has {len(df)} rows and cleaned columns: {list(df.columns)}")

        # Enhanced column mapping with more variations
        column_mapping = {}
        for col in df.columns:
            # Date/Session mapping
            if any(term in col for term in ['seance', 'date', 'session']):
                column_mapping[col] = 'SEANCE'
            # Group mapping
            elif any(term in col for term in ['groupe', 'group', 'grp']):
                column_mapping[col] = 'GROUPE'
            # Code mapping (be more specific to avoid conflicts)
            elif 'code' in col and any(term in col for term in ['val', 'valeur', 'isin']):
                column_mapping[col] = 'CODE'
            elif col in ['code', 'code_valeur', 'symbol']:
                column_mapping[col] = 'CODE'
            # Company name mapping
            elif any(term in col for term in ['valeur', 'lib', 'societe', 'company', 'name', 'libelle']):
                column_mapping[col] = 'VALEUR'
            # Price mappings
            elif any(term in col for term in ['ouverture', 'open', 'opening']):
                column_mapping[col] = 'OUVERTURE'
            elif any(term in col for term in ['cloture', 'close', 'closing', 'fermeture']):
                column_mapping[col] = 'CLOTURE'
            elif any(term in col for term in ['plus_bas', 'min', 'low', 'minimum']):
                column_mapping[col] = 'PLUS_BAS'
            elif any(term in col for term in ['plus_haut', 'max', 'high', 'maximum']):
                column_mapping[col] = 'PLUS_HAUT'
            # Volume and transaction mappings
            elif any(term in col for term in ['quantite', 'volume', 'qty', 'negociee']):
                column_mapping[col] = 'QUANTITE_NEGOCIEE'
            elif any(term in col for term in ['transaction', 'nb_tran', 'trans', 'count']):
                column_mapping[col] = 'NB_TRANSACTION'
            elif any(term in col for term in ['capitaux', 'capital', 'montant', 'amount', 'value']):
                column_mapping[col] = 'CAPITAUX'

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure all required columns exist with proper defaults
        for col in self.standard_columns:
            if col not in df.columns:
                if col in ['QUANTITE_NEGOCIEE', 'NB_TRANSACTION']:
                    df[col] = 0
                else:
                    df[col] = None
                logger.info(f"Added missing column: {col}")

        # Normalize dates in CSV files
        if 'SEANCE' in df.columns:
            logger.info("Normalizing dates in CSV file...")
            df['SEANCE'] = df['SEANCE'].apply(self._normalize_date)
            # Remove rows where date normalization failed
            original_len = len(df)
            df = df[df['SEANCE'].notna()]
            removed = original_len - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed} rows with invalid dates from CSV")

        # Clean numeric columns
        numeric_columns = ['OUVERTURE', 'CLOTURE', 'PLUS_BAS', 'PLUS_HAUT', 'CAPITAUX']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        integer_columns = ['QUANTITE_NEGOCIEE', 'NB_TRANSACTION']
        for col in integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        return df[self.standard_columns].to_dict('records')

    def _process_file(self, file_path: Path) -> Tuple[List[Dict], str]:
        """Process a single file based on its format and year."""
        logger.info(f"Processing: {file_path.parent.name}/{file_path.name}")
        year = self._get_year_from_path(file_path)
        if year:
            logger.info(f"📅 Processing year: {year}")
        else:
            logger.warning(f"⚠️ Could not detect year for file: {file_path}")

        try:
            if file_path.suffix.lower() == '.csv':
                data = self._process_csv_file(file_path)
                return data, "CSV"

            # Handle TXT files based on year
            year = self._get_year_from_path(file_path)
            logger.info(f"Detected year: {year}")

            if year and 2018 <= year <= 2021:
                data = self._parse_txt_2018_2021(file_path)
                return data, f"TXT_2018_2021 (Year: {year})"
            else:
                data = self._parse_txt_old_format(file_path)
                return data, f"TXT_OLD (Year: {year or 'Unknown'})"

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return [], f"ERROR: {str(e)[:50]}"

    def _find_additional_csv_files(self) -> List[Path]:
        """Find additional CSV files for 2022-2024 data."""
        additional_files = []

        # Look for common patterns for 2022-2024 CSV files
        patterns = [
            '**/histo_cotation_2022.csv',
            '**/histo_cotation_2023.csv',
            '**/histo_cotation_2024.csv',
            '**/cotation_2022*.csv',
            '**/cotation_2023*.csv',
            '**/cotation_2024*.csv',
            '**/*2022*.csv',
            '**/*2023*.csv',
            '**/*2024*.csv'
        ]

        for pattern in patterns:
            files = list(self.data_directory.glob(pattern))
            additional_files.extend(files)

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file in additional_files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)

        return unique_files

    def convert_to_csv(self, validate_data: bool = True) -> Optional[pd.DataFrame]:
        """Convert all files in the data directory to a single CSV file."""
        if not self.data_directory.exists():
            logger.error(f"Directory {self.data_directory} does not exist!")
            return None

        # Find all files recursively
        txt_files = list(self.data_directory.rglob('*.txt'))
        csv_files = list(self.data_directory.rglob('*.csv'))

        # Find additional 2022-2024 CSV files
        additional_csv = self._find_additional_csv_files()
        csv_files.extend(additional_csv)

        # Remove duplicates
        csv_files = list(set(csv_files))

        all_files = sorted(txt_files + csv_files, key=lambda x: str(x))

        if not all_files:
            logger.error(f"No .txt or .csv files found in {self.data_directory}")
            return None

        logger.info(f"Found {len(txt_files)} TXT files and {len(csv_files)} CSV files")
        for file in all_files:
            logger.info(f"  - {file.relative_to(self.data_directory)}")

        # Process all files
        all_data = []
        for file_path in all_files:
            file_data, file_type = self._process_file(file_path)

            if file_data:
                all_data.extend(file_data)
                self.processed_files.append({
                    'file': file_path.name,
                    'type': file_type,
                    'rows': len(file_data)
                })
                logger.info(f"  -> Added {len(file_data)} rows from {file_type}")
            else:
                self.failed_files.append({
                    'file': file_path.name,
                    'type': file_type,
                    'error': 'No data extracted'
                })
                logger.warning(f"  -> No data extracted from {file_path.name}")

            self.total_rows_processed = len(all_data)
            logger.info(f"  -> Total rows so far: {self.total_rows_processed}")

        if not all_data:
            logger.error("No data was extracted from any files!")
            logger.error(f"Failed file details: {self.failed_files}")
            return None

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Data validation and cleaning
        if validate_data:
            df = self._validate_and_clean_data(df)

        # IMPROVED DATE HANDLING - Don't drop rows with valid dates
        logger.info("Processing dates for sorting...")

        try:
            # Count valid dates before processing
            valid_dates_before = df['SEANCE'].apply(self._validate_date).sum()
            logger.info(f"Valid dates before processing: {valid_dates_before}/{len(df)}")

            # Parse dates for sorting only - don't filter out invalid ones yet
            df['SEANCE_PARSED'] = pd.to_datetime(df['SEANCE'], format='%d/%m/%Y', errors='coerce')

            # Sort by parsed date where available, then by company name
            df = df.sort_values(['SEANCE_PARSED', 'VALEUR'], na_position='last')

            # Drop the parsing column
            df = df.drop('SEANCE_PARSED', axis=1)

            logger.info(f"Data sorted successfully. Total rows preserved: {len(df)}")

        except Exception as e:
            logger.warning(f"Could not sort by date: {e}. Sorting by company name only.")
            df = df.sort_values(['VALEUR'])

        # Save to CSV
        try:
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            logger.info(f"\nSuccessfully created {self.output_file}")

            # Print comprehensive summary
            self._print_comprehensive_summary(df)

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return None

        return df

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data."""
        logger.info("Validating and cleaning data...")

        original_len = len(df)

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Remove rows without essential data (SEANCE, CODE, VALEUR)
        # But be more lenient with date validation
        essential_columns = ['CODE', 'VALEUR']  # Don't require SEANCE here
        df = df.dropna(subset=essential_columns)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Clean string columns
        string_columns = ['VALEUR', 'CODE', 'GROUPE']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        logger.info(f"Data cleaning: {original_len} -> {len(df)} rows ({original_len - len(df)} removed)")

        return df

    def _print_comprehensive_summary(self, df: pd.DataFrame) -> None:
        """Print comprehensive summary statistics of the processed data."""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)

        logger.info(f"Total files processed: {len(self.processed_files)}")
        logger.info(f"Failed files: {len(self.failed_files)}")
        logger.info(f"Total rows in final dataset: {len(df)}")

        # File processing summary
        if self.processed_files:
            logger.info("\nSuccessfully processed files:")
            for file_info in self.processed_files:
                logger.info(f"  - {file_info['file']}: {file_info['rows']} rows ({file_info['type']})")

        if self.failed_files:
            logger.info("\nFailed files:")
            for file_info in self.failed_files:
                logger.info(f"  - {file_info['file']}: {file_info['error']}")

        # Data summary
        if len(df) > 0:
            logger.info(f"\nDATA OVERVIEW:")

            # Date analysis
            if 'SEANCE' in df.columns and df['SEANCE'].notna().any():
                valid_dates = df[df['SEANCE'].apply(self._validate_date)]
                invalid_dates = len(df) - len(valid_dates)

                logger.info(f"Valid dates: {len(valid_dates)}/{len(df)} ({len(valid_dates) / len(df) * 100:.1f}%)")
                if invalid_dates > 0:
                    logger.warning(f"Invalid dates: {invalid_dates}")

                if len(valid_dates) > 0:
                    try:
                        date_series = pd.to_datetime(valid_dates['SEANCE'], format='%d/%m/%Y', errors='coerce')
                        valid_parsed = date_series.dropna()
                        if len(valid_parsed) > 0:
                            logger.info(
                                f"Date range: {valid_parsed.min().strftime('%d/%m/%Y')} to {valid_parsed.max().strftime('%d/%m/%Y')}")
                            logger.info(f"Total trading days: {valid_parsed.nunique()}")

                            # Year distribution
                            year_dist = valid_parsed.dt.year.value_counts().sort_index()
                            logger.info("Year distribution:")
                            for year, count in year_dist.items():
                                logger.info(f"  {year}: {count:,} records")
                    except Exception as e:
                        logger.info(f"Date range: Unable to parse dates - {e}")

            if 'VALEUR' in df.columns and df['VALEUR'].notna().any():
                logger.info(f"Unique companies: {df['VALEUR'].nunique()}")

            if 'QUANTITE_NEGOCIEE' in df.columns:
                total_volume = df['QUANTITE_NEGOCIEE'].sum()
                logger.info(f"Total volume traded: {total_volume:,.0f}")

            if 'NB_TRANSACTION' in df.columns:
                total_transactions = df['NB_TRANSACTION'].sum()
                logger.info(f"Total number of transactions: {total_transactions:,.0f}")

            # Sample data preview
            logger.info("\nSample data (first 3 rows):")
            logger.info("-" * 100)
            sample_df = df.head(3)
            for _, row in sample_df.iterrows():
                company_name = str(row['VALEUR'])[:30] if pd.notna(row['VALEUR']) else 'N/A'
                close_price = row['CLOTURE'] if pd.notna(row['CLOTURE']) else 'N/A'
                volume = row['QUANTITE_NEGOCIEE'] if pd.notna(row['QUANTITE_NEGOCIEE']) else 'N/A'
                logger.info(f"Date: {row['SEANCE']}, Company: {company_name:<30}, "
                            f"Close: {close_price}, Volume: {volume}")

        logger.info("=" * 60)


def main():
    """Main execution function with enhanced error handling."""
    DATA_DIRECTORY = './data/'
    OUTPUT_FILE = 'combined_stock_data2.csv'

    try:
        converter = StockDataConverter(DATA_DIRECTORY, OUTPUT_FILE)
        result_df = converter.convert_to_csv(validate_data=True)

        if result_df is not None:
            logger.info(f"\n🎉 Conversion completed successfully!")
            logger.info(f"📁 Output file: {OUTPUT_FILE}")
            logger.info(f"📊 Total records: {len(result_df):,}")

            # Create visualization of year distribution
            if 'SEANCE' in result_df.columns:
                logger.info("Creating year distribution visualization...")

                # Parse dates and extract years for valid dates only
                valid_dates_mask = result_df['SEANCE'].apply(converter._validate_date)
                valid_dates_df = result_df[valid_dates_mask].copy()

                if len(valid_dates_df) > 0:
                    try:
                        valid_dates_df['Année'] = pd.to_datetime(
                            valid_dates_df['SEANCE'],
                            format='%d/%m/%Y',
                            errors='coerce'
                        ).dt.year

                        year_counts = valid_dates_df['Année'].value_counts().sort_index()

                        # Create the plot
                        plt.figure(figsize=(12, 8))
                        bars = plt.bar(year_counts.index, year_counts.values,
                                       color='skyblue', edgecolor='navy', alpha=0.7)

                        # Add value labels on bars
                        for bar, count in zip(bars, year_counts.values):
                            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + count * 0.01,
                                     f'{count:,}', ha='center', va='bottom', fontweight='bold')

                        plt.title('Distribution of Stock Data Records by Year',
                                  fontsize=16, fontweight='bold', pad=20)
                        plt.xlabel('Year', fontsize=12)
                        plt.ylabel('Number of Records', fontsize=12)
                        plt.xticks(rotation=45)
                        plt.grid(axis='y', alpha=0.3)

                        # Add statistics text
                        total_valid = len(valid_dates_df)
                        total_records = len(result_df)
                        plt.text(0.02, 0.98,
                                 f'Total Records: {total_records:,}\n'
                                 f'Valid Dates: {total_valid:,} ({total_valid / total_records * 100:.1f}%)\n'
                                 f'Date Range: {year_counts.index.min()}-{year_counts.index.max()}',
                                 transform=plt.gca().transAxes,
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                        plt.tight_layout()
                        plt.savefig('year_distribution.png', dpi=300, bbox_inches='tight')
                        logger.info("📊 Year distribution chart saved as 'year_distribution.png'")

                        # Show detailed year statistics
                        logger.info(f"\nDetailed Year Statistics:")
                        logger.info(
                            f"Valid date records: {total_valid:,} out of {total_records:,} total ({total_valid / total_records * 100:.1f}%)")
                        logger.info("Records per year:")
                        for year, count in year_counts.items():
                            percentage = (count / total_valid) * 100
                            logger.info(f"  {year}: {count:,} records ({percentage:.1f}%)")

                        plt.show()

                    except Exception as e:
                        logger.error(f"Error creating visualization: {e}")

                else:
                    logger.warning("No valid dates found for visualization")

            # Additional data quality report
            logger.info("\n" + "=" * 60)
            logger.info("DATA QUALITY REPORT")
            logger.info("=" * 60)

            # Check for missing values in key columns
            key_columns = ['SEANCE', 'CODE', 'VALEUR', 'CLOTURE']
            for col in key_columns:
                if col in result_df.columns:
                    missing_count = result_df[col].isna().sum()
                    missing_pct = (missing_count / len(result_df)) * 100
                    logger.info(f"{col}: {missing_count:,} missing ({missing_pct:.1f}%)")

            # Check for zero/null prices
            if 'CLOTURE' in result_df.columns:
                zero_prices = (result_df['CLOTURE'] == 0).sum()
                null_prices = result_df['CLOTURE'].isna().sum()
                logger.info(f"Zero closing prices: {zero_prices:,}")
                logger.info(f"Null closing prices: {null_prices:,}")

            # Top companies by number of records
            if 'VALEUR' in result_df.columns:
                top_companies = result_df['VALEUR'].value_counts().head(10)
                logger.info(f"\nTop 10 companies by number of records:")
                for company, count in top_companies.items():
                    logger.info(f"  {company}: {count:,} records")

        else:
            logger.error("❌ Conversion failed!")
            return 1

    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())