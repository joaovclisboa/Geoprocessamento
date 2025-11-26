import logging
import os

import pandas as pd


# ---------------- CONFIGURAÇÃO ESTÁTICA ----------------
# Ajuste os caminhos e opções conforme necessário antes de executar o script.
PARQUET_INPUT = "arroz_concat_to_20251125_sentinel_idx.parquet"  # caminho para o arquivo .parquet de entrada
CSV_OUTPUT = None  # caminho para o .csv de saída; use None para gerar automaticamente
KEEP_INDEX = False  # True para manter o índice do DataFrame no CSV
VERBOSE = False  # True para exibir logs em nível DEBUG
# -------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def convert_parquet_to_csv(parquet_path: str, csv_path: str, index: bool = False) -> None:
    logging.info("Lendo arquivo Parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logging.debug("DataFrame carregado com %d linhas e %d colunas", len(df), len(df.columns))

    logging.info("Salvando CSV em: %s", csv_path)
    df.to_csv(csv_path, index=index)
    logging.info("Conversão concluída")


def resolve_output_path(parquet_path: str, csv_path: str | None) -> str:
    if csv_path:
        return csv_path
    base, _ = os.path.splitext(parquet_path)
    return base + ".csv"


def main() -> None:
    if PARQUET_INPUT is None:
        raise ValueError("Defina PARQUET_INPUT com o caminho do arquivo .parquet de entrada.")

    configure_logging(VERBOSE)

    parquet_path = PARQUET_INPUT
    csv_path = resolve_output_path(parquet_path, CSV_OUTPUT)

    logging.info("Configuração: parquet=%s, csv=%s, keep_index=%s", parquet_path, csv_path, KEEP_INDEX)
    convert_parquet_to_csv(parquet_path, csv_path, index=KEEP_INDEX)


if __name__ == "__main__":
    main()
