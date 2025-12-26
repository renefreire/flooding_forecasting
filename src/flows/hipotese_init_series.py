# -*- coding: utf-8 -*-
# src/flows/hipotese_init_series.py
"""
Requisitos: pip install -r requirements.txt
"""

# ==========================
# Imports
# ==========================
import sys
import os

# Caminho absoluto para o diretório 'src'
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

# Módulos utilizados
from modules.data_io import DataIO
from modules.dataset import StationDataset
from modules.preprocessing import Preprocessor
from modules.forecast_experiment import ForecastExperiment
from modules.evaluation import Evaluator
from modules.report import save_report

# =====================================================
# Fluxo principal
# =====================================================
def flow():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    nc_folder = os.path.join(BASE_DIR, '..', 'dataset')

    reader = DataIO(nc_folder)
    files = reader.list_files()

    ds = reader.open_dataset(files[0])

    station = StationDataset(ds, station_id="gage_001")
    df = station.to_dataframe(time_col="DateTime",
                              target_col="Streamflow",
                              exog_cols=["Precipitation"])

    df = Preprocessor().fit_transform(df)

    experiment = ForecastExperiment(freq="H",
                                    input_size=168,
                                    horizon=24,
                                    max_steps=200)

    cv_results = experiment.cross_validate(df=df,
                                           n_windows=3,
                                           step_size=168)

    evaluator = Evaluator(df, cv_results)
    metrics = evaluator.evaluate([m.__class__.__name__ for m in experiment.models])
    evaluator.summary(metrics)

    save_report(station,
                nc_folder,
                files,
                df,
                experiment,
                cv_results,
                metrics,
                evaluator)

# =====================================================
# Execução do fluxo principal
# =====================================================
if __name__ == "__main__":
    flow()

# --------------------------
# Observações finais do script:
# - para rodar em múltiplos gauges, faça loop sobre arquivos NetCDF, construa um df concatenado com 
#   multiple unique_id
# - Ajuste hyperparams: max_steps, batch_size, input_size, learning_rate
# - Para saídas probabilísticas (DeepAR, etc.), use colunas probabilísticas retornadas e calcule CRPS 
#   (biblioteca properscoring ajuda)