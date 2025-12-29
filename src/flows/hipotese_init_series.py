# -*- coding: utf-8 -*-
# src/flows/hipotese_init_series.py
"""
Requisitos: pip install -r requirements.txt
"""

# ==========================
# Imports
# ==========================

# Caminho absoluto para o diretório 'src'
import sys
import os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, src_path)

# Módulos utilizados
from modules.data_io import DataIO                          # Entrada e saída de dados em formato NetCDF (.nc)
from modules.dataset import StationDataset                  # Dados de uma estação hidrológica específica
from modules.preprocessing import Preprocessor              # Pré-processamento dos dados
from modules.forecast_experiment import ForecastExperiment  # Configuração e execução de experimentos de previsão
from modules.evaluation import Evaluator                    # Alinhamento entre observado e previsto
from modules.report import save_report                      # Relatório completo do experimento

# =====================================================
# Fluxo principal
# =====================================================
def flow():

    # Obtém o diretório absoluto do arquivo atual
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define o caminho da pasta onde estão os arquivos NetCDF (.nc)
    # ".." indica um nível acima do diretório atual
    nc_folder = os.path.join(BASE_DIR, '..', 'dataset')

    # Cria um objeto DataIO responsável por operações de entrada/saída
    # (listagem e abertura de arquivos NetCDF)
    reader = DataIO(nc_folder)

    # Lista todos os arquivos .nc existentes na pasta dataset
    files = reader.list_files()

    # Abre o primeiro arquivo NetCDF encontrado
    # Retorna um objeto xarray.Dataset
    ds = reader.open_dataset(files[0])

    # Cria um objeto StationDataset que representa uma estação específica
    # station_id será usado como identificador único da série temporal
    station = StationDataset(ds, station_id="gage_001")

    # Converte o xarray.Dataset para um pandas.DataFrame
    # - time_col: coluna temporal
    # - target_col: variável alvo (vazão)
    # - exog_cols: variáveis exógenas (ex.: precipitação)
    df = station.to_dataframe(time_col="DateTime",
                              target_col="Streamflow",
                              exog_cols=["Precipitation"])

    # Aplica o pré-processador aos dados
    # Atualmente não modifica o DataFrame, mas mantém a interface fit/transform
    # para permitir extensões futuras (normalização, filtros, etc.)
    df = Preprocessor().fit_transform(df)

    # Cria um experimento de previsão configurando:
    # - freq: frequência temporal dos dados ("H" = horário)
    # - input_size: tamanho da janela de entrada (ex.: 168 horas = 7 dias)
    # - horizon: horizonte de previsão (ex.: 24 horas à frente)
    # - max_steps: número máximo de iterações de treinamento
    experiment = ForecastExperiment(freq="H",
                                    input_size=168,
                                    horizon=24,
                                    max_steps=200)

    # Executa validação cruzada temporal com janelas deslizantes
    # - n_windows: número de janelas de validação
    # - step_size: deslocamento entre janelas
    # Retorna um DataFrame com previsões para cada modelo
    cv_results = experiment.cross_validate(df=df,
                                           n_windows=3,
                                           step_size=168)

    # Cria um avaliador que compara valores observados com previstos
    evaluator = Evaluator(df, cv_results)

    # Avalia os modelos utilizando métricas estatísticas
    # Os nomes dos modelos são obtidos dinamicamente a partir das instâncias
    metrics = evaluator.evaluate([m.__class__.__name__ for m in experiment.models])

    # Imprime no console um resumo das métricas calculadas
    evaluator.summary(metrics)

    # Gera um relatório completo contendo:
    # - Informações do dataset
    # - Configuração dos modelos
    # - Resultados da validação cruzada
    # - Métricas de desempenho
    # - Gráficos salvos em disco
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