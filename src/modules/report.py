from contextlib import contextmanager
import sys
from datetime import datetime

@contextmanager
def save_report(report_path):
    """
    Redireciona todos os prints para um arquivo de texto,
    mantendo também a saída no terminal.
    """
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    with open(report_path, 'w', encoding='utf-8') as f:
        header = (
            "=" * 80 + "\n"
            "RELATÓRIO DE BENCHMARK HIDROLÓGICO\n"
            f"Data/Hora: {datetime.now()}\n"
            "=" * 80 + "\n\n"
        )
        f.write(header)
        sys.stdout = Tee(sys.stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout

def generate_report(report_file):
    with save_report(report_file):

        print("=== INÍCIO DO SCRIPT ===\n")

        print("Pasta dataset:", nc_folder)
        print(ds_all)

        print("\nPrimeiras linhas do DataFrame:")
        print(df.head())

        print("\nResultados da validação cruzada:")
        print(cv_results.head())

        print("\n=== MÉTRICAS FINAIS ===")
        for model, mets in metrics.items():
            print(f'\nModel: {model}')
            for k, v in mets.items():
                print(f'  {k}: {np.nanmean(v):.4f}')

        print("\n=== FIM DO SCRIPT ===")