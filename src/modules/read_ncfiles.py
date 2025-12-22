import os
import glob
import xarray as xr
import dask

def read_ncfiles(rootfolder):
    nc_folder = os.path.join(rootfolder, '..', 'dataset')
    nc_folder = os.path.abspath(nc_folder)
    pattern = os.path.join(nc_folder, '*.nc')

    # Debug seguro
    print("Pasta dataset:", nc_folder)
    #print("Arquivos encontrados:", glob.glob(pattern))
    nc_files = glob.glob(pattern)
    if not nc_files:
        raise FileNotFoundError(f"Nenhum arquivo .nc encontrado em {nc_folder}")

    # Use open_mfdataset para abrir e concatenar todos os arquivos automaticamente
    # concat_dim='time' assume que o eixo de interesse é o tempo
    ds_all = xr.open_mfdataset(pattern,
                            combine='by_coords',
                            #concat_dim='time',
                            parallel=True)  # precisa dask instalado, mas acelera pra muitos arquivos

    print(ds_all)