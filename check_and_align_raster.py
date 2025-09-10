import rasterio
import numpy as np
import pandas as pd
import os
from rasterio.warp import reproject, Resampling

def check_and_align_raster(reference_path, source_path, output_dir="temp_aligned"):
    """
    Verifica se um raster de origem está alinhado com um de referência.
    Se não estiver, cria uma versão alinhada.
    Retorna o caminho para o arquivo alinhado (novo ou original).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(reference_path) as ref, rasterio.open(source_path) as src:
        # Compara as propriedades geoespaciais
        if (ref.crs == src.crs and 
            ref.transform == src.transform and 
            ref.shape == src.shape):
            print(f"  - '{os.path.basename(source_path)}' já está alinhado.")
            return source_path # Retorna o caminho original

        print(f"  - '{os.path.basename(source_path)}' não está alinhado. Reprojetando...")
        
        # Define o caminho para o novo arquivo alinhado
        aligned_filename = os.path.splitext(os.path.basename(source_path))[0] + "_aligned.tif"
        aligned_path = os.path.join(output_dir, aligned_filename)
        
        # Copia o perfil do raster de referência
        profile = ref.profile.copy()
        profile.update({'compress': 'lzw'})

        with rasterio.open(aligned_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.nearest # Essencial para dados de classe
            )
        
        print(f"  - Arquivo alinhado criado em: '{aligned_path}'")
        return aligned_path