"""
Script para verificar la estructura del dataset después de la descarga.
"""

import os
from pathlib import Path

def verify_dataset_structure():
    """Verifica que el dataset esté correctamente organizado."""
    
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / 'data' / 'raw' / 'chest_xray'
    
    print("🔍 VERIFICACIÓN DEL DATASET")
    print("=" * 50)
    
    if not dataset_path.exists():
        print("❌ No se encontró el dataset en:", dataset_path)
        return False
    
    # Verificar estructura esperada
    expected_structure = {
        'train': ['NORMAL', 'PNEUMONIA'],
        'test': ['NORMAL', 'PNEUMONIA'], 
        'val': ['NORMAL', 'PNEUMONIA']
    }
    
    total_images = 0
    all_good = True
    
    for split, classes in expected_structure.items():
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"❌ No se encontró el directorio: {split}")
            all_good = False
            continue
            
        print(f"\n📁 {split.upper()}:")
        split_total = 0
        
        for class_name in classes:
            class_path = split_path / class_name
            
            if not class_path.exists():
                print(f"  ❌ No se encontró: {class_name}")
                all_good = False
                continue
            
            # Contar imágenes (jpg, jpeg, png)
            image_files = (
                list(class_path.glob('*.jpeg')) +
                list(class_path.glob('*.jpg')) +
                list(class_path.glob('*.png'))
            )
            
            count = len(image_files)
            split_total += count
            total_images += count
            
            status = "✅" if count > 0 else "⚠️"
            print(f"  {status} {class_name}: {count} imágenes")
        
        print(f"  📊 Total {split}: {split_total} imágenes")
    
    print("\n" + "=" * 50)
    print(f"📈 TOTAL DEL DATASET: {total_images} imágenes")
    
    if all_good and total_images > 0:
        print("✅ Dataset verificado correctamente!")
        
        # Mostrar distribución de clases
        print("\n📊 DISTRIBUCIÓN POR CLASE:")
        for split in expected_structure.keys():
            normal_count = len(list((dataset_path / split / 'NORMAL').glob('*.jpeg')))
            pneumonia_count = len(list((dataset_path / split / 'PNEUMONIA').glob('*.jpeg')))
            total_split = normal_count + pneumonia_count
            
            if total_split > 0:
                normal_pct = (normal_count / total_split) * 100
                pneumonia_pct = (pneumonia_count / total_split) * 100
                print(f"  {split.upper()}: Normal {normal_pct:.1f}% | Pneumonia {pneumonia_pct:.1f}%")
        
        return True
    else:
        print("❌ Hay problemas con la estructura del dataset")
        return False

if __name__ == "__main__":
    verify_dataset_structure()