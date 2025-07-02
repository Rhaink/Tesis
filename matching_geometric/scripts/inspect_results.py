#!/usr/bin/env python3
"""
Inspect the structure of saved results file.
"""

import pickle
import os

PROJECT_ROOT = '/home/donrobot/Projects/Tesis'

def inspect_results():
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("Structure of results file:")
    print(f"Type: {type(results)}")
    
    if isinstance(results, dict):
        print(f"Keys: {list(results.keys())}")
        
        for key, value in results.items():
            print(f"\nKey: {key}")
            print(f"  Type: {type(value)}")
            
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First element type: {type(value[0])}")
                    if hasattr(value[0], 'shape'):
                        print(f"  First element shape: {value[0].shape}")
            elif isinstance(value, dict):
                print(f"  Sub-keys: {list(value.keys())[:5]}...")
    
    # Look for individual image results
    if 'predictions' in results:
        predictions = results['predictions']
        print(f"\nPredictions type: {type(predictions)}")
        if hasattr(predictions, 'shape'):
            print(f"Predictions shape: {predictions.shape}")
        elif isinstance(predictions, list):
            print(f"Predictions length: {len(predictions)}")
            if len(predictions) > 0:
                print(f"First prediction shape: {predictions[0].shape}")
    
    if 'image_names' in results:
        image_names = results['image_names']
        print(f"\nImage names type: {type(image_names)}")
        if isinstance(image_names, list):
            print(f"Number of images: {len(image_names)}")
            print(f"First 5 image names: {image_names[:5]}")

if __name__ == "__main__":
    inspect_results()