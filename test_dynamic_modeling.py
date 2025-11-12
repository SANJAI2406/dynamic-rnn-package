"""
TEST DYNAMIC MODELING - Synthetic Data Demo
============================================

This script creates synthetic dynamic data and tests the full pipeline.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/claude')

from ENHANCED_DYNAMIC_RNN import (
    DynamicDataLoader,
    DynamicModelTrainer,
    DynamicVisualization
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_synthetic_dynamic_data(n_configs=50, n_freq=1000, 
                                   n_inputs=5, n_outputs=2,
                                   freq_min=1, freq_max=4000):
    """
    Generate synthetic dynamic data for testing.
    
    Simulates a gear dynamics system with:
    - Design parameters affect resonance frequencies and damping
    - Frequency responses show peaks and valleys
    - Realistic nonlinear relationships
    """
    print("Generating synthetic dynamic data...")
    
    # Frequency array
    frequency = np.linspace(freq_min, freq_max, n_freq)
    
    # Generate random input configurations
    np.random.seed(42)
    X = np.random.uniform(low=[3, 3, -20, -20, 0],
                         high=[10, 10, 20, 20, 10],
                         size=(n_configs, n_inputs))
    
    input_names = [f"Input_Param_{i+1}" for i in range(n_inputs)]
    output_names = [f"Output_{i+1}" for i in range(n_outputs)]
    
    # Generate frequency responses with realistic dynamics
    Y = np.zeros((n_configs, n_freq, n_outputs))
    
    for i in range(n_configs):
        # Extract design parameters
        param1, param2, param3, param4, param5 = X[i]
        
        # Simulate resonance frequencies (influenced by design params)
        resonance1 = 500 + param1 * 50 + param2 * 30
        resonance2 = 1500 + param3 * 20 + param4 * 15
        resonance3 = 2500 + param5 * 40
        
        # Simulate damping (influenced by design params)
        damping1 = 0.05 + param1 * 0.002
        damping2 = 0.08 + param2 * 0.003
        damping3 = 0.10 + param5 * 0.004
        
        # Baseline response
        baseline1 = 1.0 + param1 * 0.1
        baseline2 = 2.0 + param2 * 0.15
        
        # Generate Output 1: Transmission error (vibration amplitude)
        response1 = baseline1 * np.ones(n_freq)
        
        # Add resonance peaks (Lorentzian)
        response1 += 5.0 / (1 + ((frequency - resonance1) / (damping1 * resonance1))**2)
        response1 += 3.0 / (1 + ((frequency - resonance2) / (damping2 * resonance2))**2)
        response1 += 2.0 / (1 + ((frequency - resonance3) / (damping3 * resonance3))**2)
        
        # Add some noise
        response1 += np.random.normal(0, 0.05, n_freq)
        
        Y[i, :, 0] = response1
        
        # Generate Output 2: Force (different frequency response)
        response2 = baseline2 * np.ones(n_freq)
        
        # Different resonance characteristics
        response2 += 10.0 / (1 + ((frequency - resonance1 * 0.9) / (damping1 * resonance1 * 1.2))**2)
        response2 += 6.0 / (1 + ((frequency - resonance2 * 1.1) / (damping2 * resonance2 * 0.8))**2)
        
        # Add frequency-dependent term
        response2 += 0.001 * frequency
        
        # Add noise
        response2 += np.random.normal(0, 0.1, n_freq)
        
        Y[i, :, 1] = response2
    
    print(f"✓ Generated {n_configs} configurations")
    print(f"✓ Frequency range: {freq_min} - {freq_max} Hz")
    print(f"✓ Frequency points: {n_freq}")
    
    return X, Y, frequency, input_names, output_names


def create_dynamic_dataframe(X, Y, frequency, input_names, output_names):
    """
    Create dataframe in the required format for dynamic modeling.
    """
    n_configs, n_freq, n_outputs = Y.shape
    total_rows = n_configs * n_freq
    
    # Initialize lists for each column
    freq_col = []
    input_cols = {name: [] for name in input_names}
    output_cols = {name: [] for name in output_names}
    
    # Fill data
    for i in range(n_configs):
        # Repeat frequency array
        freq_col.extend(frequency)
        
        # Repeat input parameters for all frequencies
        for j, name in enumerate(input_names):
            input_cols[name].extend([X[i, j]] * n_freq)
        
        # Add frequency-varying outputs
        for j, name in enumerate(output_names):
            output_cols[name].extend(Y[i, :, j])
    
    # Create dataframe
    data_dict = {'Frequency': freq_col}
    data_dict.update(input_cols)
    data_dict.update(output_cols)
    
    df = pd.DataFrame(data_dict)
    
    print(f"\n✓ Created dataframe:")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Structure verified: {len(df) == total_rows}")
    
    return df


def main():
    """
    Main test function.
    """
    print("="*80)
    print("TESTING DYNAMIC MODELING PIPELINE")
    print("="*80)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic data...")
    X, Y, frequency, input_names, output_names = generate_synthetic_dynamic_data(
        n_configs=50,
        n_freq=1000,
        n_inputs=5,
        n_outputs=2,
        freq_min=1,
        freq_max=4000
    )
    
    # Step 2: Create dataframe and save
    print("\n[2/5] Creating dataframe...")
    df = create_dynamic_dataframe(X, Y, frequency, input_names, output_names)
    
    test_file = '/mnt/user-data/outputs/synthetic_dynamic_data.csv'
    df.to_csv(test_file, index=False)
    print(f"✓ Saved test data: {test_file}")
    
    # Step 3: Load using DynamicDataLoader
    print("\n[3/5] Testing DynamicDataLoader...")
    loader = DynamicDataLoader(test_file)
    data_dict = loader.load_and_reshape()
    
    # Verify
    assert data_dict['X'].shape == X.shape, "X shape mismatch!"
    assert data_dict['Y'].shape == Y.shape, "Y shape mismatch!"
    assert np.allclose(data_dict['frequency'], frequency), "Frequency mismatch!"
    print("✓ Data loading verified!")
    
    # Step 4: Train models
    print("\n[4/5] Training dynamic models...")
    trainer = DynamicModelTrainer(n_components=30, variance_threshold=0.99)
    stats = trainer.train_all_outputs(
        data_dict['X'],
        data_dict['Y'],
        data_dict['frequency'],
        data_dict['input_names'],
        data_dict['output_names']
    )
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    
    # Training summary
    fig1 = DynamicVisualization.plot_training_summary(stats, output_names)
    fig1.savefig('/mnt/user-data/outputs/test_training_summary.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: test_training_summary.png")
    plt.close(fig1)
    
    # For each output
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_dict['X'], data_dict['Y'], test_size=0.2, random_state=42
    )
    
    for idx, output_name in enumerate(output_names):
        model = trainer.models[output_name]
        Y_test_single = Y_test[:, :, idx]
        
        # Predictions
        Y_pred = model.predict(X_test)
        Y_pred_unc, Y_std = model.predict_with_uncertainty(X_test)
        
        # Frequency response
        fig2 = DynamicVisualization.plot_frequency_response_prediction(
            frequency, Y_test_single, Y_pred, Y_std,
            config_idx=0, output_name=output_name
        )
        fig2.savefig(f'/mnt/user-data/outputs/test_freq_response_{output_name}.png',
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: test_freq_response_{output_name}.png")
        plt.close(fig2)
        
        # Prediction grid
        fig3 = DynamicVisualization.plot_prediction_grid(
            frequency, Y_test_single, Y_pred, output_name, n_samples=4
        )
        fig3.savefig(f'/mnt/user-data/outputs/test_prediction_grid_{output_name}.png',
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: test_prediction_grid_{output_name}.png")
        plt.close(fig3)
        
        # Error analysis
        fig4 = DynamicVisualization.plot_error_analysis(
            frequency, Y_test_single, Y_pred, output_name
        )
        fig4.savefig(f'/mnt/user-data/outputs/test_error_analysis_{output_name}.png',
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: test_error_analysis_{output_name}.png")
        plt.close(fig4)
        
        # PCA analysis
        fig5 = DynamicVisualization.plot_pca_analysis(
            model, frequency, output_name
        )
        fig5.savefig(f'/mnt/user-data/outputs/test_pca_analysis_{output_name}.png',
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: test_pca_analysis_{output_name}.png")
        plt.close(fig5)
    
    # Print final statistics
    print("\n" + "="*80)
    print("TEST COMPLETE - RESULTS SUMMARY")
    print("="*80)
    
    for output_name in output_names:
        print(f"\n{output_name}:")
        print(f"  R² Train:  {stats[output_name]['r2_train']:.4f}")
        print(f"  R² Test:   {stats[output_name]['r2_test']:.4f}")
        print(f"  RMSE Test: {stats[output_name]['rmse_test']:.4f}")
        print(f"  PCs Used:  {stats[output_name]['n_components']}")
        print(f"  Variance:  {stats[output_name]['variance_explained']*100:.2f}%")
    
    print("\n" + "="*80)
    print("All visualizations saved to: /mnt/user-data/outputs/")
    print("="*80)
    
    return trainer, stats, data_dict


if __name__ == "__main__":
    trainer, stats, data_dict = main()
    print("\n✅ Test completed successfully!")
