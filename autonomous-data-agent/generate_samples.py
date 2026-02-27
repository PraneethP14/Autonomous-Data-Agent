"""
Generate sample datasets for demonstration and testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_ecommerce_dataset():
    """
    Create a realistic e-commerce dataset with data quality issues:
    - Missing values in various columns
    - Duplicate entries
    - Invalid dates  and formats
    - Outliers
    """
    np.random.seed(42)
    n_rows = 1000
    
    # Generate customer ages with some missing and outliers
    ages = []
    for i in range(n_rows):
        rand = np.random.random()
        if rand < 0.05:  # 5% missing
            ages.append(None)
        elif rand < 0.08:  # 3% outliers/invalid
            ages.append(np.random.choice([999, -5, 200]))
        else:  # Normal ages
            ages.append(np.random.randint(18, 75))
    
    data = {
        'order_id': range(1001, 1001 + n_rows),
        'customer_id': np.random.randint(1, 200, n_rows),
        'product_category': np.random.choice(
            ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', None],
            n_rows,
            p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05]
        ),
        'order_value': np.random.exponential(50, n_rows),
        'purchase_date': [datetime.now() - timedelta(days=int(x)) for x in np.random.exponential(30, n_rows)],
        'customer_email': [f"customer{i}@email.com" if np.random.random() > 0.08 else None for i in range(n_rows)],
        'shipping_address': np.random.choice(
            ['123 Main St', '456 Oak Ave', '789 Pine Rd', None, 'INVALID_ADDRESS'],
            n_rows,
            p=[0.3, 0.3, 0.25, 0.1, 0.05]
        ),
        'order_status': np.random.choice(
            ['Delivered', 'Processing', 'Shipped', 'Cancelled', None],
            n_rows,
            p=[0.5, 0.2, 0.15, 0.1, 0.05]
        ),
        'payment_method': np.random.choice(
            ['Credit Card', 'Debit Card', 'PayPal', 'Google Pay', None],
            n_rows,
            p=[0.4, 0.3, 0.15, 0.1, 0.05]
        ),
        'discount_applied': np.random.choice([True, False, None], n_rows, p=[0.3, 0.6, 0.1]),
        'customer_age': ages
    }
    
    df = pd.DataFrame(data)
    
    # Add some duplicates
    dup_indices = np.random.choice(df.index, size=50, replace=False)
    df = pd.concat([df, df.loc[dup_indices]], ignore_index=True)
    
    # Add a completely empty column
    df['unused_field'] = np.nan
    
    return df.sample(frac=1).reset_index(drop=True)

def create_medical_dataset():
    """
    Create a medical dataset with realistic issues:
    - Missing medical readings
    - Duplicate patient records
    - Invalid date formats
    - Outlier measurements
    """
    np.random.seed(123)
    n_rows = 800
    
    # Ensure patient_id has same length as n_rows
    num_patients = n_rows // 4
    patient_ids = list(range(1, num_patients + 1)) * 4
    patient_ids = patient_ids[:n_rows]  # Trim to exact length
    
    data = {
        'patient_id': patient_ids,
        'visit_date': [datetime.now() - timedelta(days=int(x)) for x in np.random.exponential(60, n_rows)],
        'blood_pressure_systolic': np.random.normal(120, 15, n_rows),
        'blood_pressure_diastolic': np.random.normal(80, 10, n_rows),
        'heart_rate': np.random.normal(72, 12, n_rows),
        'temperature': np.random.normal(98.6, 0.5, n_rows),
        'diagnosis': np.random.choice(
            ['Hypertension', 'Diabetes', 'Asthma', 'None', None],
            n_rows,
            p=[0.3, 0.25, 0.2, 0.2, 0.05]
        ),
        'medication': np.random.choice(
            ['Lisinopril', 'Metformin', 'Albuterol', 'None', None],
            n_rows,
            p=[0.25, 0.25, 0.2, 0.2, 0.1]
        ),
        'doctor_notes': np.random.choice(
            ['Patient stable', 'Follow-up needed', 'Medication adjusted', None, ''],
            n_rows,
            p=[0.5, 0.2, 0.15, 0.1, 0.05]
        )
    }
    
    df = pd.DataFrame(data)
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[outlier_indices, 'blood_pressure_systolic'] = np.random.uniform(200, 300, 30)
    df.loc[outlier_indices, 'heart_rate'] = np.random.uniform(200, 300, 30)
    
    # Add duplicates
    dup_indices = np.random.choice(df.index, size=40, replace=False)
    df = pd.concat([df, df.loc[dup_indices]], ignore_index=True)
    
    return df.sample(frac=1).reset_index(drop=True)

def create_sample_datasets():
    """Create and save sample datasets"""
    
    print("Generating sample datasets...")
    
    # Create data/samples directory
    samples_dir = Path('data/samples')
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # E-commerce dataset
    print("\n[1] E-commerce Dataset")
    df_ecom = create_ecommerce_dataset()
    ecom_path = samples_dir / 'ecommerce_orders.csv'
    df_ecom.to_csv(ecom_path, index=False)
    print(f"    Shape: {df_ecom.shape}")
    print(f"    Missing: {df_ecom.isnull().sum().sum()} cells")
    print(f"    Duplicates: {df_ecom.duplicated().sum()} rows")
    print(f"    [+] Saved: {ecom_path}")
    
    # Medical dataset
    print("\n[2] Medical Dataset")
    df_med = create_medical_dataset()
    med_path = samples_dir / 'medical_records.csv'
    df_med.to_csv(med_path, index=False)
    print(f"    Shape: {df_med.shape}")
    print(f"    Missing: {df_med.isnull().sum().sum()} cells")
    print(f"    Duplicates: {df_med.duplicated().sum()} rows")
    print(f"    [+] Saved: {med_path}")
    
    print(f"\n[+] Sample datasets created in: {samples_dir}")
    return ecom_path, med_path

if __name__ == '__main__':
    create_sample_datasets()
