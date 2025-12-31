import argparse
from pathlib import Path
import sys

def main(csv_path: Path, out_dir: Path):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        print("Missing dependency: make sure pandas, matplotlib, seaborn are installed.")
        raise

    df = pd.read_csv(csv_path, dtype=str)

    # Normalize column names (strip accidental spaces)
    df.columns = [c.strip() for c in df.columns]

    # Convert confianza to numeric
    if 'confianza' in df.columns:
        df['confianza'] = pd.to_numeric(df['confianza'], errors='coerce')
    else:
        df['confianza'] = pd.NA

    # Convert t_total_ms which is in format HH:MM:SS.sss to milliseconds
    if 't_total_ms' in df.columns:
        # some values like "00:00:05.422" parseable by to_timedelta
        df['latency_td'] = pd.to_timedelta(df['t_total_ms'], errors='coerce')
        df['latency_ms'] = df['latency_td'].dt.total_seconds() * 1000
    else:
        df['latency_ms'] = pd.NA

    # Use escenario column if present
    escenario_col = 'escenario' if 'escenario' in df.columns else None
    if escenario_col is None:
        df['escenario'] = 'unknown'
        escenario_col = 'escenario'

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style='whitegrid')

    # 1) Average confianza per escenario (ignore NaNs)
    avg_conf = df.groupby(escenario_col)['confianza'].mean().dropna()
    if not avg_conf.empty:
        plt.figure(figsize=(8, 4))
        avg_conf.sort_index().plot(kind='bar', color='C0')
        plt.xlabel('escenario')
        plt.ylabel('average confianza')
        plt.title('Average confianza per escenario')
        plt.tight_layout()
        p = out_dir / 'avg_confidence_by_escenario.png'
        plt.savefig(p)
        print(f'Saved {p}')
        plt.close()

    # 2) Average latency per escenario (ms)
    avg_lat = df.groupby(escenario_col)['latency_ms'].mean().dropna()
    if not avg_lat.empty:
        plt.figure(figsize=(8, 4))
        avg_lat.sort_index().plot(kind='bar', color='C1')
        plt.xlabel('escenario')
        plt.ylabel('average latency (ms)')
        plt.title('Average latency per escenario (ms)')
        plt.tight_layout()
        p = out_dir / 'avg_latency_by_escenario.png'
        plt.savefig(p)
        print(f'Saved {p}')
        plt.close()

    # 3) Confidence distribution (histogram)
    if df['confianza'].dropna().shape[0] > 0:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['confianza'].dropna(), bins=30, kde=False, color='C2')
        plt.xlabel('confianza')
        plt.title('Confidence distribution')
        plt.tight_layout()
        p = out_dir / 'confidence_distribution.png'
        plt.savefig(p)
        print(f'Saved {p}')
        plt.close()

    # 4) Latency boxplot by escenario (ms)
    if df['latency_ms'].dropna().shape[0] > 0:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=escenario_col, y='latency_ms', data=df)
        plt.xlabel('escenario')
        plt.ylabel('latency (ms)')
        plt.title('Latency distribution by escenario')
        plt.tight_layout()
        p = out_dir / 'latency_boxplot_by_escenario.png'
        plt.savefig(p)
        print(f'Saved {p}')
        plt.close()

    # Print short summary
    print('\nSummary:')
    if not avg_conf.empty:
        print('Average confianza per escenario:')
        print(avg_conf)
    if not avg_lat.empty:
        print('\nAverage latency (ms) per escenario:')
        print(avg_lat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plots from analysis.csv')
    parser.add_argument('--csv', type=Path, default=Path('analysis.csv'), help='Path to analysis.csv')
    parser.add_argument('--out', type=Path, default=Path('plots'), help='Output directory for plot images')
    args = parser.parse_args()
    main(args.csv, args.out)
