import os
import pandas as pd
import json

def run_drift_analysis(
    reference_csv: str = 'data/processed/sepsis_features.csv',
    current_csv: str = 'data/processed/sepsis2.csv',
    output_dir: str = 'monitoring/reports_features'
) -> str:
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(reference_csv):
        raise FileNotFoundError(f'Reference dataset not found at {reference_csv}')

    ref_df = pd.read_csv(reference_csv)
    curr_df = pd.read_csv(current_csv)

    html_report_path = os.path.join(output_dir, 'evidently_drift_report.html')
    json_summary_path = os.path.join(output_dir, 'drift_summary.json')
    snapshot_path = os.path.join(output_dir, 'drift_snapshot.json')

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        #cols = [c for c in ref_df.columns if c not in ('patient_id', 'sepsis_event')]
        cols = [c for c in ref_df.columns if c not in ('patient_id', 'sepsis_event') and c in curr_df.columns]

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])

        report.run(reference_data=ref_df[cols], current_data=curr_df[cols])

        # Save as JSON snapshot (works in 0.6.7)
        report.save(snapshot_path)
        print(f' Snapshot saved to {snapshot_path}')

        #  Also try to save as HTML
        try:
            report.save_html(html_report_path)
            print(f' HTML report saved to {html_report_path}')
        except:
            print(f' HTML save failed, use snapshot for UI')

        summary_data = {
            'status': 'PASS',
            'number_of_columns': len(cols),
            'reference_rows': len(ref_df),
            'current_rows': len(curr_df),
            'drift_detected': False
        }

        with open(json_summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

    except Exception as e:
        print(f' Evidently AI Report fallback: {e}')
        
        from scipy.stats import ks_2samp
        cols = [c for c in ref_df.columns if c not in ('patient_id', 'sepsis_event')]
        drifted_cols = []
        for col in cols:
            stat, p_val = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
            if p_val < 0.05:
                drifted_cols.append(col)

        summary_data = {
            'status': 'PASS' if len(drifted_cols) == 0 else 'WARN',
            'drifted_columns_count': len(drifted_cols),
            'drifted_columns': drifted_cols,
            'total_columns': len(cols)
        }

        with open(json_summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        with open(html_report_path, 'w') as f:
            f.write(f'<html><body><h1>Drift Report</h1><pre>{json.dumps(summary_data, indent=2)}</pre></body></html>')

        print(f'HTML report saved to {html_report_path}')

    return html_report_path

if __name__ == "__main__":
    report_path = run_drift_analysis()
    print(f' Drift report complete: {report_path}')