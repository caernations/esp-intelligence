import os
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ESP Intelligence", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

theme_colors = {
    "primary": "#1E88E5",   
    "secondary": "#FF5722", 
    "success": "#4CAF50",  
    "warning": "#FFC107",   
    "danger": "#E53935",   
    "neutral": "#757575"    
}

@st.cache_data
def load_data():
    wells = pd.read_csv("data/wells_cleaned.csv")
    installations = pd.read_csv("data/esp_well_installations_cleaned.csv")
    failure_cause = pd.read_csv("data/esp_failure_cause_cleaned.csv")
    
    date_columns = ['start_date', 'dhp_date', 'pulling_date']
    for col in date_columns:
        if col in installations.columns:
            installations[col] = pd.to_datetime(installations[col], errors='coerce')
    
    date_columns = ['installation_date', 'dhp_date', 'pulling_date', 'difa_date']
    for col in date_columns:
        if col in failure_cause.columns:
            failure_cause[col] = pd.to_datetime(failure_cause[col], errors='coerce')
    
    merged_data = pd.merge(
        installations, 
        failure_cause[['well', 'failure_item', 'failure_item_specific', 'general_failure_cause', 
                      'specific_failure_cause', 'recommendation', 'dhp_date']], 
        on=['well', 'dhp_date'], 
        how='left', 
        suffixes=('', '_failure')
    )
    
    return wells, installations, failure_cause, merged_data

wells_df, installations_df, failure_cause_df, merged_df = load_data()

def interpret_query(query):
    q = query.lower()
    categories = {
        "vendor_performance": ["vendor", "pemasok", "penyedia", "supplier", "produsen", "buat", "pabrikan", 
                              "manufacturer", "maker", "vendor terbaik", "vendor mana", "performa vendor", 
                              "reliability vendor", "keandalan vendor", "vendor reliability"],
        
        "failure_analysis": ["failure", "gagal", "kerusakan", "rusak", "mode failure", "penyebab gagal", 
                            "kegagalan", "failure mode", "macet", "dana", "loss", "root cause", 
                            "akar masalah", "penyebab utama", "failure frequency", "frekuensi kegagalan"],
        
        "area_performance": ["area", "wilayah", "cluster", "zona", "daerah", "lokasi", "tempat", 
                            "performa area", "area terbaik", "worst area", "area terburuk"],
        
        "pump_analysis": ["pump", "pompa", "tipe pompa", "pump type", "jenis pompa", "model pompa", 
                         "performa pompa", "pump performance", "pump efficiency", "efisiensi pompa"],
        
        "runlife_analysis": ["run life", "umur pakai", "run", "durasi", "waktu operasional", 
                            "waktu berjalan", "lifetime", "masa pakai", "trend run life", 
                            "historical run life", "performa jangka panjang"],
        
        "component_analysis": ["failure item", "komponen gagal", "bagian rusak", "komponen", "item gagal", 
                              "part", "komponen kritis", "critical component", "komponen paling sering rusak"],
        
        "time_analysis": ["trend", "waktu", "time", "period", "periode", "musiman", "seasonal", 
                         "tahun", "bulan", "year", "month", "historical", "history", "historis", 
                         "time series", "tren waktu"],
        
        "cost_analysis": ["cost", "biaya", "pengeluaran", "expenditure", "expense", "cost analysis", 
                         "analisis biaya", "cost per failure", "cost comparison", "perbandingan biaya", 
                         "budget", "anggaran", "cost saving", "penghematan"],
        
        "technical_specs": ["horsepower", "hp", "daya", "voltase", "volt", "ampere", "amp", "stage", 
                           "stg", "specs", "spesifikasi", "technical", "teknis", "rpm", "capacity", 
                           "kapasitas", "psd", "pump setting depth", "kedalaman pompa"],
        
        "maintenance_analysis": ["maintenance", "perawatan", "mtbf", "mean time between failure", 
                               "preventive", "pencegahan", "scheduled", "terjadwal", "repair", 
                               "perbaikan", "workover", "intervention", "intervensi"],
        
        "correlation_analysis": ["correlation", "korelasi", "hubungan", "relationship", "faktor", 
                                "factors", "affecting", "mempengaruhi", "impact", "dampak", 
                                "pengaruh", "variables", "variabel"]
    }
    
    for category, keywords in categories.items():
        if any(word in q for word in keywords):
            return category
    
    if any(word in q for word in ["compare", "comparison", "bandingkan", "perbandingan", "versus", "vs"]):
        if "vendor" in q or "manufacturer" in q:
            return "vendor_comparison"
        elif "area" in q:
            return "area_comparison"
        elif "pump" in q or "pompa" in q:
            return "pump_comparison"
        else:
            return "general_comparison"
    
    return "wells_info"

def filter_data(keyword, query=None):
    if keyword == "vendor_performance":
        df = installations_df.groupby("vendor").agg({
            'run': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'trl': ['mean', 'count']
        }).reset_index()
        df.columns = ['vendor', 'run_mean', 'run_median', 'run_std', 'run_min', 'run_max', 'installation_count', 'trl_mean', 'trl_count']
        df['run_to_trl_ratio'] = df['run_mean'] / df['trl_mean']
        df['success_rate'] = df['run_mean'] / 365 
        
        vendor_failures = failure_cause_df.groupby('manufacture').size().reset_index(name='failure_count')
        df = pd.merge(df, vendor_failures, left_on='vendor', right_on='manufacture', how='left')
        df['failure_count'] = df['failure_count'].fillna(0)
        df['failure_rate'] = df['failure_count'] / df['installation_count']
        
        return df, "vendor", "run_mean", "Vendor Performance Analysis"
    
    elif keyword == "failure_analysis":
        base_df = failure_cause_df.groupby("failure_mode").size().reset_index(name='count')
        avg_run = failure_cause_df.groupby("failure_mode")['run'].mean().reset_index()
        base_df = pd.merge(base_df, avg_run, on="failure_mode", how="left")
        total_failures = base_df['count'].sum()
        base_df['percentage'] = (base_df['count'] / total_failures * 100).round(2)
        item_counts = failure_cause_df.groupby(['failure_mode', 'failure_item']).size().reset_index(name='item_count')
        return base_df, "failure_mode", "count", "Failure Mode Analysis", item_counts
    
    elif keyword == "area_performance":
        df = installations_df.groupby("area").agg({
            'run': ['mean', 'median', 'count', 'std'],
            'well': pd.Series.nunique
        }).reset_index()
        df.columns = ['area', 'run_mean', 'run_median', 'installation_count', 'run_std', 'unique_wells']
        df['failure_rate'] = df['installation_count'] / df['unique_wells']
        area_failures = failure_cause_df.groupby('area').size().reset_index(name='failure_count')
        df = pd.merge(df, area_failures, on='area', how='left')
        df['failure_count'] = df['failure_count'].fillna(0)
        
        return df, "area", "run_mean", "Area Performance Analysis"
    
    elif keyword == "pump_analysis":
        df = installations_df.groupby("pump_type").agg({
            'run': ['mean', 'median', 'std', 'count'],
            'well': pd.Series.nunique
        }).reset_index()
        df.columns = ['pump_type', 'run_mean', 'run_median', 'run_std', 'count', 'unique_wells']
        df['installations_per_well'] = df['count'] / df['unique_wells']
        pump_vendors = installations_df.groupby(['pump_type', 'vendor']).size().reset_index(name='vendor_count')
        
        return df, "pump_type", "run_mean", "Pump Type Analysis", pump_vendors
    
    elif keyword == "runlife_analysis":
        df = installations_df.copy()
        bin_edges = [0, 30, 90, 180, 365, 547, 730, 1095, float('inf')]
        bin_labels = ['0-30 days', '31-90 days', '91-180 days', '6-12 months', '12-18 months', '18-24 months', '24-36 months', '36+ months']
        df['run_life_category'] = pd.cut(df['run'], bins=bin_edges, labels=bin_labels)
        run_dist = df['run_life_category'].value_counts().reset_index()
        run_dist.columns = ['run_life_category', 'count']
        
        if 'start_date' in df.columns:
            df['year'] = df['start_date'].dt.year
            yearly_runlife = df.groupby('year')['run'].mean().reset_index()
            yearly_runlife = yearly_runlife.sort_values('year')
        else:
            yearly_runlife = None
        
        return run_dist, "run_life_category", "count", "Run Life Distribution Analysis", yearly_runlife
    
    elif keyword == "component_analysis":
        base_df = failure_cause_df.groupby("failure_item").size().reset_index(name='count')
        component_run = failure_cause_df.groupby("failure_item")['run'].agg(['mean', 'median']).reset_index()
        base_df = pd.merge(base_df, component_run, on="failure_item", how="left")
        specific_components = failure_cause_df.groupby(['failure_item', 'failure_item_specific']).size().reset_index(name='specific_count')
        
        return base_df, "failure_item", "count", "Component Failure Analysis", specific_components
    
    elif keyword == "time_analysis":
        df = installations_df.copy()
        if 'start_date' in df.columns:
            df['year'] = df['start_date'].dt.year
            df['month'] = df['start_date'].dt.month
            df['quarter'] = df['start_date'].dt.quarter
            yearly_stats = df.groupby('year').agg({
                'run': ['mean', 'count'],
                'well': pd.Series.nunique
            }).reset_index()
            yearly_stats.columns = ['year', 'avg_run', 'installation_count', 'unique_wells']
            monthly_stats = df.groupby(['year', 'month']).size().reset_index(name='count')
            
            return yearly_stats, "year", "avg_run", "Time Trend Analysis", monthly_stats
        else:
            return df.head(10), None, None, "Date information not available", None
    
    elif keyword == "technical_specs":
        tech_cols = ['hp', 'volt', 'amp', 'stg', 'psd']
        available_cols = [col for col in tech_cols if col in installations_df.columns]
        
        if available_cols:
            corr_data = {}
            for col in available_cols:
                temp_df = installations_df[[col, 'run']].dropna()
                if not temp_df.empty:
                    corr = temp_df.corr().iloc[0, 1]
                    corr_data[col] = corr
            corr_df = pd.DataFrame(list(corr_data.items()), columns=['spec', 'correlation_with_runlife'])
            tech_dist = {}
            for col in available_cols:
                if installations_df[col].dtype in [np.float64, np.int64]:
                    tech_dist[col] = installations_df[col].value_counts().head(10).reset_index()
                    tech_dist[col].columns = [col, 'count']
            
            return corr_df, "spec", "correlation_with_runlife", "Technical Specs Analysis", tech_dist
        else:
            return installations_df.head(10), None, None, "Technical specification columns not available", None
    
    elif keyword == "correlation_analysis":
        num_cols = installations_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'run' in num_cols:
            num_cols.remove('run') 
            
            corr_data = []
            for col in num_cols:
                temp_df = installations_df[[col, 'run']].dropna()
                if not temp_df.empty and temp_df[col].nunique() > 5:  
                    corr = temp_df.corr().iloc[0, 1]
                    corr_data.append({'factor': col, 'correlation': corr})
            
            corr_df = pd.DataFrame(corr_data)
            corr_df = corr_df.sort_values('correlation', ascending=False)
            
            top_factors = corr_df.head(3)['factor'].tolist()
            scatter_data = {}
            for factor in top_factors:
                scatter_data[factor] = installations_df[[factor, 'run']].dropna().sample(min(500, installations_df.shape[0]))
            
            return corr_df, "factor", "correlation", "Correlation Analysis", scatter_data
        else:
            return installations_df.head(10), None, None, "Run life data not available for correlation", None
    
    elif keyword == "vendor_comparison" or keyword == "area_comparison" or keyword == "pump_comparison" or keyword == "general_comparison":
        entities = []
        comparison_col = ""
        
        if keyword == "vendor_comparison":
            comparison_col = "vendor"
            common_vendors = installations_df['vendor'].value_counts().head(5).index.tolist()
            for vendor in common_vendors:
                if vendor.lower() in query.lower():
                    entities.append(vendor)
        
        elif keyword == "area_comparison":
            comparison_col = "area"
            common_areas = installations_df['area'].value_counts().head(10).index.tolist()
            for area in common_areas:
                if area.lower() in query.lower():
                    entities.append(area)
        
        elif keyword == "pump_comparison":
            comparison_col = "pump_type"
            common_pumps = installations_df['pump_type'].value_counts().head(10).index.tolist()
            for pump in common_pumps:
                if pump.lower() in query.lower():
                    entities.append(pump)
        
        if not entities and comparison_col:
            entities = installations_df.groupby(comparison_col)['run'].mean().nlargest(3).index.tolist()
        
        if comparison_col:
            df = installations_df[installations_df[comparison_col].isin(entities)]
            comparison_metrics = df.groupby(comparison_col).agg({
                'run': ['mean', 'median', 'std', 'min', 'max', 'count'],
                'trl': ['mean', 'count'] if 'trl' in df.columns else ['count'],
                'well': pd.Series.nunique
            }).reset_index()
            
            comparison_metrics.columns = [f"{'' if col[0] == comparison_col else col[0]}{'_' + col[1] if col[1] != '' else ''}" for col in comparison_metrics.columns]
            return comparison_metrics, comparison_col, "run_mean", f"{comparison_col.title()} Comparison"
        
        else:
            return installations_df.head(10), None, None, "Comparison data not available"
    
    elif keyword == "maintenance_analysis":
        if 'dhp_date' in installations_df.columns and 'pulling_date' in installations_df.columns:
            df = installations_df.copy()
            df['intervention_delay'] = (df['pulling_date'] - df['dhp_date']).dt.days
            delay_by_failure = df.groupby('failure_mode')['intervention_delay'].mean().reset_index()
            delay_by_area = df.groupby('area')['intervention_delay'].mean().reset_index()
            
            if 'start_date' in df.columns:
                failures_by_well = df.sort_values(['well', 'start_date'])
                failures_by_well['next_start'] = failures_by_well.groupby('well')['start_date'].shift(-1)
                failures_by_well['days_to_next_failure'] = (failures_by_well['next_start'] - failures_by_well['start_date']).dt.days
                mtbf_by_well = failures_by_well.groupby('well')['days_to_next_failure'].mean().reset_index()
                mtbf_by_well = mtbf_by_well.rename(columns={'days_to_next_failure': 'mtbf_days'})
                mtbf_by_area = failures_by_well.groupby('area')['days_to_next_failure'].mean().reset_index()
                mtbf_by_area = mtbf_by_area.rename(columns={'days_to_next_failure': 'mtbf_days'})
                
                return mtbf_by_area, "area", "mtbf_days", "Maintenance Analysis", {'delay_by_failure': delay_by_failure, 
                                                                                  'mtbf_by_well': mtbf_by_well.sort_values('mtbf_days', ascending=False).head(10)}
            else:
                return delay_by_failure, "failure_mode", "intervention_delay", "Maintenance Analysis", {'delay_by_area': delay_by_area}
        else:
            return installations_df.head(10), None, None, "Maintenance time data not available", None
    
    elif keyword == "cost_analysis":
        df = failure_cause_df.copy()
        
        cost_by_failure_mode = df.groupby('failure_mode').size().reset_index(name='failure_count')
        cost_by_failure_mode['est_relative_cost'] = cost_by_failure_mode['failure_count'] * 1000  # Placeholder
        cost_by_vendor = df.groupby('manufacture').size().reset_index(name='failure_count')
        cost_by_vendor['est_relative_cost'] = cost_by_vendor['failure_count'] * 1000  # Placeholder
        cost_by_item = df.groupby('failure_item').size().reset_index(name='failure_count') 
        cost_by_item['est_relative_cost'] = cost_by_item['failure_count'] * 1000  # Placeholder
        
        return cost_by_failure_mode, "failure_mode", "est_relative_cost", "Cost Impact Analysis (Estimated)",  {'cost_by_vendor': cost_by_vendor, 'cost_by_item': cost_by_item}
    
    else: 
        df = wells_df.head(50)
        return df, None, None, "Wells Information", None

def plot_data(df, plot_type, x_col, y_col, title=None, additional_data=None):
    try:
        if plot_type == "bar":
            fig = px.bar(
                df, x=x_col, y=y_col, 
                title=title or f"{y_col.title()} per {x_col.title()}",
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                color_discrete_sequence=[theme_colors["primary"]]
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=50, b=100),
                height=500
            )
            
        elif plot_type == "line":
            fig = px.line(
                df, x=x_col, y=y_col, 
                title=title or f"{y_col.title()} per {x_col.title()} (Trend)",
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                markers=True,
                color_discrete_sequence=[theme_colors["primary"]]
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=50, b=100),
                height=500
            )
            
        elif plot_type == "pie":
            fig = px.pie(
                df, names=x_col, values=y_col,
                title=title or f"Distribution of {y_col.title()} by {x_col.title()}",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                height=500
            )
            
        elif plot_type == "scatter":
            if additional_data and isinstance(additional_data, dict) and len(additional_data) > 0:
                first_key = list(additional_data.keys())[0]
                scatter_df = additional_data[first_key]
                fig = px.scatter(
                    scatter_df, x=first_key, y='run',
                    title=f"Correlation between {first_key.replace('_', ' ').title()} and Run Life",
                    labels={first_key: first_key.replace('_', ' ').title(), 'run': 'Run Life (days)'},
                    trendline="ols",
                    color_discrete_sequence=[theme_colors["primary"]]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col,
                    title=title or f"Relationship between {x_col.title()} and {y_col.title()}",
                    labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                    color_discrete_sequence=[theme_colors["primary"]]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
            
        elif plot_type == "heatmap":
            if df.shape[0] > 1 and df.shape[1] > 1:
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] > 1:
                    corr_matrix = numeric_df.corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap",
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        height=600,
                        margin=dict(l=20, r=20, t=50, b=50)
                    )
                else:
                    return None
            else:
                return None
                
        elif plot_type == "histogram":
            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                fig = px.histogram(
                    df, x=y_col,
                    title=f"Distribution of {y_col.replace('_', ' ').title()}",
                    labels={y_col: y_col.replace('_', ' ').title()},
                    color_discrete_sequence=[theme_colors["primary"]]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
            else:
                return None
                
        elif plot_type == "box":
            if x_col and y_col and y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                fig = px.box(
                    df, x=x_col, y=y_col,
                    title=f"Distribution of {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                    labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                    color_discrete_sequence=[theme_colors["primary"]]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=50)
                )
            else:
                return None
                
        elif plot_type == "multi_bar":
            if df.shape[1] >= 4: 
                metrics = [col for col in df.columns if col != x_col and df[col].dtype in [np.float64, np.int64]][:3]  
                
                fig = go.Figure()
                for metric in metrics:
                    fig.add_trace(go.Bar(
                        x=df[x_col],
                        y=df[metric],
                        name=metric.replace('_', ' ').title()
                    ))
                
                fig.update_layout(
                    title=title or f"Multiple Metrics by {x_col.title()}",
                    xaxis_title=x_col.replace('_', ' ').title(),
                    yaxis_title="Value",
                    legend_title="Metrics",
                    height=500,
                    barmode='group',
                    margin=dict(l=20, r=20, t=50, b=100)
                )
            else:
                return None
        else:
            return None
            
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def generate_secondary_charts(keyword, df, x_col, y_col, additional_data):
    charts = []
    
    if keyword == "vendor_performance":
        if "run_to_trl_ratio" in df.columns and "failure_rate" in df.columns:
            metrics_fig = px.scatter(
                df, x="failure_rate", y="run_to_trl_ratio", color="vendor",
                size="installation_count", hover_name="vendor",
                labels={"failure_rate": "Failure Rate", "run_to_trl_ratio": "Run/Target Ratio", "installation_count": "Number of Installations"},
                title="Vendor Performance Matrix"
            )
            charts.append(("Vendor Performance Matrix", metrics_fig))
        
        if "vendor" in df.columns and "run" in df.columns:
            run_box_fig = px.box(
                installations_df[installations_df['vendor'].isin(df['vendor'])], 
                x="vendor", y="run",
                title="Distribution of Run Life by Vendor",
                labels={"vendor": "Vendor", "run": "Run Life (days)"}
            )
            charts.append(("Run Life Distribution by Vendor", run_box_fig))
    
    elif keyword == "failure_analysis":
        if 'dhp_date' in failure_cause_df.columns:
            failure_cause_df['failure_year'] = pd.to_datetime(failure_cause_df['dhp_date']).dt.year
            time_trend = failure_cause_df.groupby(['failure_year', 'failure_mode']).size().reset_index(name='count')
            
            time_fig = px.line(
                time_trend, x="failure_year", y="count", color="failure_mode",
                title="Failure Trend Over Time by Failure Mode",
                labels={"failure_year": "Year", "count": "Number of Failures", "failure_mode": "Failure Mode"}
            )
            charts.append(("Failure Trend Over Time", time_fig))
        
        if additional_data is not None:
            comp_fig = px.bar(
                additional_data, x="failure_mode", y="item_count", color="failure_item",
                title="Component Breakdown by Failure Mode",
                labels={"failure_mode": "Failure Mode", "item_count": "Count", "failure_item": "Failed Component"}
            )
            charts.append(("Component Breakdown by Failure Mode", comp_fig))
    
    elif keyword == "area_performance":
        if "failure_count" in df.columns and "run_mean" in df.columns:
            area_matrix_fig = px.scatter(
                df, x="failure_count", y="run_mean", size="unique_wells", 
                color="installation_count", hover_name="area",
                title="Area Performance Matrix",
                labels={"failure_count": "Number of Failures", "run_mean": "Average Run Life (days)", 
                       "unique_wells": "Number of Wells"}
            )
            charts.append(("Area Performance Matrix", area_matrix_fig))
        
        run_box_fig = px.box(
            installations_df[installations_df['area'].isin(df['area'])], 
            x="area", y="run",
            title="Distribution of Run Life by Area",
            labels={"area": "Area", "run": "Run Life (days)"}
        )
        charts.append(("Run Life Distribution by Area", run_box_fig))
    
    elif keyword == "pump_analysis":
        if additional_data is not None:
            vendor_fig = px.bar(
                additional_data, x="pump_type", y="vendor_count", color="vendor",
                title="Vendor Usage by Pump Type",
                labels={"pump_type": "Pump Type", "vendor_count": "Count", "vendor": "Vendor"}
            )
            charts.append(("Vendor Usage by Pump Type", vendor_fig))
        
        run_box_fig = px.box(
            installations_df[installations_df['pump_type'].isin(df['pump_type'])], 
            x="pump_type", y="run",
            title="Distribution of Run Life by Pump Type",
            labels={"pump_type": "Pump Type", "run": "Run Life (days)"}
        )
        charts.append(("Run Life Distribution by Pump Type", run_box_fig))
    
    elif keyword == "runlife_analysis":
        if additional_data is not None and additional_data is not False:
            trend_fig = px.line(
                additional_data, x="year", y="run",
                title="Average Run Life Trend Over Years",
                labels={"year": "Year", "run": "Average Run Life (days)"},
                markers=True
            )
            trend_fig.update_layout(xaxis_type='category')  
            charts.append(("Run Life Trend Over Time", trend_fig))
        
        hist_fig = px.histogram(
            installations_df, x="run",
            title="Run Life Distribution",
            labels={"run": "Run Life (days)"},
            nbins=30
        )
        charts.append(("Run Life Distribution", hist_fig))
    
    elif keyword == "component_analysis":
        if additional_data is not None:
            top_components = df.sort_values('count', ascending=False)['failure_item'].head(5).tolist()
            filtered_data = additional_data[additional_data['failure_item'].isin(top_components)]
            comp_detail_fig = px.bar(
                filtered_data, x="failure_item", y="specific_count", color="failure_item_specific",
                title="Detailed Component Failure Breakdown",
                labels={"failure_item": "Main Component", "specific_count": "Count", 
                        "failure_item_specific": "Specific Component"}
            )
            charts.append(("Detailed Component Breakdown", comp_detail_fig))
        
        comp_run_fig = px.bar(
            df, x="failure_item", y="mean",
            title="Average Run Life by Failed Component",
            labels={"failure_item": "Failed Component", "mean": "Average Run Life (days)"}
        )
        charts.append(("Run Life by Component", comp_run_fig))
    
    elif keyword == "time_analysis":
        if additional_data is not None:
            monthly_fig = px.line(
                additional_data.pivot_table(index='month', columns='year', values='count', aggfunc='sum').reset_index(),
                x="month", y=additional_data['year'].unique().tolist(),
                title="Monthly Installation Patterns by Year",
                labels={"month": "Month", "value": "Number of Installations", "variable": "Year"}
            )
            monthly_fig.update_layout(xaxis_type='category')  
            charts.append(("Monthly Installation Patterns", monthly_fig))
        
        if 'start_date' in installations_df.columns:
            installations_sorted = installations_df.sort_values('start_date')
            installations_sorted['cumulative_count'] = range(1, len(installations_sorted) + 1)
            
            cumulative_fig = px.line(
                installations_sorted, x="start_date", y="cumulative_count",
                title="Cumulative ESP Installations Over Time",
                labels={"start_date": "Date", "cumulative_count": "Total Installations"}
            )
            charts.append(("Cumulative Installations", cumulative_fig))
    
    elif keyword == "correlation_analysis":
        num_cols = installations_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            corr_matrix = installations_df[num_cols].corr()
            
            heatmap_fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix of ESP Parameters",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                zmin=-1, zmax=1
            )
            charts.append(("Correlation Matrix", heatmap_fig))
        
        if additional_data is not None and isinstance(additional_data, dict):
            for factor, scatter_df in additional_data.items():
                scatter_fig = px.scatter(
                    scatter_df, x=factor, y="run",
                    title=f"Correlation between {factor.replace('_', ' ').title()} and Run Life",
                    labels={factor: factor.replace('_', ' ').title(), "run": "Run Life (days)"},
                    trendline="ols"
                )
                charts.append((f"{factor.replace('_', ' ').title()} vs Run Life", scatter_fig))
    
    return charts

def build_ui():
    st.title("ESP Intelligence Dashboard")
    
    main_tabs = st.tabs(["Data Analysis"])
    
    with main_tabs[0]:
        st.header("ESP Data Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_method = st.radio("Method:", ["Natural Language Query", "Guided Analysis"], horizontal=True)
            
            if search_method == "Natural Language Query":
                user_query = st.text_input("Ask Anything about ESP:", placeholder="Example: Vendor mana dengan performa terbaik?")
                st.info("Tips: Pertanyaan spesifik akan memberikan hasil yang lebih baik.")
            else:
                analysis_type = st.selectbox("Tipe Analisis:", [
                    "Vendor Performance", "Failure Analysis", "Area Performance", 
                    "Pump Analysis", "Run Life Analysis", "Component Analysis",
                    "Time Trend Analysis", "Correlation Analysis", "Technical Specifications"
                ])
                
                if analysis_type == "Vendor Performance":
                    selected_vendors = st.multiselect("Pilih Vendor:", installations_df['vendor'].unique())
                    metric = st.selectbox("Metrik:", ["Run Life", "Failure Rate", "Installation Count"])
                    user_query = f"Analisis performa vendor untuk {', '.join(selected_vendors) if selected_vendors else 'semua vendor'} berdasarkan {metric}"
                
                elif analysis_type == "Failure Analysis":
                    selected_modes = st.multiselect("Pilih Failure Mode:", failure_cause_df['failure_mode'].unique())
                    time_period = st.selectbox("Periode Waktu:", ["Semua Waktu", "Tahun Terakhir", "3 Tahun Terakhir", "5 Tahun Terakhir"])
                    user_query = f"Analisis kegagalan untuk {', '.join(selected_modes) if selected_modes else 'semua mode'} selama {time_period}"
                
                else:
                    user_query = f"Analisis {analysis_type.lower()}"
        
        with col2:
            show_raw_data = st.checkbox("Display Raw Data", value=True)
            enable_ai_analysis = st.checkbox("AI Analysis", value=True)
            
            st.divider()
            
            default_chart = st.selectbox("Default Graph:", ["bar", "line", "pie", "scatter", "box", "histogram", "heatmap", "multi_bar"])
            color_theme = st.selectbox("Color Scheme:", ["Biru", "Hijau", "Oranye", "Ungu"])
            
            if color_theme == "Hijau":
                theme_colors["primary"] = "#4CAF50"
            elif color_theme == "Oranye":
                theme_colors["primary"] = "#FF5722"
            elif color_theme == "Ungu":
                theme_colors["primary"] = "#9C27B0"
        
        if 'user_query' in locals() and user_query:
            st.info(f"Menganalisis: **{user_query}**")
            
            with st.spinner("Memproses data dan menyiapkan visualisasi..."):
                keyword = interpret_query(user_query)
                result = filter_data(keyword, user_query)
                
                if len(result) == 5:
                    df, x_col, y_col, title, additional_data = result
                else:
                    df, x_col, y_col, title = result
                    additional_data = None
                
                results_tabs = st.tabs(["Visualisi", "Insights", "Data Explorer", "AI Analysis"])
                
                with results_tabs[0]:
                    st.subheader(title)
                    
                    if x_col and y_col:
                        chart_col1, chart_col2 = st.columns([3, 1])
                        
                        with chart_col2:
                            chart_type = st.selectbox("Tipe Grafik:", ["bar", "line", "pie", "scatter", "box", "histogram", "heatmap", "multi_bar"], 
                                                     index=["bar", "line", "pie", "scatter", "box", "histogram", "heatmap", "multi_bar"].index(default_chart))
                            
                            if chart_type == "bar" or chart_type == "line":
                                sort_by = st.selectbox("Urutkan Berdasarkan:", ["Original", "Naik", "Turun"])
                                if sort_by == "Naik":
                                    df = df.sort_values(y_col)
                                elif sort_by == "Turun":
                                    df = df.sort_values(y_col, ascending=False)
                            
                            st.divider()
                            
                            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                                st.metric("Rata-rata", f"{df[y_col].mean():.2f}")
                                st.metric("Median", f"{df[y_col].median():.2f}")
                                st.metric("Minimum", f"{df[y_col].min():.2f}")
                                st.metric("Maksimum", f"{df[y_col].max():.2f}")
                        
                        with chart_col1:
                            fig = plot_data(df, chart_type, x_col, y_col, title, additional_data)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Tidak dapat membuat visualisasi dengan opsi yang dipilih.")
                    else:
                        st.write(df)
                
                with results_tabs[1]:
                    st.subheader("Insight Tambahan")
                    
                    if x_col and y_col:
                        charts = generate_secondary_charts(keyword, df, x_col, y_col, additional_data)
                        
                        if charts:
                            for i, (chart_title, chart_fig) in enumerate(charts):
                                if i % 2 == 0:  
                                    cols = st.columns(2)
                                
                                with cols[i % 2]:
                                    st.subheader(chart_title)
                                    st.plotly_chart(chart_fig, use_container_width=True)
                        else:
                            st.write("Tidak ada insight tambahan untuk query ini.")
                    else:
                        st.write("Tidak ada insight tambahan untuk query ini.")
                
                with results_tabs[2]:
                    st.subheader("Data Explorer")
                    explore_tabs = st.tabs(["📋 Tabel Data", "🧮 Statistik Ringkasan", "🔍 Filter & Cari"])
                    
                    with explore_tabs[0]:
                        if df is not None and show_raw_data:
                            rows_per_page = st.slider("Baris per halaman", 5, 100, 20)
                            page = st.number_input("Halaman", min_value=1, max_value=max(1, len(df) // rows_per_page + 1), step=1)
                            
                            start_idx = (page - 1) * rows_per_page
                            end_idx = min(start_idx + rows_per_page, len(df))
                            
                            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
                            st.write(f"Menampilkan {start_idx+1} hingga {end_idx} dari {len(df)} entri")
                
                    with explore_tabs[1]:
                        if df is not None:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                st.write("Statistik Ringkasan:")
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                            
                            st.write("Informasi Kolom:")
                            col_info = pd.DataFrame({
                                'Kolom': df.columns,
                                'Tipe': [str(dtype) for dtype in df.dtypes],
                                'Jumlah Non-Null': df.count(),
                                'Jumlah Null': df.isna().sum(),
                                'Nilai Unik': [df[col].nunique() for col in df.columns]
                            })
                            st.dataframe(col_info, use_container_width=True)
                    
                    with explore_tabs[2]:
                        if df is not None:
                            st.write("Filter Data:")
                            
                            filter_col1, filter_col2 = st.columns(2)
                            
                            with filter_col1:
                                filter_column = st.selectbox("Pilih Kolom:", df.columns)
                            
                            with filter_col2:
                                if df[filter_column].dtype in [np.float64, np.int64]:
                                    min_val = float(df[filter_column].min())
                                    max_val = float(df[filter_column].max())
                                    filter_range = st.slider("Rentang Nilai:", min_val, max_val, (min_val, max_val))
                                    filtered_df = df[(df[filter_column] >= filter_range[0]) & (df[filter_column] <= filter_range[1])]
                                else:
                                    filter_values = st.multiselect("Pilih Nilai:", df[filter_column].unique())
                                    if filter_values:
                                        filtered_df = df[df[filter_column].isin(filter_values)]
                                    else:
                                        filtered_df = df
                            
                            st.dataframe(filtered_df, use_container_width=True)
                
                with results_tabs[3]:
                    st.subheader("AI Analysis")
                    
                    if enable_ai_analysis:
                        context = f"Tipe Analisis: {keyword}\n"
                        context += f"Ringkasan data utama:\n{df.describe().to_string()}\n\n"
                        if keyword == "vendor_performance":
                            context += "Vendor dengan kinerja terbaik berdasarkan run life:\n"
                            context += df.sort_values('run_mean', ascending=False).head(3).to_string(index=False)
                            context += "\n\nVendor dengan jumlah instalasi terbanyak:\n"
                            context += df.sort_values('installation_count', ascending=False).head(3).to_string(index=False)
                        
                        elif keyword == "failure_analysis":
                            context += "Distribusi mode kegagalan:\n"
                            context += df.to_string(index=False)
                            if additional_data is not None:
                                context += "\n\nBreakdown komponen:\n"
                                context += additional_data.head(10).to_string(index=False)
                        
                        prompt = f"""Data konteks untuk analisis ESP:
                        {context}
                        
                        Pertanyaan pengguna: {user_query}
                        
                        Berikan:
                        1. Ringkasan singkat temuan utama (3-5 poin)
                        2. Insight yang dapat ditindaklanjuti berdasarkan data
                        3. Rekomendasi untuk perbaikan operasional
                        4. Analisis lanjutan yang disarankan
                        
                        Format respons Anda dalam markdown dan fokus pada nilai bisnis. PENTING: Berikan respons dalam Bahasa Indonesia."""
                        
                        with st.spinner("Menghasilkan analisis AI..."):
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Anda adalah asisten analisis ESP minyak & gas yang canggih. Berikan insight singkat dan berbasis data dengan fokus pada nilai bisnis dan perbaikan operasional. Selalu berikan respons dalam Bahasa Indonesia."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=800,
                                    temperature=0.3,
                                )
                                answer = response.choices[0].message.content
                                st.markdown(answer)
                            except Exception as e:
                                st.error(f"Error menghasilkan analisis AI: {str(e)}")
                                st.markdown("""
                                ## Analisis Dasar
                                
                                Berdasarkan data, berikut beberapa observasi umum:
                                
                                * Data menunjukkan pola yang memerlukan investigasi lebih lanjut
                                * Pertimbangkan untuk memeriksa trend dari waktu ke waktu untuk insight yang lebih baik
                                * Cari korelasi antar variabel untuk mengidentifikasi kemungkinan penyebab
                                
                                Aktifkan API OpenAI untuk analisis yang lebih detail.
                                """)
                    else:
                        st.info("Aktifkan Analisis AI untuk mendapatkan insight cerdas berdasarkan query Anda.")
                        
                        if x_col and y_col and df[y_col].dtype in [np.float64, np.int64]:
                            st.write("Ringkasan Statistik Dasar:")
                            
                            summary_col1, summary_col2 = st.columns(2)
                            
                            with summary_col1:
                                st.metric("Rata-rata", f"{df[y_col].mean():.2f}")
                                st.metric("Median", f"{df[y_col].median():.2f}")
                                st.metric("Standar Deviasi", f"{df[y_col].std():.2f}")
                            
                            with summary_col2:
                                st.metric("Minimum", f"{df[y_col].min():.2f}")
                                st.metric("Maksimum", f"{df[y_col].max():.2f}")
                                st.metric("Jumlah", f"{df[y_col].count()}")
    
def main():
    build_ui()

if __name__ == "__main__":
    main()