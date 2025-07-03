import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class COVID19DataAnalyzer:
    def __init__(self):
        # Power BI inspired color palette
        self.colors = {
            'primary': '#0078D4',
            'secondary': '#106EBE', 
            'accent': '#00BCF2',
            'cases': '#FF6B35',
            'deaths': '#D13438',
            'recovered': '#107C10',
            'vaccinated': '#8764B8',
            'background': '#F8F9FA',
            'text': '#323130',
            'grid': '#E1E1E1'
        }
        
        # Data sources
        self.data_sources = {
            'johns_hopkins': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/',
            'owid': 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
        }
        
        # Set professional styling
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette([self.colors['primary'], self.colors['accent'], self.colors['cases'], self.colors['deaths']])
    
    def load_johns_hopkins_data(self):
        """Load data from Johns Hopkins University"""
        print("📊 Loading Johns Hopkins COVID-19 data...")
        
        try:
            # Load time series data
            confirmed_url = self.data_sources['johns_hopkins'] + 'time_series_covid19_confirmed_global.csv'
            deaths_url = self.data_sources['johns_hopkins'] + 'time_series_covid19_deaths_global.csv'
            recovered_url = self.data_sources['johns_hopkins'] + 'time_series_covid19_recovered_global.csv'
            
            confirmed_df = pd.read_csv(confirmed_url)
            deaths_df = pd.read_csv(deaths_url)
            recovered_df = pd.read_csv(recovered_url)
            
            return self.process_johns_hopkins_data(confirmed_df, deaths_df, recovered_df)
            
        except Exception as e:
            print(f"❌ Error loading Johns Hopkins data: {e}")
            return self.generate_mock_covid_data()
    
    def load_owid_data(self):
        """Load data from Our World in Data"""
        print("📊 Loading Our World in Data COVID-19 dataset...")
        
        try:
            owid_df = pd.read_csv(self.data_sources['owid'])
            return self.process_owid_data(owid_df)
            
        except Exception as e:
            print(f"❌ Error loading OWID data: {e}")
            return self.generate_mock_covid_data()
    
    def process_johns_hopkins_data(self, confirmed_df, deaths_df, recovered_df):
        """Process Johns Hopkins data into usable format"""
        # Melt the dataframes to long format
        confirmed_melted = confirmed_df.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            var_name='Date', value_name='Confirmed'
        )
        
        deaths_melted = deaths_df.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            var_name='Date', value_name='Deaths'
        )
        
        recovered_melted = recovered_df.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            var_name='Date', value_name='Recovered'
        )
        
        # Merge the datasets
        covid_data = confirmed_melted.merge(deaths_melted, on=['Province/State', 'Country/Region', 'Date'])
        covid_data = covid_data.merge(recovered_melted, on=['Province/State', 'Country/Region', 'Date'])
        
        # Clean and process
        covid_data['Date'] = pd.to_datetime(covid_data['Date'])
        covid_data['Country/Region'] = covid_data['Country/Region'].replace({
            'US': 'United States',
            'Korea, South': 'South Korea'
        })
        
        # Group by country and date
        country_data = covid_data.groupby(['Country/Region', 'Date']).agg({
            'Confirmed': 'sum',
            'Deaths': 'sum', 
            'Recovered': 'sum'
        }).reset_index()
        
        # Calculate daily new cases
        country_data = country_data.sort_values(['Country/Region', 'Date'])
        country_data['New_Cases'] = country_data.groupby('Country/Region')['Confirmed'].diff().fillna(0)
        country_data['New_Deaths'] = country_data.groupby('Country/Region')['Deaths'].diff().fillna(0)
        
        # Calculate rates
        country_data['Death_Rate'] = (country_data['Deaths'] / country_data['Confirmed'] * 100).fillna(0)
        country_data['Recovery_Rate'] = (country_data['Recovered'] / country_data['Confirmed'] * 100).fillna(0)
        
        return country_data
    
    def process_owid_data(self, owid_df):
        """Process Our World in Data format"""
        # Select relevant columns
        columns_of_interest = [
            'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 
            'new_deaths', 'total_vaccinations', 'people_vaccinated', 
            'people_fully_vaccinated', 'population'
        ]
        
        available_columns = [col for col in columns_of_interest if col in owid_df.columns]
        covid_data = owid_df[available_columns].copy()
        
        # Rename columns for consistency
        covid_data = covid_data.rename(columns={
            'location': 'Country/Region',
            'date': 'Date',
            'total_cases': 'Confirmed',
            'new_cases': 'New_Cases',
            'total_deaths': 'Deaths',
            'new_deaths': 'New_Deaths',
            'total_vaccinations': 'Total_Vaccinations',
            'people_vaccinated': 'People_Vaccinated',
            'people_fully_vaccinated': 'People_Fully_Vaccinated',
            'population': 'Population'
        })
        
        covid_data['Date'] = pd.to_datetime(covid_data['Date'])
        
        # Calculate rates
        covid_data['Death_Rate'] = (covid_data['Deaths'] / covid_data['Confirmed'] * 100).fillna(0)
        covid_data['Vaccination_Rate'] = (covid_data['People_Vaccinated'] / covid_data['Population'] * 100).fillna(0)
        
        return covid_data
    
    def generate_mock_covid_data(self):
        """Generate realistic mock COVID-19 data for demonstration"""
        print("🔄 Generating mock COVID-19 data for demonstration...")
        
        countries = ['United States', 'India', 'Brazil', 'Russia', 'France', 'Turkey', 
                    'Iran', 'Germany', 'United Kingdom', 'Italy', 'Argentina', 'Ukraine']
        
        date_range = pd.date_range(start='2020-01-22', end='2023-12-31', freq='D')
        
        data = []
        for country in countries:
            base_population = np.random.randint(50_000_000, 330_000_000)
            
            for i, date in enumerate(date_range):
                # Simulate realistic COVID progression
                days_since_start = i
                
                # Wave patterns
                wave1 = np.exp(-((days_since_start - 100) / 50) ** 2) * 1000
                wave2 = np.exp(-((days_since_start - 300) / 60) ** 2) * 1500
                wave3 = np.exp(-((days_since_start - 600) / 70) ** 2) * 2000
                
                base_cases = max(0, wave1 + wave2 + wave3 + np.random.normal(0, 100))
                
                # Cumulative cases
                if i == 0:
                    confirmed = max(1, int(base_cases))
                    deaths = max(0, int(confirmed * 0.02))
                    recovered = max(0, int(confirmed * 0.8))
                else:
                    prev_confirmed = data[-1]['Confirmed'] if data else 0
                    confirmed = prev_confirmed + max(0, int(base_cases))
                    deaths = max(data[-1]['Deaths'] if data else 0, int(confirmed * np.random.uniform(0.01, 0.03)))
                    recovered = max(data[-1]['Recovered'] if data else 0, int(confirmed * np.random.uniform(0.85, 0.95)))
                
                # Vaccination data (starts from 2021)
                if date >= pd.Timestamp('2021-01-01'):
                    vaccination_progress = min(1.0, (date - pd.Timestamp('2021-01-01')).days / 365)
                    vaccinated = int(base_population * vaccination_progress * np.random.uniform(0.3, 0.8))
                    fully_vaccinated = int(vaccinated * np.random.uniform(0.7, 0.9))
                else:
                    vaccinated = 0
                    fully_vaccinated = 0
                
                data.append({
                    'Country/Region': country,
                    'Date': date,
                    'Confirmed': confirmed,
                    'Deaths': deaths,
                    'Recovered': recovered,
                    'New_Cases': max(0, int(base_cases)),
                    'New_Deaths': max(0, int(base_cases * 0.02)),
                    'Population': base_population,
                    'People_Vaccinated': vaccinated,
                    'People_Fully_Vaccinated': fully_vaccinated,
                    'Death_Rate': (deaths / confirmed * 100) if confirmed > 0 else 0,
                    'Vaccination_Rate': (vaccinated / base_population * 100) if base_population > 0 else 0
                })
        
        return pd.DataFrame(data)
    
    def create_global_overview_dashboard(self, data):
        """Create comprehensive global overview dashboard"""
        # Get latest data for each country
        latest_data = data.groupby('Country/Region').last().reset_index()
        
        # Global totals
        global_confirmed = latest_data['Confirmed'].sum()
        global_deaths = latest_data['Deaths'].sum()
        global_recovered = latest_data['Recovered'].sum() if 'Recovered' in latest_data.columns else 0
        global_vaccinated = latest_data['People_Vaccinated'].sum() if 'People_Vaccinated' in latest_data.columns else 0
        
        # Create subplots with proper specifications
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Global COVID-19 Overview', 'Top 10 Countries by Cases', 'Death Rate by Country',
                'Daily New Cases Trend', 'Vaccination Progress', 'Cases vs Deaths Correlation',
                'Regional Distribution', 'Recovery Rate Analysis', 'Key Metrics'
            ),
            specs=[
                [{"type": "scatter", "colspan": 2}, None, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # 1. Global Overview - Key Metrics Display
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4],
                y=[1, 1, 1, 1],
                mode='text',
                text=[
                    f'Total Cases<br><b>{global_confirmed:,}</b>',
                    f'Total Deaths<br><b>{global_deaths:,}</b>',
                    f'Total Recovered<br><b>{global_recovered:,}</b>',
                    f'Countries Affected<br><b>{len(latest_data)}</b>'
                ],
                textfont=dict(size=16, color=self.colors['text']),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # 2. Top 10 Countries by Cases
        top_countries = latest_data.nlargest(10, 'Confirmed')
        fig.add_trace(
            go.Bar(
                x=top_countries['Confirmed'],
                y=top_countries['Country/Region'],
                orientation='h',
                marker_color=self.colors['cases'],
                name='Confirmed Cases',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 3. Daily New Cases Trend (Global)
        daily_global = data.groupby('Date').agg({
            'New_Cases': 'sum',
            'New_Deaths': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_global['Date'],
                y=daily_global['New_Cases'].rolling(7).mean(),
                mode='lines',
                name='7-day Avg New Cases',
                line=dict(color=self.colors['cases'], width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Vaccination Progress (if available)
        if 'People_Vaccinated' in latest_data.columns:
            vax_countries = latest_data[latest_data['People_Vaccinated'] > 0].nlargest(10, 'Vaccination_Rate')
            if not vax_countries.empty:
                fig.add_trace(
                    go.Bar(
                        x=vax_countries['Country/Region'],
                        y=vax_countries['Vaccination_Rate'],
                        marker_color=self.colors['vaccinated'],
                        name='Vaccination Rate %',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # 5. Cases vs Deaths Correlation
        fig.add_trace(
            go.Scatter(
                x=latest_data['Confirmed'],
                y=latest_data['Deaths'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=latest_data['Death_Rate'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Death Rate %", x=1.02)
                ),
                text=latest_data['Country/Region'],
                name='Countries',
                showlegend=False
            ),
            row=2, col=3
        )
        
        # 6. Regional Distribution (Top 10)
        fig.add_trace(
            go.Pie(
                labels=top_countries['Country/Region'],
                values=top_countries['Confirmed'],
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3,
                textinfo='label+percent',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 7. Recovery Rate Analysis
        if 'Recovered' in latest_data.columns:
            latest_data['Recovery_Rate'] = (latest_data['Recovered'] / latest_data['Confirmed'] * 100).fillna(0)
            recovery_countries = latest_data[latest_data['Confirmed'] > 10000].nlargest(10, 'Recovery_Rate')
            
            if not recovery_countries.empty:
                fig.add_trace(
                    go.Bar(
                        x=recovery_countries['Country/Region'],
                        y=recovery_countries['Recovery_Rate'],
                        marker_color=self.colors['recovered'],
                        name='Recovery Rate %',
                        showlegend=False
                    ),
                    row=3, col=2
                )
        
        # 8. Key Metrics Table
        global_death_rate = (global_deaths / global_confirmed * 100) if global_confirmed > 0 else 0
        global_vax_rate = (global_vaccinated / latest_data['Population'].sum() * 100) if 'Population' in latest_data.columns else 0
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Cases', f"{global_confirmed:,}"],
            ['Total Deaths', f"{global_deaths:,}"],
            ['Global Death Rate', f"{global_death_rate:.2f}%"],
            ['Countries Affected', len(latest_data)],
            ['Avg Cases per Country', f"{global_confirmed/len(latest_data):,.0f}"],
            ['Highest Death Rate', f"{latest_data['Death_Rate'].max():.2f}%"],
            ['Most Affected Country', latest_data.loc[latest_data['Confirmed'].idxmax(), 'Country/Region']]
        ]
        
        if global_vax_rate > 0:
            metrics_data.append(['Global Vaccination Rate', f"{global_vax_rate:.1f}%"])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*metrics_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'COVID-19 Global Data Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10, color=self.colors['text']),
            showlegend=False,
            height=1200,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update axes styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        
        # Hide axes for the overview text display
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)
        
        fig.show()
    
    def create_country_analysis_dashboard(self, data, country='United States'):
        """Create detailed analysis for a specific country"""
        country_data = data[data['Country/Region'] == country].copy()
        
        if country_data.empty:
            print(f"❌ No data found for {country}")
            return
        
        country_data = country_data.sort_values('Date')
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f'{country} - Cases Timeline', f'{country} - Deaths Timeline', 'Daily New Cases',
                'Death Rate Trend', 'Vaccination Progress', 'Key Statistics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # 1. Cases Timeline
        fig.add_trace(
            go.Scatter(
                x=country_data['Date'],
                y=country_data['Confirmed'],
                mode='lines',
                name='Total Cases',
                line=dict(color=self.colors['cases'], width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        if 'Recovered' in country_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=country_data['Date'],
                    y=country_data['Recovered'],
                    mode='lines',
                    name='Recovered',
                    line=dict(color=self.colors['recovered'], width=2)
                ),
                row=1, col=1
            )
        
        # 2. Deaths Timeline
        fig.add_trace(
            go.Scatter(
                x=country_data['Date'],
                y=country_data['Deaths'],
                mode='lines',
                name='Total Deaths',
                line=dict(color=self.colors['deaths'], width=3),
                fill='tonexty'
            ),
            row=1, col=2
        )
        
        # 3. Daily New Cases
        fig.add_trace(
            go.Bar(
                x=country_data['Date'],
                y=country_data['New_Cases'].rolling(7).mean(),
                name='7-day Avg New Cases',
                marker_color=self.colors['cases'],
                opacity=0.7
            ),
            row=1, col=3
        )
        
        # 4. Death Rate Trend
        fig.add_trace(
            go.Scatter(
                x=country_data['Date'],
                y=country_data['Death_Rate'].rolling(7).mean(),
                mode='lines',
                name='7-day Avg Death Rate',
                line=dict(color=self.colors['deaths'], width=2)
            ),
            row=2, col=1
        )
        
        # 5. Vaccination Progress
        if 'People_Vaccinated' in country_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=country_data['Date'],
                    y=country_data['Vaccination_Rate'],
                    mode='lines',
                    name='Vaccination Rate',
                    line=dict(color=self.colors['vaccinated'], width=3),
                    fill='tonexty'
                ),
                row=2, col=2
            )
        
        # 6. Key Statistics Table
        latest = country_data.iloc[-1]
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Cases', f"{latest['Confirmed']:,}"],
            ['Total Deaths', f"{latest['Deaths']:,}"],
            ['Death Rate', f"{latest['Death_Rate']:.2f}%"],
            ['Latest New Cases', f"{latest['New_Cases']:,}"],
            ['Peak Daily Cases', f"{country_data['New_Cases'].max():,}"]
        ]
        
        if 'Vaccination_Rate' in country_data.columns:
            stats_data.append(['Vaccination Rate', f"{latest['Vaccination_Rate']:.1f}%"])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*stats_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'{country} COVID-19 Detailed Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def create_vaccination_analysis(self, data):
        """Create vaccination-focused analysis"""
        if 'People_Vaccinated' not in data.columns:
            print("❌ Vaccination data not available")
            return
        
        # Filter for countries with vaccination data
        vax_data = data[data['People_Vaccinated'] > 0].copy()
        latest_vax = vax_data.groupby('Country/Region').last().reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Vaccination Rate by Country', 'Vaccination Timeline (Top Countries)',
                'Vaccination vs Cases Correlation', 'Vaccination Progress Metrics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # 1. Vaccination Rate by Country
        top_vax = latest_vax.nlargest(15, 'Vaccination_Rate')
        fig.add_trace(
            go.Bar(
                x=top_vax['Country/Region'],
                y=top_vax['Vaccination_Rate'],
                marker_color=self.colors['vaccinated'],
                name='Vaccination Rate %'
            ),
            row=1, col=1
        )
        
        # 2. Vaccination Timeline
        top_vax_countries = top_vax.head(5)['Country/Region'].tolist()
        for country in top_vax_countries:
            country_vax = vax_data[vax_data['Country/Region'] == country]
            fig.add_trace(
                go.Scatter(
                    x=country_vax['Date'],
                    y=country_vax['Vaccination_Rate'],
                    mode='lines',
                    name=country,
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        # 3. Vaccination vs Cases Correlation
        fig.add_trace(
            go.Scatter(
                x=latest_vax['Vaccination_Rate'],
                y=latest_vax['Confirmed'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=latest_vax['Death_Rate'],
                    colorscale='Reds',
                    showscale=True
                ),
                text=latest_vax['Country/Region'],
                name='Countries'
            ),
            row=2, col=1
        )
        
        # 4. Vaccination Metrics Table
        global_population = latest_vax['Population'].sum() if 'Population' in latest_vax.columns else 0
        global_vaccinated = latest_vax['People_Vaccinated'].sum()
        global_fully_vax = latest_vax['People_Fully_Vaccinated'].sum() if 'People_Fully_Vaccinated' in latest_vax.columns else 0
        
        vax_metrics = [
            ['Metric', 'Value'],
            ['Countries with Vaccination Data', len(latest_vax)],
            ['Global Vaccination Rate', f"{(global_vaccinated/global_population*100):.1f}%" if global_population > 0 else "N/A"],
            ['Highest Vaccination Rate', f"{latest_vax['Vaccination_Rate'].max():.1f}%"],
            ['Average Vaccination Rate', f"{latest_vax['Vaccination_Rate'].mean():.1f}%"],
            ['Total People Vaccinated', f"{global_vaccinated:,}"],
            ['Fully Vaccinated', f"{global_fully_vax:,}"],
            ['Leading Country', latest_vax.loc[latest_vax['Vaccination_Rate'].idxmax(), 'Country/Region']]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.colors['vaccinated'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*vax_metrics[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'COVID-19 Global Vaccination Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800
        )
        
        fig.show()
    
    def create_mortality_analysis(self, data):
        """Create death rate and mortality analysis"""
        latest_data = data.groupby('Country/Region').last().reset_index()
        
        # Filter countries with significant cases for meaningful death rates
        significant_cases = latest_data[latest_data['Confirmed'] >= 10000].copy()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Death Rate by Country', 'Death Rate vs Total Cases',
                'Mortality Timeline (Global)', 'Mortality Statistics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # 1. Death Rate by Country
        top_death_rate = significant_cases.nlargest(15, 'Death_Rate')
        fig.add_trace(
            go.Bar(
                x=top_death_rate['Country/Region'],
                y=top_death_rate['Death_Rate'],
                marker_color=self.colors['deaths'],
                name='Death Rate %'
            ),
            row=1, col=1
        )
        
        # 2. Death Rate vs Total Cases
        fig.add_trace(
            go.Scatter(
                x=significant_cases['Confirmed'],
                y=significant_cases['Death_Rate'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=significant_cases['Deaths'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Total Deaths")
                ),
                text=significant_cases['Country/Region'],
                name='Countries'
            ),
            row=1, col=2
        )
        
        # 3. Global Mortality Timeline
        daily_global = data.groupby('Date').agg({
            'Deaths': 'sum',
            'Confirmed': 'sum'
        }).reset_index()
        daily_global['Global_Death_Rate'] = (daily_global['Deaths'] / daily_global['Confirmed'] * 100).fillna(0)
        
        fig.add_trace(
            go.Scatter(
                x=daily_global['Date'],
                y=daily_global['Global_Death_Rate'].rolling(7).mean(),
                mode='lines',
                name='7-day Avg Global Death Rate',
                line=dict(color=self.colors['deaths'], width=3)
            ),
            row=2, col=1
        )
        
        # 4. Mortality Statistics Table
        global_death_rate = (latest_data['Deaths'].sum() / latest_data['Confirmed'].sum() * 100)
        
        mortality_stats = [
            ['Statistic', 'Value'],
            ['Global Death Rate', f"{global_death_rate:.2f}%"],
            ['Highest Death Rate', f"{significant_cases['Death_Rate'].max():.2f}%"],
            ['Lowest Death Rate', f"{significant_cases['Death_Rate'].min():.2f}%"],
            ['Average Death Rate', f"{significant_cases['Death_Rate'].mean():.2f}%"],
            ['Countries > 5% Death Rate', len(significant_cases[significant_cases['Death_Rate'] > 5])],
            ['Countries < 1% Death Rate', len(significant_cases[significant_cases['Death_Rate'] < 1])],
            ['Most Deaths (Country)', latest_data.loc[latest_data['Deaths'].idxmax(), 'Country/Region']],
            ['Highest Death Rate (Country)', significant_cases.loc[significant_cases['Death_Rate'].idxmax(), 'Country/Region']]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value'],
                           fill_color=self.colors['deaths'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*mortality_stats[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': 'COVID-19 Mortality Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800
        )
        
        fig.show()
    
    def display_summary_report(self, data):
        """Display comprehensive text summary"""
        latest_data = data.groupby('Country/Region').last().reset_index()
        
        print(f"\n📊 COVID-19 GLOBAL DATA ANALYSIS REPORT")
        print("=" * 60)
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 Data Period: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Global Statistics
        global_confirmed = latest_data['Confirmed'].sum()
        global_deaths = latest_data['Deaths'].sum()
        global_recovered = latest_data['Recovered'].sum() if 'Recovered' in latest_data.columns else 0
        global_death_rate = (global_deaths / global_confirmed * 100) if global_confirmed > 0 else 0
        
        print(f"\n🌍 GLOBAL STATISTICS")
        print("-" * 30)
        print(f"🦠 Total Confirmed Cases: {global_confirmed:,}")
        print(f"💀 Total Deaths: {global_deaths:,}")
        if global_recovered > 0:
            print(f"💚 Total Recovered: {global_recovered:,}")
        print(f"📊 Global Death Rate: {global_death_rate:.2f}%")
        print(f"🌎 Countries Affected: {len(latest_data)}")
        
        # Top Affected Countries
        print(f"\n🔝 TOP 10 MOST AFFECTED COUNTRIES")
        print("-" * 40)
        top_countries = latest_data.nlargest(10, 'Confirmed')
        for i, (_, country) in enumerate(top_countries.iterrows(), 1):
            print(f"{i:2d}. {country['Country/Region']:<20} {country['Confirmed']:>10,} cases ({country['Death_Rate']:>5.2f}% death rate)")
        
        # Death Rate Analysis
        significant_cases = latest_data[latest_data['Confirmed'] >= 10000]
        print(f"\n💀 DEATH RATE ANALYSIS")
        print("-" * 30)
        print(f"📊 Average Death Rate: {significant_cases['Death_Rate'].mean():.2f}%")
        print(f"📈 Highest Death Rate: {significant_cases['Death_Rate'].max():.2f}% ({significant_cases.loc[significant_cases['Death_Rate'].idxmax(), 'Country/Region']})")
        print(f"📉 Lowest Death Rate: {significant_cases['Death_Rate'].min():.2f}% ({significant_cases.loc[significant_cases['Death_Rate'].idxmin(), 'Country/Region']})")
        
        # Vaccination Analysis
        if 'People_Vaccinated' in latest_data.columns:
            vax_data = latest_data[latest_data['People_Vaccinated'] > 0]
            if not vax_data.empty:
                print(f"\n💉 VACCINATION ANALYSIS")
                print("-" * 30)
                print(f"🌍 Countries with Vaccination Data: {len(vax_data)}")
                print(f"📊 Average Vaccination Rate: {vax_data['Vaccination_Rate'].mean():.1f}%")
                print(f"🥇 Highest Vaccination Rate: {vax_data['Vaccination_Rate'].max():.1f}% ({vax_data.loc[vax_data['Vaccination_Rate'].idxmax(), 'Country/Region']})")
                
                if 'Population' in vax_data.columns:
                    global_vax_rate = (vax_data['People_Vaccinated'].sum() / vax_data['Population'].sum() * 100)
                    print(f"🌐 Global Vaccination Rate: {global_vax_rate:.1f}%")
        
        # Trends Analysis
        recent_data = data[data['Date'] >= (data['Date'].max() - timedelta(days=7))]
        recent_global = recent_data.groupby('Date')['New_Cases'].sum()
        
        print(f"\n📈 RECENT TRENDS (Last 7 Days)")
        print("-" * 35)
        print(f"📊 Average Daily New Cases: {recent_global.mean():,.0f}")
        print(f"📈 Peak Daily Cases: {recent_global.max():,.0f}")
        print(f"📉 Lowest Daily Cases: {recent_global.min():,.0f}")
        
        trend = "📈 Increasing" if recent_global.iloc[-1] > recent_global.iloc[0] else "📉 Decreasing"
        print(f"🔄 Trend Direction: {trend}")
        
        print(f"\n✅ Analysis complete! Check your browser for interactive dashboards.")

def main():
    analyzer = COVID19DataAnalyzer()
    
    print("🦠 COVID-19 Global Data Analysis")
    print("💼 Power BI-Style Visualizations")
    print("=" * 50)
    
    print("\nSelect data source:")
    print("1. Our World in Data (OWID) - Comprehensive dataset")
    print("2. Johns Hopkins University - Time series data")
    print("3. Use mock data for demonstration")
    
    source_choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    # Load data based on choice
    if source_choice == '2':
        covid_data = analyzer.load_johns_hopkins_data()
    elif source_choice == '3':
        covid_data = analyzer.generate_mock_covid_data()
    else:
        covid_data = analyzer.load_owid_data()
    
    if covid_data is None or covid_data.empty:
        print("❌ Failed to load data. Exiting...")
        return
    
    print(f"\n✅ Data loaded successfully!")
    print(f"📊 Dataset contains {len(covid_data)} records")
    print(f"🌍 Countries: {covid_data['Country/Region'].nunique()}")
    print(f"📅 Date range: {covid_data['Date'].min().strftime('%Y-%m-%d')} to {covid_data['Date'].max().strftime('%Y-%m-%d')}")
    
    print("\nSelect analysis type:")
    print("1. Complete dashboard suite (Global + Country + Vaccination + Mortality)")
    print("2. Global overview only")
    print("3. Country-specific analysis")
    print("4. Vaccination analysis")
    print("5. Mortality analysis")
    
    analysis_choice = input("\nEnter choice (1-5): ").strip()
    
    if analysis_choice == '1':
        # Complete analysis
        print("\n🎨 Creating comprehensive COVID-19 dashboards...")
        analyzer.create_global_overview_dashboard(covid_data)
        
        # Country analysis for top affected country
        top_country = covid_data.groupby('Country/Region')['Confirmed'].max().idxmax()
        analyzer.create_country_analysis_dashboard(covid_data, top_country)
        
        analyzer.create_vaccination_analysis(covid_data)
        analyzer.create_mortality_analysis(covid_data)
        analyzer.display_summary_report(covid_data)
        
    elif analysis_choice == '2':
        analyzer.create_global_overview_dashboard(covid_data)
        analyzer.display_summary_report(covid_data)
        
    elif analysis_choice == '3':
        available_countries = sorted(covid_data['Country/Region'].unique())
        print(f"\nAvailable countries: {', '.join(available_countries[:10])}...")
        country = input("Enter country name: ").strip()
        
        if country in available_countries:
            analyzer.create_country_analysis_dashboard(covid_data, country)
        else:
            print(f"❌ Country '{country}' not found. Using United States as default.")
            analyzer.create_country_analysis_dashboard(covid_data, 'United States')
            
    elif analysis_choice == '4':
        analyzer.create_vaccination_analysis(covid_data)
        
    elif analysis_choice == '5':
        analyzer.create_mortality_analysis(covid_data)
        
    else:
        analyzer.create_global_overview_dashboard(covid_data)
    
    print(f"\n✅ COVID-19 analysis complete!")

if __name__ == "__main__":
    main()