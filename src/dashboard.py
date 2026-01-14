import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import sys

# Page Config
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = "data/predictions"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
PREDICTIONS_FILE = os.path.join(DATA_DIR, "future_predictions_v2.csv")
VALUE_BETS_FILE = os.path.join(DATA_DIR, "value_bets_v2.csv")
MODEL_RESULTS_FILE = os.path.join(RESULTS_DIR, "model_results_v2.csv")
HISTORICAL_FILE = os.path.join(PROCESSED_DIR, "processed_features_extended.csv")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #464b5c;
    }
    .value-bet {
        color: #00FF00;
        font-weight: bold;
    }
    .team-stat {
        font-size: 18px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        preds = pd.read_csv(PREDICTIONS_FILE)
        if 'date' in preds.columns:
            preds['date'] = pd.to_datetime(preds['date']).dt.date
            
        value_bets = pd.read_csv(VALUE_BETS_FILE)
        if 'date' in value_bets.columns:
            value_bets['date'] = pd.to_datetime(value_bets['date']).dt.date
            
        model_results = pd.read_csv(MODEL_RESULTS_FILE, index_col=0)
        
        return preds, value_bets, model_results
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_historical_data():
    """Load historical match data for team stats"""
    try:
        df = pd.read_csv(HISTORICAL_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        return None

def get_team_stats(df, team_name, num_matches=10):
    """Calculate statistics for a specific team"""
    # Find matches as home or away
    home_matches = df[df['home_team'] == team_name].sort_values('date', ascending=False)
    away_matches = df[df['away_team'] == team_name].sort_values('date', ascending=False)
    
    # Recent form (last N matches)
    all_matches = pd.concat([
        home_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league']].assign(venue='home'),
        away_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league']].assign(venue='away')
    ]).sort_values('date', ascending=False).head(num_matches)
    
    if len(all_matches) == 0:
        return None
    
    # Calculate form
    wins = draws = losses = goals_scored = goals_conceded = 0
    form = []
    
    for _, match in all_matches.iterrows():
        if match['venue'] == 'home':
            gs = match['home_score']
            gc = match['away_score']
        else:
            gs = match['away_score']
            gc = match['home_score']
        
        goals_scored += gs
        goals_conceded += gc
        
        if gs > gc:
            wins += 1
            form.append('W')
        elif gs < gc:
            losses += 1
            form.append('L')
        else:
            draws += 1
            form.append('D')
    
    return {
        'matches_played': len(all_matches),
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'goal_diff': goals_scored - goals_conceded,
        'points': wins * 3 + draws,
        'form': ''.join(form[:5]),  # Last 5 only
        'recent_matches': all_matches,
        'home_matches': len(home_matches.head(num_matches)),
        'away_matches': len(away_matches.head(num_matches))
    }

def main():
    st.title("⚽ Football Match Predictor AI")
    st.markdown("### Powered by XGBoost & 1,000+ Features")
    
    preds, value_bets, model_results = load_data()
    historical_df = load_historical_data()
    
    if preds is None:
        st.warning("Please run the training pipeline first to generate predictions.")
        return

    # Sidebar - simplified
    st.sidebar.header("⚙️ Options")
    only_value_bets = st.sidebar.checkbox("Show Only Value Bets (EV > 0)", value=False)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(preds))
    with col2:
        st.metric("Value Bets (EV > 15%)", len(value_bets))
    with col3:
        test_acc = model_results.loc['Test', 'accuracy'] if 'Test' in model_results.index else 0
        st.metric("Model Test Accuracy", f"{float(test_acc):.1%}")
    with col4:
        st.metric("Best ROI Zone", "Odds 1.5-2.0")

    # Tabs - Added High Confidence Strategy
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Upcoming Predictions", 
        "🎯 High Confidence (60%+)",
        "💎 Value Bets", 
        "📊 Model Performance", 
        "🔍 Team Search",
        "⚙️ Data Management"
    ])
    
    with tab1:
        st.header(f"Upcoming Matches ({len(preds)})")
        
        display_cols = [
            'date', 'league', 'home_team', 'away_team', 
            'pred_home_prob', 'pred_draw_prob', 'pred_away_prob',
            'predicted_result', 'confidence'
        ]
        
        leagues = sorted(preds['league'].unique())
        selected_league = st.multiselect("Filter by League", leagues, default=leagues)
        
        view_df = preds[preds['league'].isin(selected_league)].copy()
        
        def calculate_pred_ev(row):
            p = row['predicted_result']
            if p == 1:
                odds = row.get('odds_1x2_home', 0)
                prob = row.get('pred_home_prob', 0)
            elif p == 0:
                odds = row.get('odds_1x2_draw', 0)
                prob = row.get('pred_draw_prob', 0)
            else:
                odds = row.get('odds_1x2_away', 0)
                prob = row.get('pred_away_prob', 0)
            
            if pd.notna(odds) and odds > 0:
                return (prob * odds) - 1
            return 0

        view_df['Prediction EV'] = view_df.apply(calculate_pred_ev, axis=1)
        
        if only_value_bets:
            view_df = view_df[view_df['Prediction EV'] > 0]
            st.info(f"Showing {len(view_df)} value bets (EV > 0)")
            
        view_df['EV %'] = (view_df['Prediction EV'] * 100).map('{:+.1f}%'.format)
        view_df['Home Win %'] = (view_df['pred_home_prob'] * 100).map('{:.1f}%'.format)
        view_df['Draw %'] = (view_df['pred_draw_prob'] * 100).map('{:.1f}%'.format)
        view_df['Away Win %'] = (view_df['pred_away_prob'] * 100).map('{:.1f}%'.format)
        view_df['Prediction'] = view_df['predicted_result'].map({1: 'Home', 0: 'Draw', -1: 'Away'})
        
        st.dataframe(
            view_df[['date', 'league', 'home_team', 'away_team', 
                     'Prediction', 'Home Win %', 'Draw %', 'Away Win %', 'EV %']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "EV %": st.column_config.TextColumn(
                    "Expected Value",
                    help="ROI of the predicted outcome (Probability * Odds - 1)",
                )
            }
        )

    with tab2:
        st.header("🎯 High Confidence Picks")
        st.success("**Strategy Stats**: 71.6% Win Rate | +15.9% ROI in backtest (on matches where model was most confident)")
        
        # Get confidence stats
        max_conf = preds['confidence'].max() * 100
        min_conf = preds['confidence'].min() * 100
        median_conf = preds['confidence'].median() * 100
        
        st.info(f"📊 Current predictions: Confidence range {min_conf:.1f}% - {max_conf:.1f}% (median: {median_conf:.1f}%)")
        
        # User-adjustable threshold
        conf_threshold = st.slider(
            "Confidence Threshold (%)", 
            min_value=int(min_conf), 
            max_value=int(max_conf), 
            value=int(median_conf),
            step=1,
            help="Show matches where model confidence exceeds this threshold"
        ) / 100.0
        
        # Filter for high confidence
        high_conf = preds[preds['confidence'] >= conf_threshold].copy()
        
        if len(high_conf) == 0:
            st.warning(f"No matches meet the {conf_threshold*100:.0f}%+ confidence threshold. Try lowering it.")
        else:
            # Calculate EV for each bet
            def calc_ev_high_conf(row):
                p = row['predicted_result']
                if p == 1:
                    odds = row.get('odds_1x2_home', 0)
                    prob = row.get('pred_home_prob', 0)
                elif p == 0:
                    odds = row.get('odds_1x2_draw', 0)
                    prob = row.get('pred_draw_prob', 0)
                else:
                    odds = row.get('odds_1x2_away', 0)
                    prob = row.get('pred_away_prob', 0)
                if pd.notna(odds) and odds > 0:
                    return (prob * odds) - 1
                return 0
            
            high_conf['EV'] = high_conf.apply(calc_ev_high_conf, axis=1)
            high_conf['Bet Type'] = high_conf['predicted_result'].map({1: 'HOME', 0: 'DRAW', -1: 'AWAY'})
            
            # Get odds for bet
            def get_bet_odds(row):
                p = row['predicted_result']
                if p == 1:
                    return row.get('odds_1x2_home', 0)
                elif p == 0:
                    return row.get('odds_1x2_draw', 0)
                else:
                    return row.get('odds_1x2_away', 0)
            
            high_conf['Bet Odds'] = high_conf.apply(get_bet_odds, axis=1)
            
            # Stats columns
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Qualifying Bets", len(high_conf))
            with col_s2:
                avg_conf = high_conf['confidence'].mean() * 100
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with col_s3:
                avg_odds = high_conf['Bet Odds'].mean()
                st.metric("Avg Odds", f"{avg_odds:.2f}")
            
            st.markdown("---")
            
            # Breakdown by bet type
            st.subheader("By Bet Type")
            type_stats = high_conf.groupby('Bet Type').agg({
                'home_team': 'count',
                'confidence': 'mean',
                'Bet Odds': 'mean'
            }).reset_index()
            type_stats.columns = ['Bet Type', 'Count', 'Avg Confidence', 'Avg Odds']
            type_stats['Avg Confidence'] = (type_stats['Avg Confidence'] * 100).map('{:.1f}%'.format)
            type_stats['Avg Odds'] = type_stats['Avg Odds'].map('{:.2f}'.format)
            st.dataframe(type_stats, use_container_width=True, hide_index=True)
            
            # Breakdown by league
            st.subheader("By League")
            league_stats = high_conf.groupby('league').agg({
                'home_team': 'count',
                'confidence': 'mean'
            }).reset_index()
            league_stats.columns = ['League', 'Bets', 'Avg Confidence']
            league_stats['Avg Confidence'] = (league_stats['Avg Confidence'] * 100).map('{:.1f}%'.format)
            league_stats = league_stats.sort_values('Bets', ascending=False)
            st.dataframe(league_stats, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("All Qualifying Bets")
            
            # Format for display
            display_hc = high_conf.copy()
            display_hc['Confidence'] = (display_hc['confidence'] * 100).map('{:.1f}%'.format)
            display_hc['EV %'] = (display_hc['EV'] * 100).map('{:+.1f}%'.format)
            display_hc['Bet Odds'] = display_hc['Bet Odds'].map('{:.2f}'.format)
            
            st.dataframe(
                display_hc[['date', 'league', 'home_team', 'away_team', 'Bet Type', 'Bet Odds', 'Confidence', 'EV %']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.TextColumn(
                        "Model Confidence",
                        help="Model's confidence in this prediction (>60% = high confidence)"
                    ),
                    "Bet Type": "Bet On"
                }
            )

    with tab3:
        st.header("Value Betting Opportunities 💎")
        st.info(f"Showing bets with EV > 15% (Conservative Strategy)")
        
        bet_types = ['HOME', 'DRAW', 'AWAY']
        selected_types = st.multiselect("Bet Type", bet_types, default=bet_types)
        
        vb_view = value_bets[value_bets['best_bet'].isin(selected_types)].copy()
        
        vb_view['EV %'] = (vb_view['best_ev'] * 100).map('{:+.1f}%'.format)
        vb_view['Model Prob'] = vb_view.apply(
            lambda x: x['pred_home_prob'] if x['best_bet']=='HOME' else (x['pred_draw_prob'] if x['best_bet']=='DRAW' else x['pred_away_prob']), 
            axis=1
        )
        vb_view['Model Prob'] = (vb_view['Model Prob'] * 100).map('{:.1f}%'.format)
        
        def get_odds(row):
            if row['best_bet'] == 'HOME': return row['odds_1x2_home']
            if row['best_bet'] == 'DRAW': return row['odds_1x2_draw']
            return row['odds_1x2_away']
            
        vb_view['Odds'] = vb_view.apply(get_odds, axis=1).map('{:.2f}'.format)
        
        st.dataframe(
            vb_view[['date', 'league', 'home_team', 'away_team', 'best_bet', 'Odds', 'Model Prob', 'EV %']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "EV %": st.column_config.TextColumn(
                    "Expected Value",
                    help="Theoretical ROI based on model probability",
                ),
                "best_bet": "Bet On"
            }
        )
        
        fig = px.bar(
            vb_view.head(15), 
            x='best_ev', 
            y='home_team', 
            orientation='h',
            title="Top 15 Value Bets by EV",
            labels={'best_ev': 'Expected Value (>0.15 is good)', 'home_team': 'Match (Home Team)'},
            color='best_ev',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Model Analytics 📊")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.subheader("Accuracy by Split")
            acc_data = pd.DataFrame({
                'Split': ['Train', 'Valid', 'Test'],
                'Accuracy': [
                    float(model_results.loc['Train', 'accuracy']),
                    float(model_results.loc['Valid', 'accuracy']),
                    float(model_results.loc['Test', 'accuracy'])
                ]
            })
            fig_acc = px.bar(acc_data, x='Split', y='Accuracy', text_auto='.1%', title="Model Accuracy Check")
            fig_acc.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col_m2:
            st.subheader("ROI Analysis (80/20 Validation)")
            roi_data = pd.DataFrame({
                'Odds Range': ['1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0'],
                'ROI': [-0.058, 0.062, -0.015, 0.105],
                'Verdict': ['Avoid', 'Best Value', 'Neutral', 'High Risk']
            })
            
            colors = ['red' if x < 0 else 'green' for x in roi_data['ROI']]
            
            fig_roi = go.Figure(data=[
                go.Bar(x=roi_data['Odds Range'], y=roi_data['ROI'], marker_color=colors, text=roi_data['ROI'].apply(lambda x: f"{x:+.1%}"))
            ])
            fig_roi.update_layout(title="Return on Investment (ROI) by Odds", yaxis_tickformat=".1%")
            st.plotly_chart(fig_roi, use_container_width=True)

        st.markdown("---")
        st.subheader("Top Features")
        try:
            st.image("results/feature_importance.png", caption="Top 20 Features Driving the Model")
        except:
            st.text("Feature importance image not found.")

    with tab5:
        st.header("🔍 Team Search & Stats")
        
        if historical_df is None:
            st.warning("Historical data not found. Run process_data_extended.py first.")
        else:
            # Get all unique teams
            all_teams = sorted(set(
                list(historical_df['home_team'].unique()) + 
                list(historical_df['away_team'].unique())
            ))
            
            # Team search
            selected_team = st.selectbox(
                "Search for a team",
                options=[""] + all_teams,
                format_func=lambda x: "Type to search..." if x == "" else x
            )
            
            if selected_team:
                stats = get_team_stats(historical_df, selected_team, num_matches=10)
                
                if stats:
                    st.subheader(f"📊 {selected_team}")
                    
                    # Form display
                    form_display = ""
                    for char in stats['form']:
                        if char == 'W':
                            form_display += "🟢"
                        elif char == 'D':
                            form_display += "🟡"
                        else:
                            form_display += "🔴"
                    
                    st.markdown(f"**Recent Form (Last 5):** {form_display}")
                    
                    # Stats in columns
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Matches (Last 10)", stats['matches_played'])
                    with c2:
                        st.metric("Record", f"{stats['wins']}W-{stats['draws']}D-{stats['losses']}L")
                    with c3:
                        st.metric("Goals", f"{stats['goals_scored']}-{stats['goals_conceded']}")
                    with c4:
                        st.metric("Points", stats['points'])
                    
                    st.markdown("---")
                    
                    # Recent matches table
                    st.subheader("Recent Matches")
                    recent = stats['recent_matches'].copy()
                    recent['Result'] = recent.apply(
                        lambda x: f"{int(x['home_score'])}-{int(x['away_score'])}", axis=1
                    )
                    recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        recent[['date', 'league', 'home_team', 'away_team', 'Result', 'venue']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Goals over time chart
                    st.subheader("Goals Trend (Last 10 Matches)")
                    goals_data = []
                    for _, match in stats['recent_matches'].iterrows():
                        if match['venue'] == 'home':
                            goals_data.append({'date': match['date'], 'Goals Scored': match['home_score'], 'Goals Conceded': match['away_score']})
                        else:
                            goals_data.append({'date': match['date'], 'Goals Scored': match['away_score'], 'Goals Conceded': match['home_score']})
                    
                    goals_df = pd.DataFrame(goals_data).sort_values('date')
                    
                    fig_goals = px.line(
                        goals_df, x='date', y=['Goals Scored', 'Goals Conceded'],
                        title=f"{selected_team} - Goals per Match",
                        markers=True
                    )
                    st.plotly_chart(fig_goals, use_container_width=True)
                    
                    # Upcoming matches for this team
                    st.subheader("Upcoming Matches")
                    upcoming = preds[
                        (preds['home_team'] == selected_team) | 
                        (preds['away_team'] == selected_team)
                    ]
                    if len(upcoming) > 0:
                        st.dataframe(
                            upcoming[['date', 'league', 'home_team', 'away_team', 'pred_home_prob', 'pred_draw_prob', 'pred_away_prob']],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No upcoming matches scheduled for this team.")
                else:
                    st.warning(f"No historical data found for {selected_team}")

    with tab6:
        st.header("⚙️ Data Management")
        st.markdown("Run data scrapers and refresh predictions from here.")
        
        st.warning("⚠️ **Note:** Local scraping may fail due to Cloudflare blocking. If scripts fail, use the Colab notebooks instead.")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📅 Future Matches")
            st.markdown("Scrape upcoming matches for next 7 days")
            
            if st.button("🔄 Scrape Future Matches", key="btn_future"):
                with st.spinner("Scraping future matches..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "src/scrape_future.py"],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            st.success("✅ Future matches scraped successfully!")
                            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                        else:
                            st.error("❌ Scraping failed")
                            st.code(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("⏱️ Timeout - scraping took too long")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col_b:
            st.subheader("📊 Historical Data")
            st.markdown("Update with new finished matches")
            
            if st.button("🔄 Update Historical Data", key="btn_historical"):
                with st.spinner("Updating historical data..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "src/scrape_update.py"],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )
                        if result.returncode == 0:
                            st.success("✅ Historical data updated!")
                            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                        else:
                            st.error("❌ Update failed")
                            st.code(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("⏱️ Timeout - update took too long")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("---")
        
        st.subheader("🤖 Retrain Model")
        st.markdown("Process data and generate new predictions")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("📥 Process Data", key="btn_process"):
                with st.spinner("Processing features..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "src/process_data_extended.py"],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            st.success("✅ Data processed!")
                            st.code(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
                        else:
                            st.error("❌ Processing failed")
                            st.code(result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col_d:
            if st.button("🧠 Train & Predict", key="btn_train"):
                with st.spinner("Training model and generating predictions..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "src/train_model_v2.py"],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            st.success("✅ Model trained! New predictions generated.")
                            st.code(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                            st.info("🔄 Refresh the page to see new predictions")
                        else:
                            st.error("❌ Training failed")
                            st.code(result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("---")
        st.subheader("📁 Colab Notebooks")
        st.markdown("""
        If local scraping fails, use these notebooks on Google Colab:
        - **Future Matches:** `data/predictions/sofascore_future_v2.ipynb`
        - **Historical Update:** `data/predictions/incremental_update.ipynb`
        - **Full Scraper:** `src/sofascore_colab.ipynb`
        """)

if __name__ == "__main__":
    main()
