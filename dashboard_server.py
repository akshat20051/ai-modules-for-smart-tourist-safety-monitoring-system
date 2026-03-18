#!/usr/bin/env python3
"""
Tourist Safety AI Dashboard - Backend API
Flask server that connects the AI system with the web dashboard.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import threading
import time

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection import TouristAnomalyDetector, DemoDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='.')
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for dashboard data
dashboard_data = {
    'tourists': {},
    'alerts': [],
    'zones': {},
    'statistics': {
        'total_tourists': 0,
        'active_alerts': 0,
        'avg_risk_score': 0.0,
        'safe_zones': 0
    },
    'risk_history': [],
    'crowd_data': {},
    'last_update': datetime.now(timezone.utc).isoformat()
}

# AI Detector instance
detector = None
demo_gen = None

class DashboardDataManager:
    """Manages real-time data for the dashboard."""
    
    def __init__(self):
        self.running = False
        self.update_interval = 5  # seconds
        
    async def initialize_ai_system(self):
        """Initialize the AI system for dashboard use."""
        global detector, demo_gen
        
        try:
            # Create mock Redis for dashboard demo
            class DashboardMockRedis:
                def __init__(self):
                    self.data = {}
                    self.tourists = {
                        'tourist_001': {'name': 'John Smith', 'nationality': 'USA', 'zone': 'Gateway of India'},
                        'tourist_002': {'name': 'Maria Garcia', 'nationality': 'Spain', 'zone': 'Marine Drive'},
                        'tourist_003': {'name': 'Chen Wei', 'nationality': 'China', 'zone': 'Colaba Market'},
                        'tourist_004': {'name': 'Sarah Johnson', 'nationality': 'UK', 'zone': 'Taj Hotel Area'},
                        'tourist_005': {'name': 'Raj Patel', 'nationality': 'India', 'zone': 'Gateway of India'}
                    }
                
                async def get(self, key): return self.data.get(key)
                async def set(self, key, value): self.data[key] = value
                async def sadd(self, key, value): pass
                async def smembers(self, key): 
                    if 'active_tourists' in key:
                        return set(self.tourists.keys())
                    elif 'active_zones' in key:
                        return {'zone_001', 'zone_002', 'zone_003', 'zone_004'}
                    return set()
                async def setex(self, key, ttl, value): self.data[key] = value
                async def lpush(self, key, value): pass
                async def lrange(self, key, start, end): return []
                async def delete(self, key): pass
                async def sismember(self, key, member): return False
            
            detector = TouristAnomalyDetector.__new__(TouristAnomalyDetector)
            detector.redis_client = DashboardMockRedis()
            
            # Initialize models
            from sklearn.ensemble import IsolationForest, RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            detector.scaler = StandardScaler()
            detector.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            detector.risk_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            detector.dbscan = DBSCAN(eps=0.0005, min_samples=5, metric='haversine')
            detector.lstm_model = None
            
            # Initialize detector methods
            detector._utcnow = lambda: datetime.now(timezone.utc)
            detector._seconds_since = lambda earlier, later=None: (
                (later or datetime.now(timezone.utc)) - earlier
            ).total_seconds()
            
            # Fit models
            dummy_data = np.random.random((100, 16))
            detector.scaler.fit(dummy_data)
            detector.isolation_forest.fit(dummy_data)
            
            demo_gen = DemoDataGenerator()
            logger.info("AI system initialized for dashboard")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI system: {e}")
            raise
    
    async def update_dashboard_data(self):
        """Update dashboard data with AI analysis."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Get active tourists
            active_tourists = await detector.redis_client.smembers('active_tourists')
            tourist_info = detector.redis_client.tourists
            
            # Generate and analyze tourist data
            tourists_data = {}
            risk_scores = []
            alerts = []
            
            for tourist_id in active_tourists:
                # Generate realistic data
                tourist_data = demo_gen.generate_normal_tourist_data(tourist_id)
                
                # Randomly add some anomalies for demo
                import random
                if random.random() < 0.3:  # 30% chance of anomaly
                    anomaly_types = ['panic', 'missing', 'restricted_zone', 'health_emergency']
                    anomaly_type = random.choice(anomaly_types)
                    tourist_data = demo_gen.generate_anomaly_tourist_data(tourist_id, anomaly_type)
                
                # AI Analysis
                risk_score = detector.predict_risk_score(tourist_data)
                risk_scores.append(risk_score)
                
                # Location anomaly detection
                location_anomaly = await detector.detect_location_anomaly(tourist_id, tourist_data.get('location', {}))
                
                # Missing person check
                missing_check = await detector.detect_missing_person_pattern(tourist_id)
                
                # Store tourist data
                tourists_data[tourist_id] = {
                    'id': tourist_id,
                    'name': tourist_info[tourist_id]['name'],
                    'nationality': tourist_info[tourist_id]['nationality'],
                    'location': tourist_data['location'],
                    'risk_score': risk_score,
                    'risk_level': 'high' if risk_score > 70 else 'medium' if risk_score > 40 else 'low',
                    'heart_rate': tourist_data['heart_rate'],
                    'temperature': tourist_data['body_temperature'],
                    'last_update': tourist_data['location']['timestamp'],
                    'zone': tourist_info[tourist_id]['zone'],
                    'status': 'normal'
                }
                
                # Generate alerts
                if location_anomaly.get('anomaly_detected'):
                    alert = {
                        'id': f"alert_{int(current_time.timestamp())}_{tourist_id}",
                        'tourist_id': tourist_id,
                        'tourist_name': tourist_info[tourist_id]['name'],
                        'type': location_anomaly['anomaly_type'],
                        'severity': location_anomaly['risk_level'],
                        'message': f"{location_anomaly['anomaly_type'].replace('_', ' ').title()} detected",
                        'location': tourist_data['location'],
                        'timestamp': current_time.isoformat(),
                        'recommendations': location_anomaly['recommendations']
                    }
                    alerts.append(alert)
                    tourists_data[tourist_id]['status'] = 'alert'
                
                if missing_check.get('missing_probability', 0) > 70:
                    alert = {
                        'id': f"missing_{int(current_time.timestamp())}_{tourist_id}",
                        'tourist_id': tourist_id,
                        'tourist_name': tourist_info[tourist_id]['name'],
                        'type': 'missing_person',
                        'severity': 'high',
                        'message': f"Missing person alert - {missing_check['missing_probability']}% probability",
                        'location': tourist_data['location'],
                        'timestamp': current_time.isoformat(),
                        'recommendations': ['Initiate search protocol', 'Contact emergency services']
                    }
                    alerts.append(alert)
                    tourists_data[tourist_id]['status'] = 'missing'
            
            # Update global dashboard data
            dashboard_data['tourists'] = tourists_data
            dashboard_data['alerts'] = alerts[-20:]  # Keep last 20 alerts
            
            # Update statistics
            dashboard_data['statistics'] = {
                'total_tourists': len(tourists_data),
                'active_alerts': len(alerts),
                'avg_risk_score': round(sum(risk_scores) / len(risk_scores) if risk_scores else 0, 1),
                'safe_zones': 4 - len([a for a in alerts if 'zone' in a.get('type', '')])
            }
            
            # Add risk history point
            dashboard_data['risk_history'].append({
                'timestamp': current_time.isoformat(),
                'avg_risk': dashboard_data['statistics']['avg_risk_score']
            })
            
            # Keep only last 20 points
            dashboard_data['risk_history'] = dashboard_data['risk_history'][-20:]
            
            # Update crowd data
            zones = ['Gateway of India', 'Marine Drive', 'Colaba Market', 'Taj Hotel Area']
            dashboard_data['crowd_data'] = {
                zone: {
                    'occupancy': random.randint(20, 95),
                    'capacity': random.randint(100, 300),
                    'risk_level': random.choice(['low', 'medium', 'high'])
                } for zone in zones
            }
            
            dashboard_data['last_update'] = current_time.isoformat()
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def start_background_updates(self):
        """Start background thread for continuous updates."""
        def update_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def continuous_update():
                await self.initialize_ai_system()
                while self.running:
                    await self.update_dashboard_data()
                    await asyncio.sleep(self.update_interval)
            
            loop.run_until_complete(continuous_update())
        
        self.running = True
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        logger.info("Background update thread started")

# Initialize data manager
data_manager = DashboardDataManager()

# API Routes
@app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template('index.html')

@app.route('/api/dashboard')
def get_dashboard_data():
    """Get all dashboard data."""
    return jsonify(dashboard_data)

@app.route('/api/tourists')
def get_tourists():
    """Get tourist data."""
    return jsonify(dashboard_data['tourists'])

@app.route('/api/alerts')
def get_alerts():
    """Get active alerts."""
    return jsonify(dashboard_data['alerts'])

@app.route('/api/statistics')
def get_statistics():
    """Get dashboard statistics."""
    return jsonify(dashboard_data['statistics'])

@app.route('/api/risk-history')
def get_risk_history():
    """Get risk score history."""
    return jsonify(dashboard_data['risk_history'])

@app.route('/api/crowd-data')
def get_crowd_data():
    """Get crowd density data."""
    return jsonify(dashboard_data['crowd_data'])

@app.route('/api/emergency/<tourist_id>')
def trigger_emergency(tourist_id):
    """Trigger emergency protocol for a tourist."""
    try:
        if tourist_id in dashboard_data['tourists']:
            alert = {
                'id': f"emergency_{int(datetime.now(timezone.utc).timestamp())}_{tourist_id}",
                'tourist_id': tourist_id,
                'tourist_name': dashboard_data['tourists'][tourist_id]['name'],
                'type': 'emergency_triggered',
                'severity': 'critical',
                'message': 'Manual emergency protocol activated',
                'location': dashboard_data['tourists'][tourist_id]['location'],
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': ['Dispatch immediate help', 'Contact emergency services', 'Notify police']
            }
            dashboard_data['alerts'].append(alert)
            dashboard_data['tourists'][tourist_id]['status'] = 'emergency'
            
            return jsonify({'success': True, 'alert': alert})
        else:
            return jsonify({'success': False, 'error': 'Tourist not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start-monitoring')
def start_monitoring():
    """Start the AI monitoring system."""
    if not data_manager.running:
        data_manager.start_background_updates()
        return jsonify({'success': True, 'message': 'Monitoring started'})
    else:
        return jsonify({'success': True, 'message': 'Monitoring already running'})

if __name__ == '__main__':
    print("🚀 Starting Tourist Safety AI Dashboard Server...")
    print("=" * 60)
    print("🌐 Dashboard URL: http://localhost:5000")
    print("📊 API Endpoint: http://localhost:5000/api/dashboard")
    print("🔄 Auto-refresh: Every 5 seconds")
    print("=" * 60)
    
    # Start background monitoring
    data_manager.start_background_updates()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)