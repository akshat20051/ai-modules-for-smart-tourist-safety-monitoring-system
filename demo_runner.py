#!/usr/bin/env python3
"""
Demo Runner for Tourist Safety AI System
This script demonstrates the AI's capabilities without requiring Redis setup.
"""

import os
import sys
import json
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add the current directory to path so we can import anomaly_detection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly_detection import TouristAnomalyDetector, DemoDataGenerator

class OfflineDemoRunner:
    """Run AI demos without Redis dependency."""
    
    def __init__(self):
        self.detector = None
        self.demo_gen = DemoDataGenerator()
        
    async def initialize_detector(self):
        """Initialize detector with offline mode."""
        print("🔧 Initializing AI system (offline mode)...")
        try:
            # Mock Redis client for offline demo
            class MockRedis:
                def __init__(self):
                    self.data = {}
                    self.location_histories = {}
                    self.tourist_data = {}
                
                async def get(self, key): 
                    # Return stored data if available
                    if key in self.data:
                        return self.data[key]
                    
                    # Default mock data for specific keys
                    if 'last_update' in key:
                        return '{"timestamp": "2025-09-14T01:50:00+00:00", "location": {"lat": 18.9220, "lon": 72.8347}}'
                    return None
                
                async def set(self, key, value): 
                    self.data[key] = value
                
                async def sadd(self, key, value): pass
                
                async def smembers(self, key): 
                    if 'active_tourists' in key:
                        return {'demo_tourist_001', 'demo_tourist_002', 'demo_tourist_003'}
                    elif 'active_zones' in key:
                        return {'zone_001', 'zone_002', 'zone_003'}
                    return set()
                
                async def setex(self, key, ttl, value): 
                    self.data[key] = value
                
                async def lpush(self, key, value): pass
                
                async def lrange(self, key, start, end):
                    # Return mock location history
                    if 'location_history' in key:
                        history = []
                        base_time = datetime.now(timezone.utc)
                        for i in range(5):  # Return 5 history points
                            history.append(json.dumps({
                                'lat': 18.9220 + (i * 0.0001),
                                'lon': 72.8347 + (i * 0.0001),
                                'timestamp': (base_time - timedelta(minutes=i*10)).isoformat()
                            }))
                        return history
                    return []
                
                async def delete(self, key): 
                    if key in self.data:
                        del self.data[key]
                
                async def sismember(self, key, member): return False
            
            self.detector = TouristAnomalyDetector.__new__(TouristAnomalyDetector)
            self.detector.redis_client = MockRedis()
            
            # Initialize the detector manually
            from sklearn.ensemble import IsolationForest, RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            import tensorflow as tf
            from tensorflow import keras
            
            self.detector.scaler = StandardScaler()
            self.detector.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.detector.risk_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.detector.dbscan = DBSCAN(eps=0.0005, min_samples=5, metric='haversine')
            self.detector.lstm_model = None  # Skip LSTM for quick demo
            
            # Fit scaler with dummy data
            dummy_data = np.random.random((100, 16))
            self.detector.scaler.fit(dummy_data)
            self.detector.isolation_forest.fit(dummy_data)
            
            print("✅ AI Models initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing detector: {e}")
            raise
    
    async def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all AI capabilities."""
        print("🚀 Tourist Safety AI - Comprehensive Demo")
        print("=" * 60)
        print("🌟 This AI system protects tourists through:")
        print("   • Real-time location anomaly detection")
        print("   • Missing person pattern recognition") 
        print("   • Emergency response automation")
        print("   • Crowd density and stampede risk analysis")
        print("   • Health emergency detection")
        print("   • Restricted zone monitoring")
        print()
        
        await self.initialize_detector()
        
        scenarios = [
            ("👥 Normal Tourist Activity", "normal", "Low risk scenario with typical tourist behavior"),
            ("🚨 Emergency - Panic Button", "panic", "Tourist pressed panic button - immediate response needed"),
            ("🔍 Missing Person Alert", "missing", "Tourist hasn't been seen for several hours"),
            ("⚠️ Restricted Zone Entry", "restricted_zone", "Tourist entered high-risk or forbidden area"),
            ("🏥 Health Emergency", "health_emergency", "Abnormal vital signs detected"),
            ("🌪️ Erratic Movement Pattern", "unusual_movement", "Unusual or suspicious movement detected")
        ]
        
        results_summary = []
        
        for i, (scenario_name, scenario_type, description) in enumerate(scenarios, 1):
            print(f"\n📋 Test Case {i}: {scenario_name}")
            print(f"📖 Description: {description}")
            print("-" * 50)
            
            tourist_id = f"demo_tourist_{i:03d}"
            
            # Generate scenario data
            if scenario_type == "normal":
                tourist_data = self.demo_gen.generate_normal_tourist_data(tourist_id)
            else:
                tourist_data = self.demo_gen.generate_anomaly_tourist_data(tourist_id, scenario_type)
            
            # Display input data
            location = tourist_data['location']
            print(f"📍 Location: {location['lat']:.4f}°N, {location['lon']:.4f}°E")
            print(f"💓 Vital Signs: HR={tourist_data['heart_rate']:.0f} bpm, Temp={tourist_data['body_temperature']:.1f}°C")
            print(f"⏱️ Last Update: {tourist_data['time_since_last_update']} seconds ago")
            
            # Run AI analysis
            print("\n🤖 AI Analysis:")
            
            # Risk scoring
            risk_score = self.detector.predict_risk_score(tourist_data)
            risk_level = "🟢 LOW" if risk_score < 30 else "🟡 MEDIUM" if risk_score < 70 else "🔴 HIGH"
            print(f"   📊 Risk Score: {risk_score:.1f}/100 ({risk_level})")
            
            # Store tourist data in mock redis for proper analysis
            await self.detector.redis_client.set(f'tourist:{tourist_id}:current', json.dumps(tourist_data))
            await self.detector.redis_client.set(f'tourist:{tourist_id}:last_update', json.dumps({
                'timestamp': tourist_data['location']['timestamp'],
                'location': tourist_data['location'],
                'panic_pressed': tourist_data.get('panic_button_pressed', 0),
                'zone_risk_level': tourist_data.get('current_zone_risk', 0),
                'device_disconnected': scenario_type == 'missing',
                'route_deviation': 6000 if scenario_type == 'restricted_zone' else 0
            }))
            
            # Location anomaly detection
            location_anomaly = await self.detector.detect_location_anomaly(tourist_id, location)
            
            # Missing person detection  
            missing_check = await self.detector.detect_missing_person_pattern(tourist_id)
            
            # Display results
            if location_anomaly['anomaly_detected']:
                print(f"   🚨 ANOMALY DETECTED: {location_anomaly['anomaly_type'].replace('_', ' ').title()}")
                print(f"   ⚡ Severity: {location_anomaly['risk_level'].upper()}")
                if location_anomaly['recommendations']:
                    print(f"   💡 AI Recommendations:")
                    for rec in location_anomaly['recommendations']:
                        print(f"      • {rec}")
                
                # Generate alert
                alert = await self.detector.generate_alert(location_anomaly)
                print(f"   🔔 Alert ID: {alert['id']}")
                print(f"   🚨 Auto-Response: {', '.join(alert['auto_response']) if alert['auto_response'] else 'None'}")
                
                results_summary.append(f"❌ {scenario_name}: {location_anomaly['anomaly_type']} detected")
            else:
                print("   ✅ No location anomalies detected")
                results_summary.append(f"✅ {scenario_name}: Normal behavior")
            
            if missing_check['missing_probability'] > 20:
                print(f"   🔍 Missing Person Risk: {missing_check['missing_probability']}%")
                if missing_check['indicators']:
                    print(f"   ⚠️ Warning Indicators: {', '.join(missing_check['indicators'])}")
            
            await asyncio.sleep(1)  # Brief pause for readability
        
        # Crowd analysis demo
        print(f"\n\n🏙️ Crowd Density Analysis Demo")
        print("=" * 50)
        
        for zone_id, zone_data in self.demo_gen.demo_zones.items():
            # Simulate varying crowd levels
            crowd_scenarios = [
                (zone_data['capacity'] // 4, "Normal crowd levels"),
                (int(zone_data['capacity'] * 0.8), "High occupancy"),
                (int(zone_data['capacity'] * 1.2), "Over capacity - Risk scenario")
            ]
            
            for tourist_count, scenario_desc in crowd_scenarios:
                print(f"\n🏛️ {zone_data['name']} - {scenario_desc}")
                
                # Mock crowd data
                class MockRedisWithCrowd:
                    def __init__(self, tourist_count, capacity):
                        self.tourist_count = tourist_count
                        self.capacity = capacity
                        
                    async def get(self, key): 
                        if 'capacity' in key: return str(self.capacity)
                        if 'last_update' in key:
                            return '{"timestamp": "2025-09-14T01:50:00+00:00", "location": {"lat": 18.9220, "lon": 72.8347}}'
                        return None
                        
                    async def smembers(self, key): 
                        if 'tourists' in key: return {f'tourist_{i}' for i in range(self.tourist_count)}
                        return set()
                        
                    async def set(self, key, value): pass
                    async def sadd(self, key, value): pass
                    async def setex(self, key, ttl, value): pass
                    async def lpush(self, key, value): pass
                    async def lrange(self, key, start, end): return []
                    async def delete(self, key): pass
                    async def sismember(self, key, member): return False
                
                self.detector.redis_client = MockRedisWithCrowd(tourist_count, zone_data['capacity'])
                density_analysis = await self.detector.analyze_crowd_density(zone_id)
                
                occupancy = (tourist_count / zone_data['capacity']) * 100
                print(f"   👥 Current Occupancy: {tourist_count}/{zone_data['capacity']} ({occupancy:.0f}%)")
                print(f"   📊 Density Level: {density_analysis['density_level'].upper()}")
                
                if density_analysis['stampede_risk'] > 0:
                    print(f"   ⚠️ Stampede Risk: {density_analysis['stampede_risk']}%")
                
                if density_analysis['recommendations']:
                    print(f"   🎯 AI Recommendations:")
                    for rec in density_analysis['recommendations']:
                        print(f"      • {rec}")
        
        # Final summary
        print(f"\n\n📈 Demo Results Summary")
        print("=" * 50)
        for result in results_summary:
            print(f"   {result}")
        
        print(f"\n✨ Key AI Capabilities Demonstrated:")
        print(f"   🎯 Machine Learning Models: Isolation Forest, Random Forest, DBSCAN")
        print(f"   📊 Real-time Risk Scoring: Dynamic threat assessment")
        print(f"   🚨 Automated Alert Generation: Immediate response triggers")
        print(f"   👥 Crowd Intelligence: Stampede prevention analytics")
        print(f"   🌍 Geographic Analysis: Location pattern recognition")
        print(f"   ⏱️ Temporal Analysis: Time-based anomaly detection")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"🚀 Ready for deployment in tourist safety monitoring systems.")

async def main():
    """Main demo runner."""
    demo = OfflineDemoRunner()
    try:
        await demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())