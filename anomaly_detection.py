"""
AI-based Anomaly Detection System for Tourist Safety Monitoring
Corrected and hardened implementation (async-friendly, robust datetime handling,
scaler usage fixed, DBSCAN for geographic clustering using haversine,
safer model loading/saving, and other fixes requested).

Drop this file into your project and review the NOTES and CONFIG block below.
"""

import os
import json
import joblib
import logging
import traceback
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# TensorFlow / Keras imports (optional if not training LSTM locally)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from geopy.distance import geodesic

# Use sync redis if aioredis not available; wrap blocking calls with to_thread
try:
    import redis
    SYNC_REDIS_AVAILABLE = True
except Exception:
    SYNC_REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------
# CONFIG / CONSTANTS
# -----------------
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
IFOREST_PATH = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
RISK_CLF_PATH = os.path.join(MODEL_DIR, 'risk_classifier.pkl')
LSTM_PATH = os.path.join(MODEL_DIR, 'lstm_anomaly.h5')

# DBSCAN haversine eps (radians). 0.001 ~ 6.37 km (approx). Tune per deployment.
DBSCAN_EPS_RAD = float(os.environ.get('DBSCAN_EPS_RAD', 0.0005))
DBSCAN_MIN_SAMPLES = int(os.environ.get('DBSCAN_MIN_SAMPLES', 5))

# thresholds
LOCATION_DROPOUT_SECONDS = int(os.environ.get('LOCATION_DROPOUT_SECONDS', 3600))
SIGNIFICANT_ROUTE_DEVIATION_METERS = int(os.environ.get('ROUTE_DEVIATION_METERS', 5000))

# -----------------
# Helper wrappers
# -----------------

class AsyncRedisWrapper:
    """Simple async wrapper around sync redis-py client using asyncio.to_thread.
    This avoids an external dependency on aioredis while keeping an async
    interface for the rest of the code.
    """
    def __init__(self, host='localhost', port=6379, decode_responses=True):
        if not SYNC_REDIS_AVAILABLE:
            raise RuntimeError('redis package not available. Please `pip install redis`')
        self._client = redis.Redis(host=host, port=port, decode_responses=decode_responses)

    async def get(self, key: str) -> Optional[str]:
        return await asyncio.to_thread(self._client.get, key)

    async def set(self, key: str, value: str):
        return await asyncio.to_thread(self._client.set, key, value)

    async def setex(self, key: str, ttl: int, value: str):
        return await asyncio.to_thread(self._client.setex, key, ttl, value)

    async def lpush(self, key: str, value: str):
        return await asyncio.to_thread(self._client.lpush, key, value)

    async def lrange(self, key: str, start: int, end: int):
        return await asyncio.to_thread(self._client.lrange, key, start, end)

    async def smembers(self, key: str):
        return await asyncio.to_thread(self._client.smembers, key)

    async def sismember(self, key: str, member: str):
        return await asyncio.to_thread(self._client.sismember, key, member)

    async def sadd(self, key: str, member: str):
        return await asyncio.to_thread(self._client.sadd, key, member)

    async def delete(self, key: str):
        return await asyncio.to_thread(self._client.delete, key)

    def blocking_client(self):
        """Expose the sync client for operations where async wrapper not needed."""
        return self._client

# -----------------
# Main Detector
# -----------------

class TouristAnomalyDetector:
    """Main class for detecting anomalies in tourist behavior.

    Notes:
      - This implementation prefers async Redis calls (via AsyncRedisWrapper)
        but will work synchronously if you replace redis_client with a direct
        redis.Redis instance and call blocking methods.
      - All datetime math uses timezone-aware datetimes (UTC) to avoid ambiguity.
    """

    def __init__(self, redis_host='127.0.0.1', redis_port=6379):
        # Async wrapper (uses sync redis internally) so we don't require aioredis.
        if SYNC_REDIS_AVAILABLE:
            self.redis_client = AsyncRedisWrapper(host=redis_host, port=redis_port)
        else:
            raise RuntimeError('redis library is required. pip install redis')

        # Models and utils
        self.scaler = StandardScaler()
        self.isolation_forest: Optional[IsolationForest] = None
        self.lstm_model: Optional[keras.Model] = None
        self.risk_classifier: Optional[RandomForestClassifier] = None

        # DBSCAN configured to use haversine metric; set eps in radians when used
        self.dbscan = DBSCAN(eps=DBSCAN_EPS_RAD, min_samples=DBSCAN_MIN_SAMPLES, metric='haversine')

        # Ensure model dir exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Try load models
        self.load_models()

    # -----------------
    # Utility helpers
    # -----------------
    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def _seconds_since(self, earlier: datetime, later: Optional[datetime] = None) -> float:
        """Return total seconds between two datetimes (use total_seconds)."""
        later = later or self._utcnow()
        if earlier.tzinfo is None:
            earlier = earlier.replace(tzinfo=timezone.utc)
        return (later - earlier).total_seconds()

    # -----------------
    # Model load/save
    # -----------------
    def load_models(self):
        """Load pre-trained models or initialize new ones (improved error handling)."""
        try:
            if os.path.exists(IFOREST_PATH):
                self.isolation_forest = joblib.load(IFOREST_PATH)
            if os.path.exists(RISK_CLF_PATH):
                self.risk_classifier = joblib.load(RISK_CLF_PATH)
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)

            if os.path.exists(LSTM_PATH):
                self.lstm_model = keras.models.load_model(LSTM_PATH)

            # If isolation forest missing, initialize defaults
            if self.isolation_forest is None or self.risk_classifier is None:
                logger.info('One or more models missing; initializing new models')
                self.initialize_models()
            else:
                logger.info('Loaded existing models successfully')

        except Exception as exc:
            logger.warning(f'Could not load models: {exc}')
            logger.debug(traceback.format_exc())
            # Initialize models if anything goes wrong
            self.initialize_models()
            logger.info('Initialized new models')

    def initialize_models(self):
        """Initialize ML models for anomaly detection"""
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        # DBSCAN already configured in __init__
        # Build LSTM model for sequence anomaly detection (feature_dim must match extract_features)
        self.lstm_model = self.build_lstm_model(feature_dim=16)

    def save_models(self):
        """Save trained models to disk (atomic save where possible)."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            joblib.dump(self.isolation_forest, IFOREST_PATH)
            joblib.dump(self.risk_classifier, RISK_CLF_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            if self.lstm_model is not None:
                # Save to temp then move for atomicity
                tmp_path = LSTM_PATH + '.tmp'
                self.lstm_model.save(tmp_path)
                os.replace(tmp_path, LSTM_PATH)
            logger.info('Models saved successfully')
        except Exception:
            logger.exception('Failed to save models')

    # -----------------
    # LSTM builder
    # -----------------
    def build_lstm_model(self, timesteps: Optional[int] = None, feature_dim: int = 16) -> keras.Model:
        """
        Build an LSTM model for sequence anomaly detection.
        feature_dim must match the number of features produced by extract_features().
        timesteps can be None for variable-length sequences (use input_shape=(None, feature_dim))
        """
        input_shape = (timesteps, feature_dim) if timesteps else (None, feature_dim)
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    # -----------------
    # Feature extraction
    # -----------------
    def extract_features(self, tourist_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from tourist data. Returns shape (1, feature_dim).
        NOTE: Keep this consistent with LSTM feature_dim (default 16 here).
        """
        features: List[float] = []

        # Location-based features -> 6 values
        if 'location_history' in tourist_data and isinstance(tourist_data['location_history'], list):
            locations = tourist_data['location_history']
            if len(locations) > 1:
                distances = []
                speeds = []
                for i in range(1, len(locations)):
                    prev = locations[i - 1]
                    cur = locations[i]
                    # support both 'lat'/'lon' and 'latitude'/'longitude'
                    prev_lat = prev.get('lat') or prev.get('latitude')
                    prev_lon = prev.get('lon') or prev.get('longitude')
                    cur_lat = cur.get('lat') or cur.get('latitude')
                    cur_lon = cur.get('lon') or cur.get('longitude')

                    if prev_lat is None or prev_lon is None or cur_lat is None or cur_lon is None:
                        continue

                    # allow timestamps as strings or datetimes
                    t_prev = prev['timestamp'] if isinstance(prev['timestamp'], datetime) else datetime.fromisoformat(prev['timestamp'])
                    t_cur = cur['timestamp'] if isinstance(cur['timestamp'], datetime) else datetime.fromisoformat(cur['timestamp'])

                    dist = geodesic((prev_lat, prev_lon), (cur_lat, cur_lon)).meters
                    time_diff = self._seconds_since(t_prev, t_cur)
                    if time_diff > 0:
                        speeds.append(dist / time_diff)
                    distances.append(dist)

                features.extend([
                    float(np.mean(distances)) if distances else 0.0,
                    float(np.std(distances)) if distances else 0.0,
                    float(np.mean(speeds)) if speeds else 0.0,
                    float(np.std(speeds)) if speeds else 0.0,
                    float(max(distances)) if distances else 0.0,
                    float(min(distances)) if distances else 0.0
                ])
            else:
                features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 6)

        # Time features (use UTC)
        now = self._utcnow()
        features.append(float(now.hour))
        features.append(float(now.weekday()))

        # Activity features
        features.append(float(tourist_data.get('time_since_last_update', 0)))
        features.append(float(tourist_data.get('panic_button_pressed', 0)))

        # Zone risk features
        features.append(float(tourist_data.get('current_zone_risk', 0)))
        features.append(float(tourist_data.get('restricted_zone_entry', 0)))

        # Communication features
        features.append(float(tourist_data.get('emergency_contacts_notified', 0)))
        features.append(float(tourist_data.get('sos_signals', 0)))

        # Health metrics
        features.append(float(tourist_data.get('heart_rate', 70.0)))
        features.append(float(tourist_data.get('body_temperature', 36.5)))

        # Ensure final shape is (1, 16)
        feat_array = np.array(features, dtype=float).reshape(1, -1)
        return feat_array

    # -----------------
    # Location & movement anomaly detection
    # -----------------
    async def detect_location_anomaly(self, tourist_id: str, current_location: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in tourist location patterns (async version)."""
        anomaly_result = {
            'tourist_id': tourist_id,
            'anomaly_detected': False,
            'anomaly_type': None,
            'risk_level': 'low',
            'recommendations': []
        }

        location_history = await self.get_location_history(tourist_id)
        if not location_history:
            return anomaly_result

        # Check for sudden location drop-off
        last_update = location_history[-1] if location_history else None
        if last_update:
            last_ts = last_update.get('timestamp')
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts)
            time_diff = self._seconds_since(last_ts, self._utcnow())
            if time_diff > LOCATION_DROPOUT_SECONDS:
                anomaly_result['anomaly_detected'] = True
                anomaly_result['anomaly_type'] = 'location_dropout'
                anomaly_result['risk_level'] = 'high' if time_diff > (2 * LOCATION_DROPOUT_SECONDS) else 'medium'
                anomaly_result['recommendations'].append('Initiate contact attempt')

        # Check for deviation from planned route
        if current_location and 'itinerary' in current_location and current_location.get('lat') is not None:
            deviation = self.calculate_route_deviation(current_location, location_history)
            if deviation > SIGNIFICANT_ROUTE_DEVIATION_METERS:
                anomaly_result['anomaly_detected'] = True
                anomaly_result['anomaly_type'] = 'route_deviation'
                anomaly_result['risk_level'] = 'medium'
                anomaly_result['recommendations'].append('Send route correction notification')

        # Check for unusual speed or movement pattern
        if len(location_history) > 2:
            movement_anomaly = self.detect_movement_anomaly(location_history)
            if movement_anomaly:
                anomaly_result['anomaly_detected'] = True
                anomaly_result['anomaly_type'] = 'unusual_movement'
                anomaly_result['risk_level'] = movement_anomaly['risk_level']
                anomaly_result['recommendations'].extend(movement_anomaly['recommendations'])

        return anomaly_result

    def detect_movement_anomaly(self, location_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect anomalies in movement patterns (fixed time diff handling)."""
        if len(location_history) < 3:
            return None

        speeds: List[float] = []
        for i in range(1, len(location_history)):
            prev = location_history[i - 1]
            cur = location_history[i]
            prev_lat = prev.get('lat') or prev.get('latitude')
            prev_lon = prev.get('lon') or prev.get('longitude')
            cur_lat = cur.get('lat') or cur.get('latitude')
            cur_lon = cur.get('lon') or cur.get('longitude')

            if prev_lat is None or prev_lon is None or cur_lat is None or cur_lon is None:
                continue

            t_prev = prev['timestamp'] if isinstance(prev['timestamp'], datetime) else datetime.fromisoformat(prev['timestamp'])
            t_cur = cur['timestamp'] if isinstance(cur['timestamp'], datetime) else datetime.fromisoformat(cur['timestamp'])

            dist = geodesic((prev_lat, prev_lon), (cur_lat, cur_lon)).meters
            time_diff = self._seconds_since(t_prev, t_cur)
            if time_diff > 0:
                speed_kmh = (dist / time_diff) * 3.6
                speeds.append(speed_kmh)

        if not speeds:
            return None

        avg_speed = float(np.mean(speeds))
        max_speed = float(max(speeds))
        std_speed = float(np.std(speeds))

        # Detect anomalies with clear thresholds (document them)
        if max_speed > 300:  # Likely GPS spoofing or wrong units
            return {
                'type': 'impossible_speed',
                'risk_level': 'high',
                'recommendations': ['Verify GPS data', 'Check for spoofing or device tampering']
            }
        if avg_speed < 0.5 and len(speeds) > 10:
            return {
                'type': 'stationary_too_long',
                'risk_level': 'medium',
                'recommendations': ['Send wellness check', 'Attempt device ping']
            }
        if std_speed > 50:
            return {
                'type': 'erratic_movement',
                'risk_level': 'medium',
                'recommendations': ['Monitor closely', 'Alert nearby patrol']
            }

        return None

    def calculate_route_deviation(self, current_location: Dict[str, Any], location_history: List[Dict[str, Any]]) -> float:
        """Calculate deviation from planned route (meters)."""
        if 'planned_route' not in current_location:
            return 0.0

        planned_route = current_location['planned_route']
        cur_lat = current_location.get('lat') or current_location.get('latitude')
        cur_lon = current_location.get('lon') or current_location.get('longitude')
        if cur_lat is None or cur_lon is None:
            return 0.0

        current_pos = (cur_lat, cur_lon)
        min_distance = float('inf')
        for point in planned_route:
            pt_lat = point.get('lat') or point.get('latitude')
            pt_lon = point.get('lon') or point.get('longitude')
            if pt_lat is None or pt_lon is None:
                continue
            distance = geodesic(current_pos, (pt_lat, pt_lon)).meters
            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else 0.0

    # -----------------
    # Risk scoring
    # -----------------
    def predict_risk_score(self, tourist_data: Dict[str, Any]) -> float:
        """
        Predict risk score with proper scaler usage. Returns 0-100 float.
        """
        features = self.extract_features(tourist_data)  # shape (1, feature_dim)
        try:
            # Use transform — scaler must be fitted during training
            features_scaled = self.scaler.transform(features)
        except Exception:
            # If scaler was never fitted (e.g., fresh model), use features unscaled but log
            logger.warning('Scaler not fitted yet; using raw features for risk prediction')
            features_scaled = features

        anomaly_score = 0.0
        if self.isolation_forest is not None:
            try:
                anomaly_score = float(self.isolation_forest.decision_function(features_scaled)[0])
            except Exception:
                logger.exception('Error computing anomaly score')
                anomaly_score = 0.0

        base = 50.0 - anomaly_score * 50.0
        risk_score = float(np.clip(base, 0.0, 100.0))

        # Override adjustments for explicit flags
        if tourist_data.get('panic_button_pressed'):
            risk_score = max(risk_score, 90.0)
        if tourist_data.get('restricted_zone_entry'):
            risk_score = max(risk_score, 70.0)
        if tourist_data.get('time_since_last_update', 0) > 7200:
            risk_score = max(risk_score, 80.0)

        return float(risk_score)

    # -----------------
    # Missing person detection
    # -----------------
    async def detect_missing_person_pattern(self, tourist_id: str) -> Dict[str, Any]:
        """Detect patterns indicating a potentially missing person (async)."""
        result = {
            'tourist_id': tourist_id,
            'missing_probability': 0,
            'indicators': [],
            'last_known_location': None,
            'time_missing': 0
        }

        last_data_raw = await self.redis_client.get(f'tourist:{tourist_id}:last_update')
        if not last_data_raw:
            return result

        try:
            last_data = json.loads(last_data_raw)
        except Exception:
            logger.exception('Invalid JSON in last_update')
            return result

        last_ts = last_data.get('timestamp')
        if isinstance(last_ts, str):
            last_update_time = datetime.fromisoformat(last_ts)
        elif isinstance(last_ts, datetime):
            last_update_time = last_ts
        else:
            last_update_time = self._utcnow()

        time_since_update = self._seconds_since(last_update_time, self._utcnow())
        result['time_missing'] = time_since_update
        result['last_known_location'] = last_data.get('location')

        indicators = []
        probability = 0

        if time_since_update > 3600:
            indicators.append('no_recent_update')
            probability += 20
        if time_since_update > 7200:
            probability += 30

        if last_data.get('device_disconnected'):
            indicators.append('device_disconnected')
            probability += 25

        if last_data.get('zone_risk_level', 0) > 7:
            indicators.append('high_risk_zone')
            probability += 20

        if last_data.get('panic_pressed'):
            indicators.append('panic_activated')
            probability += 40

        if last_data.get('route_deviation', 0) > SIGNIFICANT_ROUTE_DEVIATION_METERS:
            indicators.append('significant_route_deviation')
            probability += 15

        result['indicators'] = indicators
        result['missing_probability'] = min(100, probability)
        return result

    # -----------------
    # Crowd analysis
    # -----------------
    async def analyze_crowd_density(self, zone_id: str) -> Dict[str, Any]:
        """Analyze crowd density and detect potential stampede risks (async)."""
        tourists_in_zone = await self.redis_client.smembers(f'zone:{zone_id}:tourists') or set()
        if isinstance(tourists_in_zone, list):
            tourists_in_zone = set(tourists_in_zone)

        density_analysis = {
            'zone_id': zone_id,
            'tourist_count': len(tourists_in_zone),
            'density_level': 'low',
            'stampede_risk': 0,
            'recommendations': []
        }

        zone_capacity_raw = await self.redis_client.get(f'zone:{zone_id}:capacity')
        if zone_capacity_raw:
            try:
                capacity = int(zone_capacity_raw)
                occupancy_rate = len(tourists_in_zone) / max(1, capacity)
                if occupancy_rate < 0.5:
                    density_analysis['density_level'] = 'low'
                elif occupancy_rate < 0.75:
                    density_analysis['density_level'] = 'medium'
                elif occupancy_rate < 0.9:
                    density_analysis['density_level'] = 'high'
                    density_analysis['recommendations'].append('Monitor crowd flow')
                else:
                    density_analysis['density_level'] = 'critical'
                    density_analysis['stampede_risk'] = min(100, int(occupancy_rate * 100))
                    density_analysis['recommendations'].extend([
                        'Activate crowd control measures',
                        'Deploy additional security',
                        'Consider entry restrictions'
                    ])
            except Exception:
                logger.exception('Invalid zone capacity value')

        if len(tourists_in_zone) > 50:
            movement_data = await self.analyze_crowd_movement(tourists_in_zone)
            if movement_data.get('convergence_detected'):
                density_analysis['stampede_risk'] = min(100, density_analysis['stampede_risk'] + 30)
                density_analysis['recommendations'].append('Redirect crowd flow')

        return density_analysis

    async def analyze_crowd_movement(self, tourist_ids: set) -> Dict[str, Any]:
        """Analyze crowd movement patterns using DBSCAN with haversine metric."""
        movements = []
        # sample up to 200 tourists for analysis
        for tourist_id in list(tourist_ids)[:200]:
            raw = await self.redis_client.get(f'location:{tourist_id}')
            if not raw:
                continue
            try:
                m = json.loads(raw)
            except Exception:
                continue
            # expect either 'lat'/'lon' or 'latitude'/'longitude'
            lat = m.get('lat') or m.get('latitude')
            lon = m.get('lon') or m.get('longitude')
            if lat is None or lon is None:
                continue
            movements.append((float(lat), float(lon)))

        if len(movements) < 10:
            return {'convergence_detected': False}

        # Convert to radians for haversine
        locations_array = np.radians(np.array(movements))
        try:
            clustering = self.dbscan.fit_predict(locations_array)
            unique_clusters = len(set(clustering)) - (1 if -1 in clustering else 0)
            if unique_clusters <= 1 and len(movements) > 20:
                return {'convergence_detected': True}
        except Exception:
            logger.exception('DBSCAN clustering failed')

        return {'convergence_detected': False}

    # -----------------
    # Storage & retrieval
    # -----------------
    async def get_location_history(self, tourist_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get location history for a tourist (async)."""
        history = []
        history_key = f'tourist:{tourist_id}:location_history'
        raw_history = await self.redis_client.lrange(history_key, 0, -1) or []
        for item in raw_history:
            try:
                location_data = json.loads(item)
                history.append(location_data)
            except Exception:
                logger.debug('Skipping invalid history item')
                continue

        cutoff_time = self._utcnow() - timedelta(hours=hours)
        filtered = [h for h in history if datetime.fromisoformat(h['timestamp']) > cutoff_time]
        return sorted(filtered, key=lambda x: x['timestamp'])

    # -----------------
    # Alerts
    # -----------------
    async def generate_alert(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alert based on anomaly detection and store in Redis (async)."""
        alert = {
            'id': f"alert_{int(self._utcnow().timestamp())}_{anomaly_data.get('tourist_id', 'unknown')}",
            'timestamp': self._utcnow().isoformat(),
            'tourist_id': anomaly_data.get('tourist_id'),
            'alert_type': anomaly_data.get('anomaly_type', 'unknown'),
            'severity': anomaly_data.get('risk_level', 'low'),
            'description': self.get_alert_description(anomaly_data),
            'recommendations': anomaly_data.get('recommendations', []),
            'auto_response': self.determine_auto_response(anomaly_data)
        }

        try:
            await self.redis_client.setex(f"alert:{alert['id']}", 3600 * 24, json.dumps(alert))
            await self.redis_client.lpush('alert_queue', json.dumps(alert))
        except Exception:
            logger.exception('Failed to write alert to Redis')

        return alert

    def get_alert_description(self, anomaly_data: Dict[str, Any]) -> str:
        """Generate human-readable alert description"""
        anomaly_type = anomaly_data.get('anomaly_type', 'unknown')
        descriptions = {
            'location_dropout': 'Tourist location not updated for extended period',
            'route_deviation': 'Tourist has deviated significantly from planned route',
            'unusual_movement': 'Unusual movement pattern detected',
            'restricted_zone': 'Tourist entered restricted area',
            'panic_signal': 'Panic button activated by tourist',
            'health_emergency': 'Abnormal health metrics detected',
            'missing_person': 'Tourist shows patterns consistent with missing person'
        }
        return descriptions.get(anomaly_type, 'Anomaly detected in tourist behavior')

    def determine_auto_response(self, anomaly_data: Dict[str, Any]) -> List[str]:
        responses = []
        risk_level = anomaly_data.get('risk_level', 'low')
        anomaly_type = anomaly_data.get('anomaly_type')

        if risk_level == 'high':
            responses.append('notify_emergency_services')
            responses.append('alert_nearest_patrol')

        if anomaly_type == 'panic_signal':
            responses.append('dispatch_immediate_help')
            responses.append('notify_emergency_contacts')

        if anomaly_type == 'location_dropout':
            responses.append('attempt_device_reconnection')
            responses.append('check_last_known_location')

        if anomaly_type == 'health_emergency':
            responses.append('dispatch_medical_team')
            responses.append('notify_nearest_hospital')

        return responses

    # -----------------
    # Training
    # -----------------
    def train_models(self, training_data: pd.DataFrame):
        """Train the ML models with historical data (blocking call)."""
        feature_columns = [col for col in training_data.columns if col not in ['tourist_id', 'anomaly', 'timestamp']]
        X = training_data[feature_columns].values
        y = training_data['anomaly'].values if 'anomaly' in training_data.columns else None

        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        if y is not None:
            self.risk_classifier.fit(X_scaled, y)
        self.save_models()
        logger.info('Models trained successfully')

    # -----------------
    # Continuous monitoring loop (async)
    # -----------------
    async def continuous_monitoring(self):
        """Continuously monitor all tourists for anomalies (async loop)."""
        logger.info('🚀 Starting continuous monitoring loop')
        print("🤖 AI Tourist Safety System - Live Monitoring")
        print("=" * 50)
        print("⏰ Monitoring started at:", self._utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        print()
        
        monitoring_cycle = 0
        while True:
            try:
                monitoring_cycle += 1
                print(f"🔄 Monitoring Cycle #{monitoring_cycle} - {self._utcnow().strftime('%H:%M:%S')}")
                
                active_tourists = await self.redis_client.smembers('active_tourists') or set()
                if isinstance(active_tourists, list):
                    active_tourists = set(active_tourists)

                if not active_tourists:
                    print("   📭 No active tourists found in system")
                else:
                    print(f"   👥 Monitoring {len(active_tourists)} active tourists")

                # Process tourists concurrently but limited
                sem = asyncio.Semaphore(50)
                anomalies_detected = 0
                high_risk_tourists = 0

                async def process_tourist(tid: str):
                    nonlocal anomalies_detected, high_risk_tourists
                    async with sem:
                        raw = await self.redis_client.get(f'tourist:{tid}:current')
                        if not raw:
                            return
                        try:
                            tourist_data = json.loads(raw)
                        except Exception:
                            return

                        # Calculate risk score first
                        risk_score = self.predict_risk_score(tourist_data)
                        if risk_score > 70:
                            high_risk_tourists += 1
                            print(f"   ⚠️ HIGH RISK: Tourist {tid} - Risk Score: {risk_score:.1f}")

                        location = tourist_data.get('location', {})
                        location_anomaly = await self.detect_location_anomaly(tid, location)
                        if location_anomaly.get('anomaly_detected'):
                            anomalies_detected += 1
                            alert = await self.generate_alert(location_anomaly)
                            print(f"   🚨 ANOMALY: {tid} - {location_anomaly['anomaly_type']} (Risk: {location_anomaly['risk_level']})")
                            logger.warning(f"Anomaly detected for tourist {tid}: {alert['id']}")

                        missing_check = await self.detect_missing_person_pattern(tid)
                        if missing_check.get('missing_probability', 0) > 70:
                            alert_data = {
                                'tourist_id': tid,
                                'anomaly_type': 'missing_person',
                                'risk_level': 'high',
                                'recommendations': ['Initiate search protocol', 'File E-FIR']
                            }
                            alert = await self.generate_alert(alert_data)
                            print(f"   🆘 MISSING PERSON ALERT: {tid} - Probability: {missing_check['missing_probability']}%")
                            logger.critical(f"Possible missing person: {tid}")

                        # store risk score as string for Redis
                        await self.redis_client.set(f'tourist:{tid}:risk_score', json.dumps({'score': risk_score, 'ts': self._utcnow().isoformat()}))

                # spawn tasks
                tasks = [asyncio.create_task(process_tourist(tid)) for tid in active_tourists]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Zones analysis
                zones = await self.redis_client.smembers('active_zones') or set()
                if isinstance(zones, list):
                    zones = set(zones)
                
                crowded_zones = 0
                for zone_id in zones:
                    density = await self.analyze_crowd_density(zone_id)
                    if density.get('density_level') in ['high', 'critical']:
                        crowded_zones += 1
                        print(f"   🏙️ CROWD ALERT: Zone {zone_id} - {density['density_level'].upper()} density ({density['tourist_count']} tourists)")
                    
                    if density.get('stampede_risk', 0) > 50:
                        print(f"   ⚡ STAMPEDE RISK: Zone {zone_id} - Risk: {density['stampede_risk']}%")
                        logger.warning(f"High stampede risk in zone {zone_id}: {density}")

                # Summary for this cycle
                if anomalies_detected > 0 or high_risk_tourists > 0 or crowded_zones > 0:
                    print(f"   📊 Cycle Summary: {anomalies_detected} anomalies, {high_risk_tourists} high-risk tourists, {crowded_zones} crowded zones")
                else:
                    print("   ✅ All systems normal - No anomalies detected")
                
                print()  # Empty line for readability
                await asyncio.sleep(30)
                
            except Exception:
                logger.exception('Error in continuous monitoring')
                print(f"   ❌ Error in monitoring cycle #{monitoring_cycle}")
                await asyncio.sleep(60)


# -----------------
# API wrapper (lightweight)
# -----------------
class AnomalyDetectionAPI:
    def __init__(self, redis_host='127.0.0.1', redis_port=6379):
        self.detector = TouristAnomalyDetector(redis_host=redis_host, redis_port=redis_port)

    async def check_tourist_anomaly(self, tourist_id: str) -> Dict[str, Any]:
        raw = await self.detector.redis_client.get(f'tourist:{tourist_id}:current')
        if not raw:
            return {'error': 'Tourist not found'}
        tourist_data = json.loads(raw)
        location_anomaly = await self.detector.detect_location_anomaly(tourist_id, tourist_data.get('location', {}))
        missing_check = await self.detector.detect_missing_person_pattern(tourist_id)
        risk_score = self.detector.predict_risk_score(tourist_data)
        return {
            'tourist_id': tourist_id,
            'risk_score': risk_score,
            'location_anomaly': location_anomaly,
            'missing_probability': missing_check.get('missing_probability', 0),
            'timestamp': self.detector._utcnow().isoformat()
        }

    async def get_zone_analysis(self, zone_id: str) -> Dict[str, Any]:
        return await self.detector.analyze_crowd_density(zone_id)

    async def trigger_emergency_analysis(self, tourist_id: str) -> Dict[str, Any]:
        raw = await self.detector.redis_client.get(f'tourist:{tourist_id}:current')
        if not raw:
            return {'error': 'Tourist not found'}
        tourist_data = json.loads(raw)
        tourist_data['panic_button_pressed'] = 1
        risk_score = self.detector.predict_risk_score(tourist_data)
        location_anomaly = await self.detector.detect_location_anomaly(tourist_id, tourist_data.get('location', {}))
        alert_data = {
            'tourist_id': tourist_id,
            'anomaly_type': 'panic_signal',
            'risk_level': 'high',
            'recommendations': [
                'Dispatch immediate help',
                'Notify emergency contacts',
                'Track location continuously'
            ]
        }
        alert = await self.detector.generate_alert(alert_data)
        return {
            'tourist_id': tourist_id,
            'emergency_alert': alert,
            'risk_score': risk_score,
            'immediate_actions': alert['auto_response']
        }


# -----------------
# Demo data generation for testing
# -----------------
class DemoDataGenerator:
    """Generate realistic demo data for testing the anomaly detection system."""
    
    def __init__(self):
        self.demo_zones = {
            'zone_001': {'name': 'Gateway of India', 'capacity': 200, 'risk_level': 3},
            'zone_002': {'name': 'Marine Drive', 'capacity': 150, 'risk_level': 2},
            'zone_003': {'name': 'Colaba Market', 'capacity': 100, 'risk_level': 5},
            'zone_004': {'name': 'Taj Hotel Area', 'capacity': 80, 'risk_level': 7}
        }
        
        self.demo_tourists = {
            'tourist_001': {'name': 'John Smith', 'nationality': 'USA', 'group_size': 2},
            'tourist_002': {'name': 'Maria Garcia', 'nationality': 'Spain', 'group_size': 1},
            'tourist_003': {'name': 'Chen Wei', 'nationality': 'China', 'group_size': 4},
            'tourist_004': {'name': 'Sarah Johnson', 'nationality': 'UK', 'group_size': 1},
            'tourist_005': {'name': 'Raj Patel', 'nationality': 'India', 'group_size': 3}
        }
    
    def generate_normal_tourist_data(self, tourist_id: str) -> Dict[str, Any]:
        """Generate normal tourist behavior data."""
        base_locations = {
            'tourist_001': {'lat': 18.9220, 'lon': 72.8347},  # Gateway of India
            'tourist_002': {'lat': 18.9400, 'lon': 72.8234},  # Marine Drive
            'tourist_003': {'lat': 18.9167, 'lon': 72.8333},  # Colaba
            'tourist_004': {'lat': 18.9220, 'lon': 72.8347},  # Gateway of India
            'tourist_005': {'lat': 18.9400, 'lon': 72.8234}   # Marine Drive
        }
        
        base_loc = base_locations.get(tourist_id, {'lat': 18.9220, 'lon': 72.8347})
        
        return {
            'tourist_id': tourist_id,
            'location': {
                'lat': float(base_loc['lat'] + np.random.normal(0, 0.001)),
                'lon': float(base_loc['lon'] + np.random.normal(0, 0.001)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'location_history': self.generate_location_history(base_loc),
            'time_since_last_update': int(np.random.randint(1, 300)),
            'panic_button_pressed': 0,
            'current_zone_risk': int(np.random.randint(1, 4)),
            'restricted_zone_entry': 0,
            'emergency_contacts_notified': 0,
            'sos_signals': 0,
            'heart_rate': float(np.random.normal(75, 10)),
            'body_temperature': float(np.random.normal(36.5, 0.3))
        }
    
    def generate_anomaly_tourist_data(self, tourist_id: str, anomaly_type: str) -> Dict[str, Any]:
        """Generate tourist data with specific anomalies."""
        data = self.generate_normal_tourist_data(tourist_id)
        
        if anomaly_type == 'panic':
            data['panic_button_pressed'] = 1
            data['heart_rate'] = float(np.random.normal(120, 15))
            data['sos_signals'] = 1
            
        elif anomaly_type == 'missing':
            data['time_since_last_update'] = int(np.random.randint(7200, 14400))  # 2-4 hours
            data['location']['timestamp'] = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
            
        elif anomaly_type == 'restricted_zone':
            data['restricted_zone_entry'] = 1
            data['current_zone_risk'] = 8
            data['location']['lat'] = 18.9100  # Restricted area coordinates
            data['location']['lon'] = 72.8200
            
        elif anomaly_type == 'health_emergency':
            data['heart_rate'] = float(np.random.choice([45, 150]))  # Very low or very high
            data['body_temperature'] = float(np.random.choice([35.0, 39.5]))  # Hypothermia or fever
            
        elif anomaly_type == 'unusual_movement':
            # Generate erratic movement pattern
            erratic_history = []
            base_lat, base_lon = 18.9220, 72.8347
            for i in range(10):
                erratic_history.append({
                    'lat': base_lat + np.random.normal(0, 0.01),  # Large random movements
                    'lon': base_lon + np.random.normal(0, 0.01),
                    'timestamp': (datetime.now(timezone.utc) - timedelta(minutes=i*5)).isoformat()
                })
            data['location_history'] = erratic_history
            
        return data
    
    def generate_location_history(self, base_loc: Dict[str, float], hours: int = 2) -> List[Dict[str, Any]]:
        """Generate realistic location history."""
        history = []
        current_time = datetime.now(timezone.utc)
        
        for i in range(hours * 6):  # Every 10 minutes
            timestamp = current_time - timedelta(minutes=i*10)
            # Small random walk from base location
            lat_offset = float(np.random.normal(0, 0.0005) * (i + 1) * 0.1)
            lon_offset = float(np.random.normal(0, 0.0005) * (i + 1) * 0.1)
            
            history.append({
                'lat': float(base_loc['lat'] + lat_offset),
                'lon': float(base_loc['lon'] + lon_offset),
                'timestamp': timestamp.isoformat()
            })
        
        return sorted(history, key=lambda x: x['timestamp'])

async def demo_mode():
    """Run the system in demonstration mode with sample data."""
    print("🚀 Starting Tourist Safety AI - Demo Mode")
    print("=" * 60)
    
    detector = TouristAnomalyDetector()
    demo_gen = DemoDataGenerator()
    
    # Setup demo data in Redis
    try:
        # Add demo tourists as active
        for tourist_id in demo_gen.demo_tourists.keys():
            await detector.redis_client.sadd('active_tourists', tourist_id)
        
        # Add demo zones
        for zone_id, zone_data in demo_gen.demo_zones.items():
            await detector.redis_client.sadd('active_zones', zone_id)
            await detector.redis_client.set(f'zone:{zone_id}:capacity', str(zone_data['capacity']))
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("Running in offline demo mode...")
    
    print("\n📊 Initializing AI Models...")
    print("✅ Isolation Forest for anomaly detection")
    print("✅ LSTM Neural Network for sequence analysis")
    print("✅ Random Forest for risk classification")
    print("✅ DBSCAN for crowd analysis")
    
    print("\n🎯 Demo Scenarios:")
    scenarios = [
        ("Normal Tourist Behavior", "normal"),
        ("Emergency - Panic Button Pressed", "panic"),
        ("Missing Person Alert", "missing"),
        ("Restricted Zone Entry", "restricted_zone"),
        ("Health Emergency Detected", "health_emergency"),
        ("Unusual Movement Pattern", "unusual_movement")
    ]
    
    for i, (scenario_name, scenario_type) in enumerate(scenarios, 1):
        print(f"\n🔍 Scenario {i}: {scenario_name}")
        print("-" * 40)
        
        tourist_id = f"tourist_{i:03d}"
        
        if scenario_type == "normal":
            tourist_data = demo_gen.generate_normal_tourist_data(tourist_id)
        else:
            tourist_data = demo_gen.generate_anomaly_tourist_data(tourist_id, scenario_type)
        
        # Analyze the tourist data
        try:
            # Store demo data in Redis if available
            await detector.redis_client.set(f'tourist:{tourist_id}:current', json.dumps(tourist_data))
        except:
            pass  # Continue without Redis
        
        # Run anomaly detection
        location_anomaly = await detector.detect_location_anomaly(tourist_id, tourist_data.get('location', {}))
        missing_check = await detector.detect_missing_person_pattern(tourist_id)
        risk_score = detector.predict_risk_score(tourist_data)
        
        # Display results
        print(f"👤 Tourist: {tourist_id}")
        print(f"📍 Location: {tourist_data['location']['lat']:.4f}, {tourist_data['location']['lon']:.4f}")
        print(f"⚡ Risk Score: {risk_score:.1f}/100")
        
        if location_anomaly['anomaly_detected']:
            print(f"🚨 ANOMALY DETECTED: {location_anomaly['anomaly_type']}")
            print(f"📈 Risk Level: {location_anomaly['risk_level'].upper()}")
            print(f"💡 Recommendations: {', '.join(location_anomaly['recommendations'])}")
            
            # Generate alert
            alert = await detector.generate_alert(location_anomaly)
            print(f"🔔 Alert Generated: {alert['id']}")
        else:
            print("✅ Normal behavior detected")
        
        if missing_check['missing_probability'] > 20:
            print(f"🔍 Missing Person Probability: {missing_check['missing_probability']}%")
            if missing_check['indicators']:
                print(f"⚠️ Indicators: {', '.join(missing_check['indicators'])}")
        
        await asyncio.sleep(2)  # Pause between scenarios
    
    print("\n🏙️ Crowd Analysis Demo:")
    print("-" * 40)
    
    for zone_id, zone_data in demo_gen.demo_zones.items():
        # Simulate tourists in zone
        tourist_count = np.random.randint(zone_data['capacity'] // 4, zone_data['capacity'] + 20)
        try:
            for i in range(tourist_count):
                await detector.redis_client.sadd(f'zone:{zone_id}:tourists', f'demo_tourist_{i}')
        except:
            pass
        
        density_analysis = await detector.analyze_crowd_density(zone_id)
        
        print(f"🏛️ {zone_data['name']} (Zone {zone_id})")
        print(f"   👥 Tourist Count: {density_analysis['tourist_count']}")
        print(f"   📊 Density Level: {density_analysis['density_level'].upper()}")
        if density_analysis['stampede_risk'] > 0:
            print(f"   ⚠️ Stampede Risk: {density_analysis['stampede_risk']}%")
        if density_analysis['recommendations']:
            print(f"   💡 Actions: {', '.join(density_analysis['recommendations'])}")
        print()
    
    print("🎉 Demo completed! The AI system continuously monitors:")
    print("   • Tourist location patterns")
    print("   • Emergency signals and panic buttons") 
    print("   • Missing person detection")
    print("   • Crowd density and stampede risks")
    print("   • Health emergency detection")
    print("   • Restricted zone violations")
    
    return detector

# -----------------
# If run as script, start monitoring
# -----------------
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run demo mode
        try:
            asyncio.run(demo_mode())
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
    else:
        # Run continuous monitoring
        detector = TouristAnomalyDetector()
        try:
            asyncio.run(detector.continuous_monitoring())
        except KeyboardInterrupt:
            logger.info('Shutting down continuous monitoring')
