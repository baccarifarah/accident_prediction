# Prédiction d'accidents routiers

Analyse de vidéos de trafic : détection des véhicules (YOLO11n), suivi (ByteTrack), vitesse et trajectoire en mètres, puis estimation du risque d'accident par réseau bayésien (pgmpy), classé en faible, moyen ou élevé avec alertes.

Documentation complète : [`docs/index.html`](docs/index.html) (à ouvrir dans un navigateur).

```bash
cd examples/speed_estimation
python3 -m venv venv && source venv/bin/activate
pip install -e ../.. ultralytics trackers pgmpy jsonargparse
python video_downloader.py
python ultralytics_example.py --source_video_path data/vehicles.mp4 --target_video_path data/vehicles-risk.mp4
```

Code du projet : `examples/speed_estimation/` (`accident_risk.py` pour le module de risque). `src/supervision/` est la bibliothèque Roboflow Supervision (MIT).
