
from collections import defaultdict, deque

import cv2
import numpy as np
from trackers import ByteTrackTracker
from ultralytics import YOLO
import supervision as sv

from accident_risk import RISK_LEVELS, assess_frame



# CONFIGURATION
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1]
])


MODEL_NAME = "yolo11n.pt"

# Résolution utilisée par YOLO
IMAGE_SIZE = 640

# Couleurs par niveau de risque : faible, moyen, élevé
RISK_PALETTE = sv.ColorPalette.from_hex(["#2ecc71", "#f39c12", "#e74c3c"])

# Délai minimum entre deux alertes console pour la même paire (secondes)
ALERT_COOLDOWN_S = 1.0



# TRANSFORMATION PERSPECTIVE
class ViewTransformer:

    def __init__(self, source, target):

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):

        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)

        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.m
        )

        return transformed_points.reshape(-1, 2)



# PROGRAMME PRINCIPAL
def main(
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
):

    print("Chargement de la vidéo...")

    video_info = sv.VideoInfo.from_video_path(
        video_path=source_video_path
    )

    print("Résolution :", video_info.resolution_wh)
    print("FPS :", video_info.fps)

    # Chargement du modèle léger
    print("Chargement du modèle YOLO11n...")

    model = YOLO(MODEL_NAME)

    # ByteTrack
    byte_track = ByteTrackTracker(
        frame_rate=video_info.fps,
        track_activation_threshold=confidence_threshold
    )

    # Annotations
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )

    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
    )

    box_annotator = sv.BoxAnnotator(
        color=RISK_PALETTE,
        thickness=thickness
    )

    label_annotator = sv.LabelAnnotator(
        color=RISK_PALETTE,
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER
    )

    # Zone de détection
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path
    )

    polygon_zone = sv.PolygonZone(
        polygon=SOURCE
    )

    view_transformer = ViewTransformer(
        source=SOURCE,
        target=TARGET
    )

    # Historique des positions (frame, x, y) sur ~1 s
    coordinates = defaultdict(
        lambda: deque(maxlen=int(video_info.fps))
    )

    # Dernière alerte console par paire (anti-répétition)
    last_alert = {}

    # Enregistrement vidéo
    with sv.VideoSink(
        target_video_path,
        video_info
    ) as sink:

        window = sv.ImageWindow("Vehicle Speed Estimation")

        print("Début du traitement...")

        for frame_index, frame in enumerate(frame_generator):

            # Détection YOLO avec résolution réduite
            result = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]

            # Conversion des détections
            detections = sv.Detections.from_ultralytics(result)

            # Filtrage des véhicules dans la zone
            # (hors zone calibrée, les mètres sont faux et créeraient de fausses alertes)
            detections = detections[
                polygon_zone.trigger(detections)
            ]

            # Suivi des véhicules
            detections = byte_track.update(detections)

            detections = detections[
                detections.tracker_id != -1
            ]

            # Position du centre inférieur des véhicules
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )

            # (pas de .astype(int) : arrondir au mètre fausse vitesses et TTC)
            points = view_transformer.transform_points(
                points=points
            )

            # Enregistrer les positions (mètres, vue de dessus)
            for tracker_id, (x, y) in zip(
                detections.tracker_id,
                points,
                strict=True
            ):

                coordinates[tracker_id].append(
                    (frame_index, x, y)
                )

            # Cinématique, TTC, réseau bayésien -> risque par véhicule
            kin, vehicle_risk, pairs = assess_frame(
                {tid: coordinates[tid] for tid in detections.tracker_id},
                video_info.fps
            )

            labels = []
            risk_index = []

            for tracker_id in detections.tracker_id:

                level, probability = vehicle_risk.get(tracker_id, (0, 0.0))
                risk_index.append(level)

                if tracker_id not in kin:
                    labels.append(f"#{tracker_id}")
                    continue

                labels.append(
                    f"#{tracker_id} {int(kin[tracker_id]['speed_kmh'])} km/h"
                    f" P={probability:.2f}"
                )

            risk_index = np.array(risk_index, dtype=int)

            # Alertes : paires à risque élevé
            danger = [p for p in pairs if p["level"] == 2]

            for pair in danger:

                ids = pair["ids"]

                if frame_index - last_alert.get(ids, -np.inf) < ALERT_COOLDOWN_S * video_info.fps:
                    continue

                last_alert[ids] = frame_index

                print(
                    f"[ALERTE] t={frame_index / video_info.fps:.1f}s"
                    f" frame={frame_index} #{ids[0]}<->#{ids[1]}"
                    f" P={pair['probability']:.2f}"
                    f" TTC={pair['ttc']:.1f}s"
                    f" headway={pair['headway']:.1f}s"
                    f" risque={RISK_LEVELS[pair['level']]}"
                )

            # Dessiner les annotations
            annotated_frame = frame.copy()

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                custom_color_lookup=risk_index
            )

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
                custom_color_lookup=risk_index
            )

            # Ligne rouge entre véhicules en danger + bandeau d'alerte
            anchors = dict(zip(
                detections.tracker_id,
                detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            ))

            for pair in danger:
                a, b = (anchors[i] for i in pair["ids"])
                annotated_frame = sv.draw_line(
                    annotated_frame,
                    sv.Point(*a), sv.Point(*b),
                    color=RISK_PALETTE.by_idx(2),
                    thickness=thickness * 2
                )

            if danger:
                worst = danger[0]
                # ponytail: texte ASCII, la police Hershey d'OpenCV ne gère pas les accents
                annotated_frame = sv.draw_text(
                    annotated_frame,
                    f"ALERTE: risque eleve #{worst['ids'][0]}-#{worst['ids'][1]}"
                    f" P={worst['probability']:.2f} TTC={worst['ttc']:.1f}s",
                    text_anchor=sv.Point(video_info.resolution_wh[0] // 2, int(80 * text_scale)),
                    text_color=sv.Color.WHITE,
                    text_scale=text_scale * 1.5,
                    text_thickness=thickness,
                    background_color=RISK_PALETTE.by_idx(2)
                )

            # Sauvegarder la vidéo
            sink.write_frame(annotated_frame)

            # Afficher la vidéo
            window.show(annotated_frame)

            key = window.wait_key(1)

            if not window.is_open or key == "q":
                break

            # Afficher la progression
            if frame_index % 100 == 0:

                print(
                    f"Frames traitées : {frame_index}"
                )

        window.close()

    print("Traitement terminé !")
    print("Vidéo enregistrée :", target_video_path)



# EXECUTION
if __name__ == "__main__":

    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(
        parse_optionals_as_positionals=True
    )

    auto_cli(
        main,
        as_positional=False
    )
