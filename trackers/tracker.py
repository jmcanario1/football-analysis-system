from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import cv2
import sys
import pandas as pd

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_ball_possession

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks, possession_per_frame = pickle.load(f)
            print(f"[INFO] Tracks loaded from stub: {tracks}")  # Depuração aqui
            return tracks, []

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        possession_per_frame = []  # Lista para armazenar quem está com a bola em cada frame

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to Supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            # Agora, após processar o quadro, determinamos quem está com a bola
            ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
            if ball_bbox:
                possessor_id = get_ball_possession(tracks["players"][frame_num], ball_bbox)
            else:
                possessor_id = None

            possession_per_frame.append(possessor_id)

            # print(f"[INFO] Frame {frame_num} processado — Posse: {possessor_id}")


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump((tracks, possession_per_frame), f)

        return tracks, possession_per_frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_possession_visual(self, frame, bbox):
        # Desenha um círculo verde ao redor do jogador com a posse da bola
        x_center, y_center = get_center_of_bbox(bbox)
        cv2.circle(frame, (int(x_center), int(y_center)), 20, (0, 255, 0), 4)
        return frame

    def draw_annotations(self, video_frames, tracks, possession_per_frame, toques):
        output_video_frames = []

        # Mapeia os frames que têm toque de primeira
        toque_frames = set()
        toque_jogadores = {}

        for toque in toques:
            for f in range(toque['start'], toque['end'] + 1):
                toque_frames.add(f)
                toque_jogadores[f] = toque['jogador_id']

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players com cor de time
            for track_id, player in player_dict.items():
                team_color = tuple(map(int, player.get("team_color", [0, 0, 255])))  # vermelho se não definido
                frame = self.draw_ellipse(frame, player["bbox"], team_color, track_id)

            # Draw Referees (cor fixa amarela)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw Ball (cor fixa verde)
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Desenhar posse de bola
            possessor_id = possession_per_frame[frame_num]
            if possessor_id:
                player = player_dict.get(possessor_id)
                if player:
                    frame = self.draw_possession_visual(frame, player["bbox"])

            # Destacar toque de primeira
            if frame_num in toque_frames:
                jogador_id = toque_jogadores.get(frame_num)
                jogador = player_dict.get(jogador_id)
                if jogador:
                    x, y = get_center_of_bbox(jogador["bbox"])

                    # Texto flutuante
                    cv2.putText(
                        frame,
                        "TOQUE DE PRIMEIRA!",
                        (int(x - 100), int(y - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )

                    # Círculo especial vermelho
                    cv2.circle(frame, (int(x), int(y)), 25, (0, 0, 255), 5)

                    # Salva o frame como imagem
                    cv2.imwrite(f"output_frames/toque_{frame_num}.jpg", frame)

            output_video_frames.append(frame)

        return output_video_frames


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions