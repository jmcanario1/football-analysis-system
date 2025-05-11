from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from event_analyzer import detectar_toques_de_primeira

def main():
    
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks, possession_per_frame = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    #Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])


    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (255, 255, 255))

            
    toques = detectar_toques_de_primeira(possession_per_frame)

    for toque in toques:
        print(f"Toque de primeira detectado: Jogador {toque['jogador_id']} de {toque['start']} a {toque['end']} → próximo jogador: {toque['proximo_jogador']}")

    # Draw Output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, possession_per_frame, toques)

    # Save video
    save_video(output_video_frames, 'output_videos/posse-de-bola-V5.avi')

if __name__ == '__main__':
    main()