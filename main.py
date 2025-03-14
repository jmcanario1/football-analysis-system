from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    
    # Read video
    # video_frames = read_video('input_videos/08fd33_4.mp4')
    # video_frames = read_video('input_videos/gol_jk.mp4')
    video_frames = read_video('input_videos/fla_marica_pt1.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    # tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs_gol_jk.pkl")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs_fla_marica.pkl")
    
    for track_id, player in tracks['players'][0].items():
        bbox = player
        frame = video_frames[0]

        cropped_image = frame[bbox['bbox'][1]:bbox['bbox'][3], bbox['bbox'][0]:bbox['bbox'][2]]

        cv2.imwrite(f"output_videos/player_{track_id}.jpg", cropped_image)


    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            


    # Draw Output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/08fd33_4.avi')
    # save_video(output_video_frames, 'output_videos/gol_jk.avi')

if __name__ == '__main__':
    main()