import numpy as np

def get_ball_possession(player_dict, ball_bbox, max_distance=60):
    if not ball_bbox:
        return None

    bx = (ball_bbox[0] + ball_bbox[2]) / 2
    by = (ball_bbox[1] + ball_bbox[3]) / 2

    min_dist = float('inf')
    possessor_id = None

    for player_id, player_data in player_dict.items():
        px = (player_data["bbox"][0] + player_data["bbox"][2]) / 2
        py = (player_data["bbox"][1] + player_data["bbox"][3]) / 2
        dist = np.sqrt((bx - px)**2 + (by - py)**2)

        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            possessor_id = player_id

    return possessor_id
