import json
import numpy as np

with open("data/train_anno_new.json", "r") as f:
    data_list = json.load(f)


dist_list = []
for data in data_list:
    q_frame = data["query_frame"]
    max_dist = 0
    for frame in data["response_track"]:
        dist = max(max_dist, q_frame - frame["frame_number"])
    dist_list.append(dist)
print(dist_list, max(dist_list), np.mean(dist_list), np.std(dist_list))
