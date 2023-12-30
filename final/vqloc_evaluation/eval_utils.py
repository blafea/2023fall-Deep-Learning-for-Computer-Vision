import warnings
import numpy as np
from utils.loss_utils import GiouLoss


def convert_annotations_to_clipwise_list(annotations):
    clipwise_annotations_list = {}
    for cuid, clip_data in annotations.items():
        for a in clip_data["annotations"]:
            aid = a["annotation_uid"]
            for qid, q in a["query_sets"].items():
                if not q["is_valid"]:
                    continue
                curr_q = {
                    "clip_uid": cuid,
                    "query_set": qid,
                    "annotation_uid": aid,
                    "query_frame": q["query_frame"],
                    "visual_crop": q["visual_crop"],
                }
                if "response_track" in q:
                    curr_q["response_track"] = q["response_track"]
                if cuid not in clipwise_annotations_list:
                    clipwise_annotations_list[cuid] = []
                clipwise_annotations_list[cuid].append(curr_q)
    return clipwise_annotations_list


def format_predictions(annotations, predicted_rts):
    # Format predictions
    predictions = {}
    for cuid, clip_data in annotations.items():
        clip_predictions = {"predictions": []}
        for a in clip_data["annotations"]:
            auid = a["annotation_uid"]
            apred = {
                "query_sets": {},
                "annotation_uid": auid,
            }
            for qid in a["query_sets"].keys():
                if (auid, qid) in predicted_rts:
                    rt_pred = predicted_rts[(auid, qid)][0].to_json()
                    apred["query_sets"][qid] = rt_pred
                else:
                    apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}
            clip_predictions["predictions"].append(apred)
        predictions[cuid] = clip_predictions
    return predictions
