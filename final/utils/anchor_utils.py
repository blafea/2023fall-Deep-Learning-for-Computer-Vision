import torch
from einops import rearrange


def generate_anchor_boxes_on_regions(image_size, 
                                     num_regions, 
                                     base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32),
                                     aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32),
                                     dtype=torch.float32, 
                                     device='cpu'):
    """
    Generate a set of anchor boxes with different sizes and aspect ratios for each region of a split image.

    Arguments:
    image_size -- tuple of two integers, the height and width of the original image
    num_regions -- tuple of two integers, the number of regions in the height and width directions
    aspect_ratios -- torch.Tensor of shape [M], containing M aspect ratios for each base size
    dtype -- the data type of the output tensor
    device -- the device of the output tensor

    Returns:
    anchor_boxes -- torch.Tensor of shape [R^2*N*M,4], containing R^2*N*M anchor boxes represented as (center_h, center_w, box_h, box_w)
    """

    # Calculate the base sizes for each region
    region_size = (image_size[0] / num_regions[0], image_size[1] / num_regions[1])

    # Calculate the anchor boxes for each region
    anchor_boxes = torch.empty((0, 4), dtype=dtype, device=device)
    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            center_h = (i + 0.5) * region_size[0]
            center_w = (j + 0.5) * region_size[1]
            base_boxes = generate_anchor_boxes(base_sizes, aspect_ratios, dtype=dtype, device=device)
            base_boxes[:, 0] += center_h
            base_boxes[:, 1] += center_w
            anchor_boxes = torch.cat([anchor_boxes, base_boxes], dim=0)

    return anchor_boxes


def generate_anchor_boxes(base_sizes, aspect_ratios, dtype=torch.float32, device='cpu'):
    """
    Generate a set of anchor boxes with different sizes and aspect ratios.

    Arguments:
    base_sizes -- torch.Tensor of shape [N,2], containing N base sizes for the anchor boxes
    aspect_ratios -- torch.Tensor of shape [M], containing M aspect ratios for each base size
    dtype -- the data type of the output tensor
    device -- the device of the output tensor

    Returns:
    anchor_boxes -- torch.Tensor of shape [N*M,4], containing N*M anchor boxes represented as (center_h, center_w, box_h, box_w)
    """

    num_base_sizes = base_sizes.shape[0]
    num_aspect_ratios = aspect_ratios.shape[0]

    # Generate base anchor boxes
    base_boxes = torch.zeros((num_base_sizes * num_aspect_ratios, 4), dtype=dtype, device=device)
    for i in range(num_base_sizes):
        for j in range(num_aspect_ratios):
            w = torch.sqrt(base_sizes[i, 0] * base_sizes[i, 1] / aspect_ratios[j])
            h = aspect_ratios[j] * w
            idx = i * num_aspect_ratios + j
            base_boxes[idx] = torch.tensor([0, 0, h, w], dtype=dtype, device=device)

    return base_boxes


# def assign_labels(proposals, gt_boxes, iou_threshold=0.5):
#     """
#     Assign labels to a set of bounding box proposals based on their IoU with ground truth boxes.

#     Arguments:
#     proposals -- torch.Tensor of shape [B,T,N,4], representing the bounding box proposals for each frame in each clip
#     gt_boxes -- torch.Tensor of shape [B,T,4], representing the ground truth boxes for each frame in each clip
#     iou_threshold -- float, the IoU threshold for a proposal to be considered a positive match with a ground truth box

#     Returns:
#     labels -- torch.Tensor of shape [B,T,N], containing the assigned labels for each proposal (0 for background, 1 for object)
#     """

#     # Initialize the labels tensor with background labels
#     labels = torch.zeros_like(proposals[:, :, :, 0], dtype=torch.long, device=proposals.device)

#     # Loop over the batches and frames
#     for b in range(proposals.shape[0]):
#         for t in range(proposals.shape[1]):
#             # Calculate the IoU between each proposal and the ground truth box
#             iou = calculate_iou(proposals[b, t], gt_boxes[b, t])    # [N]

#             # Assign labels to the proposals based on their IoU with the ground truth box
#             labels[b, t] = iou > iou_threshold

#     return labels


def max_with_non_max_zero(tensor, dim):
    # Find the maximum values and their indices along the specified dimension
    max_values, max_indices = tensor.max(dim, keepdim=True)
    
    # Create a mask of the same shape as the tensor, 
    # where the positions of the max values are True and all others are False
    max_mask = torch.eq(tensor, max_values)
    
    # Use the mask to set non-max values to 0
    result = torch.where(max_mask, tensor, torch.zeros_like(tensor))
    
    return result


def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    """Implementation of paper `A Normalized Gaussian Wasserstein Distance for
    Tiny Object Detection <https://arxiv.org/abs/2110.13389>.
    Args:
        pred (Tensor): Predicted bboxes of format (cx, cy, w, h),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    center1 = pred[:, :2]
    center2 = target[:, :2]
    whs = center1[:, :2] - center2[:, :2]
    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps
 
    w1 = pred[:, 2] + eps
    h1 = pred[:, 3] + eps
    w2 = target[:, 2] + eps
    h2 = target[:, 3] + eps
 
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


def assign_labels(anchors, gt_boxes, iou_threshold=0.5, topk=5, nwd_threshold=None):
    """
    Assign labels to a set of bounding box proposals based on their IoU with ground truth boxes.

    Arguments:
    anchors -- torch.Tensor of shape [B,T,N,4], representing the bounding box proposals for each frame in each clip
    gt_boxes -- torch.Tensor of shape [B,T,4], representing the ground truth boxes for each frame in each clip
    iou_threshold -- float, the IoU threshold for a proposal to be considered a positive match with a ground truth box

    Returns:
    labels -- torch.Tensor of shape [B,T,N], containing the assigned labels for each proposal (0 for background, 1 for object)
    """
    anchors = anchors.detach()
    gt_boxes = gt_boxes.detach()

    b,t = gt_boxes.shape[:2]    #[B,T,N,4]
    n = anchors.shape[2]

    if nwd_threshold:
        tp, tn = nwd_threshold
        # Calculate the NWD between each proposal and the ground truth box
        nwd = calculate_nwd(rearrange(anchors, 'b t n c -> (b t n) c', c=4),
                            rearrange(
                                gt_boxes.unsqueeze(2).repeat(1, 1, n, 1), 
                                'b t n c -> (b t n) c', c=4))
        nwd = rearrange(nwd, '(b t n) -> b t n', b=b, t=t)    # [B,T,N]
        nwd_max = max_with_non_max_zero(nwd, dim=-1) # [B,T,N]
        # Assign labels to the proposals based on their NWD with the ground truth box
        labels = torch.logical_or(nwd > tp, nwd_max > tn)
    else:
        # Calculate the IoU between each proposal and the ground truth box
        iou = calculate_iou(anchors.view(-1, anchors.shape[-2], anchors.shape[-1]),   # [B*T,N,4]
                           gt_boxes.view(-1, gt_boxes.shape[-1]))                    # [B*T,4] -> [B*T,N]
        iou = iou.view(anchors.shape[:-1])    # [B,T,N]

        # Assign labels to the proposals based on their IoU with the ground truth box
        labels = iou > iou_threshold

    if not labels.any():
        labels = process_labels(labels, iou, topk)

    return labels


def calculate_nwd(boxes1, boxes2):
    """
    Calculate the IoU between two sets of bounding boxes.

    Arguments:
    boxes1 -- torch.Tensor of shape [N,4], containing N bounding boxes represented as [x1, y1, x2, y2]
    boxes2 -- torch.Tensor of shape [N,4], containing a single ground truth box represented as [x1, y1, x2, y2]

    Returns:
    iou -- torch.Tensor of shape [...,N], containing the IoU between each box and the ground truth box
    """

    # Compute the coordinates of the top-left and bottom-right corners of the boxes
    boxes1_cc = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    boxes2_cc = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    boxes1_wh = boxes1[..., 2:] - boxes1[..., :2]
    boxes2_wh = boxes2[..., 2:] - boxes2[..., :2]

    # Compute the NWD between each box and the ground truth box
    nwd = wasserstein_loss(torch.cat([boxes1_cc, boxes1_wh], dim=-1), 
                           torch.cat([boxes2_cc, boxes2_wh], dim=-1))
    return nwd



def calculate_iou(boxes1, boxes2):
    """
    Calculate the IoU between two sets of bounding boxes.

    Arguments:
    boxes1 -- torch.Tensor of shape [...,N,4], containing N bounding boxes represented as [x1, y1, x2, y2]
    boxes2 -- torch.Tensor of shape [...,4], containing a single ground truth box represented as [x1, y1, x2, y2]

    Returns:
    iou -- torch.Tensor of shape [...,N], containing the IoU between each box and the ground truth box
    """

    # Add a new dimension to boxes2 for broadcasting
    boxes2 = boxes2.unsqueeze(-2)    # shape: [...,1,4]

    # Compute the coordinates of the top-left and bottom-right corners of the boxes
    boxes1_tl = boxes1[..., :2]
    boxes1_br = boxes1[..., 2:]
    boxes2_tl = boxes2[..., :2]
    boxes2_br = boxes2[..., 2:]

    # Compute the coordinates of the intersection rectangle
    tl = torch.max(boxes1_tl, boxes2_tl)
    br = torch.min(boxes1_br, boxes2_br)

    # Compute the width and height of the intersection rectangle
    wh = br - tl
    wh[wh < 0] = 0

    # Compute the area of the intersection and union rectangles
    intersection_area = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - intersection_area

    # Compute the IoU between each box and the ground truth box
    iou = intersection_area / union_area

    return iou


def process_labels(labels, iou, topk=10):
    '''
    labels: in shape [B,T,N], bool
    iou: in shape [B,T,N]
    '''
    B,T,N = labels.shape

    labels = rearrange(labels, 'b t n -> (b t n)')
    iou = rearrange(iou, 'b t n -> (b t n)')

    if not labels.any():
        # no pos assigned, choose topk anchors with largest iou as positives
        _, topk_indices = torch.topk(iou, k=topk)
        labels[topk_indices] = True
    
    labels = rearrange(labels, '(b t n) -> b t n', b=B, t=T, n=N)
    return labels

if __name__ == '__main__':
    # Test script for generate_anchor_boxes_on_regions()
    anchor = generate_anchor_boxes_on_regions((448, 448), (16, 16))
    anchor = rearrange(anchor, '(h w n m) c -> h w n m c', h=16, w=16, n=4, m=3, c=4)
    print(anchor.shape)
    x, y = -1, -1
    for i in range(4):
        for j in range(3):
            print(anchor[x][y][i][j])
        print('===')