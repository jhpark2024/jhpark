# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import numpy as np
import json

import torch
import torchvision.transforms as T
from models import build_model
from sentence_transformers import SentenceTransformer
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='example/2407349.jpg',
                        help="Path of the test image")

    parser.add_argument('--result_path', type=str, default='example_result/2407349.json',
                        help="Path of the Extracted Visual Semantics")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='pretrained/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser

def main(args):

    with open('data/rel_count.json', 'r') as file:
        rel_count = json.load(file)

    LM = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return b

    def add_object_layout(obj_class, obj_layout, threshold=0.8):
        def is_too_similar(layout1, layout2, threshold):
            overlap_x = max(0, min(layout1[2], layout2[2]) - max(layout1[0], layout2[0]))
            overlap_y = max(0, min(layout1[3], layout2[3]) - max(layout1[1], layout2[1]))
            overlap_area = overlap_x * overlap_y

            area1 = (layout1[2] - layout1[0]) * (layout1[3] - layout1[1])
            area2 = (layout2[2] - layout2[0]) * (layout2[3] - layout2[1])

            return (overlap_area / min(area1, area2)) > threshold

        if obj_class not in Visual_Semantics['Objects_Layouts']:
            Visual_Semantics['Objects_Layouts'][obj_class] = [obj_layout]
            return 0
        else:
            index = 0
            for existing_layout in Visual_Semantics['Objects_Layouts'][obj_class]:
                if is_too_similar(existing_layout, obj_layout, threshold):
                    return index
                index += 1
            Visual_Semantics['Objects_Layouts'][obj_class].append(obj_layout)
            return len(Visual_Semantics['Objects_Layouts'][obj_class]) - 1

    def get_prob_subgraph(rel_count, sub_graph):
        parts = sub_graph.split(' ')
        sub = ''.join([i for i in parts[0] if not i.isdigit()])
        obj = ''.join([i for i in parts[-1] if not i.isdigit()])
        rel = ' '.join(parts[1:-1])
        total_count = 1
        for relation in REL_CLASSES:
            total_count += rel_count[sub][relation][obj]
        return rel_count[sub][rel][obj] / total_count

    def filter_less_informative_relation(Visual_Semantics, threshold_f = 0.8): # threshold_f 0.7 ~ 0.9
        for sub_graph in Visual_Semantics['Full_SceneGraph']:
            if get_prob_subgraph(rel_count, sub_graph) < threshold_f:
                Visual_Semantics['Filtered_SceneGraph(A)'].append(sub_graph)

    def get_residual_norms(embeddings):
        num_embeddings = len(embeddings)
        residual_norms = []
        for ne in range(num_embeddings):
            embeddings = np.vstack((embeddings[1:], embeddings[:1]))
            embeddings = np.array(embeddings)
            n, m = embeddings.shape
            g_lm = np.zeros((n, m))
            for i in range(n):
                g_i = embeddings[i, :].copy()
                for j in range(i):
                    g_j = g_lm[j, :].copy()
                    g_i -= np.dot(g_i, g_j) * g_j
                if i != n - 1:
                    g_lm[i, :] = g_i
                else:
                    residual_norms.append(np.linalg.norm(g_i))
        return residual_norms

    def filter_redundant_subgraph(LM, Visual_Semantics, threshold_r = 0.8): # threshold_r 0.6 ~ 0.9
        G_filtered = Visual_Semantics['Filtered_SceneGraph(A)'].copy()
        for s in range(len(G_filtered)):
            G_filtered[s] = ''.join([i for i in G_filtered[s] if not i.isdigit()])
        G_filtered = list(set(G_filtered))
        while True:
            embedding = LM.encode(G_filtered)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            G_LM = embedding / norms
            residual_norms = get_residual_norms(G_LM)
            if min(residual_norms) < threshold_r and len(G_filtered) > 2:
                G_filtered.pop(residual_norms.index(min(residual_norms)))
            else:
                break
        for subgraph in Visual_Semantics['Filtered_SceneGraph(A)']:
            if ''.join([i for i in subgraph if not i.isdigit()]) in G_filtered:
                Visual_Semantics['Filtered_SceneGraph(B)'].append(subgraph)

    def get_filtered_scenegraph(Visual_Semantics):
        Visual_Semantics['Filtered_SceneGraph'] = Visual_Semantics['Filtered_SceneGraph(B)'].copy()
        obj_exist_a = []
        obj_exist_b = []
        for subgraph in Visual_Semantics['Filtered_SceneGraph(A)']:
            parts = subgraph.split(' ')
            obj_exist_a.append(parts[0])
            obj_exist_a.append(parts[-1])
        for subgraph in Visual_Semantics['Filtered_SceneGraph(B)']:
            parts = subgraph.split(' ')
            obj_exist_b.append(parts[0])
            obj_exist_b.append(parts[-1])
        a_sub_b = [x for x in obj_exist_a if x not in obj_exist_b]
        for obj_c in Visual_Semantics['Objects_Layouts']:
            for obj_i in range(len(Visual_Semantics['Objects_Layouts'][obj_c])):
                obj = obj_c+str(obj_i)
                if obj not in obj_exist_b and obj not in a_sub_b:
                    Visual_Semantics['Filtered_SceneGraph'].append(obj)
        del Visual_Semantics['Filtered_SceneGraph(A)']
        del Visual_Semantics['Filtered_SceneGraph(B)']
    # VG classes
    CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.eval()

    img_path = args.img_path
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])
    keep_queries = keep_queries[indices]

    Visual_Semantics = {'Objects_Layouts': {}, 'Full_SceneGraph': [], 'Filtered_SceneGraph(A)': [], 'Filtered_SceneGraph(B)': [], 'Filtered_SceneGraph': []}
    for query in keep_queries:
        query_index = query.item()
        # get objects(subjects) and layouts
        obj_class = CLASSES[probas_obj[query_index].argmax()]
        obj_layout = box_cxcywh_to_xyxy(outputs['obj_boxes'][0, query_index].tolist())
        obj_index = add_object_layout(obj_class, obj_layout)
        sub_class = CLASSES[probas_sub[query_index].argmax()]
        sub_layout = box_cxcywh_to_xyxy(outputs['sub_boxes'][0, query_index].tolist())
        sub_index = add_object_layout(sub_class, sub_layout)

        # get relation between objects
        rel_class = REL_CLASSES[probas[query_index].argmax()]
        Visual_Semantics['Full_SceneGraph'].append(sub_class + str(sub_index) + ' ' + rel_class + ' ' + obj_class + str(obj_index))


    # Scene Graph filtering 1: Less informative relation filtering
    filter_less_informative_relation(Visual_Semantics, threshold_f=0.8)

    # Scene Graph filtering 2: Redundant sub-graph filtering
    filter_redundant_subgraph(LM, Visual_Semantics, threshold_r=0.8)

    # Visualize Extracted Visual Semantics
    get_filtered_scenegraph(Visual_Semantics)

    with open(f'{args.result_path}', 'w') as file:
        json.dump(Visual_Semantics, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SGG', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
