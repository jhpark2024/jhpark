import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
def get_args_parser():
    parser = argparse.ArgumentParser('Visualization', add_help=False)
    parser.add_argument('--img_path', type=str, default='example/2407349.jpg',
                        help="Path of the test image")
    parser.add_argument('--result_path', type=str, default='example_result/2407349.json',
                        help="Path of the Extracted Visual Semantics")
    return parser

def visualize_objects_layouts(visual_semantics, im):
    width, height = im.size
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    color_idx = 0

    for obj_class, layouts in visual_semantics['Objects_Layouts'].items():
        for layout in layouts:
            x_min, y_min, x_max, y_max = layout
            rect_x_min = x_min * width
            rect_y_min = y_min * height
            rect_width = (x_max - x_min) * width
            rect_height = (y_max - y_min) * height

            rect = patches.Rectangle((rect_x_min, rect_y_min), rect_width, rect_height, linewidth=2, edgecolor=colors[color_idx % len(colors)], facecolor='none', label=obj_class if color_idx < len(colors) else "")
            ax.add_patch(rect)

        color_idx += 1

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('X Pixels')
    plt.ylabel('Y Pixels')
    plt.title('Object + Layout Visualization')
    plt.show()

def visualize_scene_graph(scene_graph, vis_rel):
    G = nx.DiGraph()

    for sub_graph in scene_graph:
        parts = sub_graph.split(' ')
        if len(parts) > 2:
            sub = parts[0]
            obj = parts[-1]
            rel = ' '.join(parts[1:-1])
            G.add_node(sub, subset=0)
            G.add_node(obj, subset=1)
            G.add_edge(sub, obj, label=rel)
        else:
            G.add_node(parts[0], subset=0)
    pos = nx.shell_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold',
            arrows=True, connectionstyle='arc3,rad=0.2')
    if vis_rel:
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.2)
    plt.title('Scene Graph Visualization')
    plt.show()

def main(args):
    with open(f'{args.result_path}', 'r') as file:
        Visual_Semantics = json.load(file)

    img_path = args.img_path
    im = Image.open(img_path)

    visualize_objects_layouts(Visual_Semantics, im)
    visualize_scene_graph(Visual_Semantics['Full_SceneGraph'], vis_rel=False)
    visualize_scene_graph(Visual_Semantics['Filtered_SceneGraph'], vis_rel=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SGG', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)