import matplotlib.pyplot as plt
import numpy as np
from mmdet3d.models.necks.view_transformer import LSSViewTransformer
from mpl_toolkits.mplot3d import proj3d
import pickle

def vis_frustum(frustum, labe_config):
    """Visualize the frustum points in the camera coordinate system."""
    D, H, W, dim_point = frustum.shape
    frustum_np = frustum.cpu().detach().numpy()
    d_coords, h_coords, w_coords = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    xs = frustum_np[..., 0].flatten()
    ys = frustum_np[..., 1].flatten()
    ds = frustum_np[..., 2].flatten()
    coord_data = frustum_np.reshape(-1, dim_point)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    scatter = ax.scatter(w_coords.flatten(), d_coords.flatten(), h_coords.flatten(),
                         c=ds, cmap='viridis', s=1)
    ax.set_xlabel('Width (W)')
    ax.set_ylabel('Depth (D)')
    ax.set_zlabel('Height (H))')
    plt.colorbar(scatter, label='Depth Value')
    ax.set_title(f'Frusum D: {D} H: {H} W: {W}')

    # hover setting
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter._offsets3d
        idx = ind["ind"][0]
        x2d, y2d, _ = proj3d.proj_transform(pos[0][idx], pos[2][idx], pos[1][idx], ax.get_proj())
        annot.xy = (x2d, y2d)

        # Retrieve the dim_point data for this point
        dim_point_data = coord_data[idx]
        text = f"Point Values:\n{labe_config['x']}: {dim_point_data[0]:.2f}\n{labe_config['y']}: {dim_point_data[1]:.2f}\n{labe_config['z']}: {dim_point_data[2]:.2f}"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

if __name__ == '__main__':
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }
    img_view_transformer_cfg=dict(
        grid_config=grid_config,
        input_size=(256, 704),
        in_channels=256,
        out_channels=64,
        downsample=16)
    view_transformer = LSSViewTransformer(**img_view_transformer_cfg)
    # step 1 create frustum
    frustum = view_transformer.frustum
    label_cfg = {
        "x" : "U",
        "y" : "V",
        "z" : "D"
    }
    vis_frustum(frustum, label_cfg)

    # step 2 project camera coordinate frustum to ego coordinate
    with open('./tools/vis/sample_calibs.pkl', 'rb') as f:
        calibs = pickle.load(f)
    ego_coord_by_frustums = view_transformer.get_lidar_coor(*calibs) # 12 cameras in 2 sample
    label_cfg.update({
        "x" : "Ego_x",
        "y" : "Ego_y",
        "z" : "Ego_z"})
    vis_frustum(ego_coord_by_frustums[0,4, ...], label_cfg)
