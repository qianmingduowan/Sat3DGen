import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import os

def parse_args():
    # Parse CLI arguments
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument('--input_img_path', type=str, required=True)
    parse_args.add_argument('--work_dir', type=str, default='./results')
    parse_args.add_argument('--num_of_point', type=int, default=79)
    parse_args.add_argument('--force', action='store_true')
    parse_args.add_argument('--save_same_name', action='store_true',
                            help='Save trajectory as a .csv file with the same name as the input image (in the same directory).')
    return parse_args.parse_args()


def select_points(sat_image,num_of_point):
    fig = plt.figure()
    fig.set_size_inches(1,1,forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.imshow(sat_image)

    coords = []
    state = {"dragging": False}
    fig.add_axes(ax)

    def on_press(event):
        if event.button == 1 and event.inaxes == ax:  # left mouse button and inside axes
            state["dragging"] = True
            print(f"Start at ({float(event.xdata)}, {float(event.ydata)})")

    def on_drag(event):
        if not state["dragging"] or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        coords.append((x, y))

    def on_release(event):
        if event.button == 1 and state["dragging"]:
            state["dragging"] = False
            print("Mouse released. Recording stopped.")
            fig.canvas.mpl_disconnect(press_cid)
            fig.canvas.mpl_disconnect(drag_cid)
            fig.canvas.mpl_disconnect(release_cid)
            plt.close(fig)

    # Connect the event handlers.
    press_cid = fig.canvas.mpl_connect('button_press_event', on_press)
    drag_cid = fig.canvas.mpl_connect('motion_notify_event', on_drag)
    release_cid = fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show()
    
    pixels = np.array(list(dict.fromkeys(coords)))  # Remove duplicates while preserving order.
    tck, u = splprep(pixels.T, s=25, per=0)
    u_new = np.linspace(u.min(), u.max(), num_of_point+1)
    x_new, y_new = splev(u_new, tck)
    smooth_path = np.array([x_new,y_new]).T
    angles = np.arctan2(y_new[1:]-y_new[:-1],x_new[1:]-x_new[:-1])

    # Final visualization.
    fig_final, ax_final = plt.subplots()
    ax_final.imshow(sat_image)
    ax_final.plot(pixels[:,0], pixels[:,1], 'o', color='red')
    ax_final.plot(smooth_path[:,0], smooth_path[:,1], '-', color='blue')
    plt.show()

    return pixels, angles, smooth_path,fig_final


def main():
    args = parse_args()
    save_path = os.path.join(args.work_dir, os.path.basename(args.input_img_path).rsplit('.', 1)[0])
    os.makedirs(save_path, exist_ok=True)

    save_csv = os.path.join(save_path, 'pixels.csv')
    if os.path.exists(save_csv):
        if not args.force:
            print(f'File already exists, please delete it first or direct use the {os.path.abspath(save_csv)}, or use --force to overwrite it.')
            return 
        else:
            print(f'File already exists, but --force is used, so overwriting the {os.path.abspath(save_csv)}.')

    print('Please left click and drag to select points on the image. Close the window when done.')
    print('Note: This requires a screen display or X11 forwarding to work properly.')
    assert os.environ.get('DISPLAY') is not None, "DISPLAY environment variable is not set. Please ensure you have a screen display or X11 forwarding."

    sat_image = plt.imread(args.input_img_path)
    pixels, angles, smooth_path,fig_final = select_points(sat_image,args.num_of_point)
    fig_final.savefig(os.path.join(save_path,'trajectory.png'),dpi=300)

    with open(save_csv, 'w') as f:
        f.write('w,h,angle\n')
        for i in range(len(smooth_path)-1):
            f.write(f'{smooth_path[i][0]},{smooth_path[i][1]},{angles[i]}\n')
    print('the visualization of trajectory and visualize img is saved to:', os.path.abspath(save_csv))

    # Save trajectory as a same-name .csv alongside the input image
    if args.save_same_name:
        input_dir = os.path.dirname(os.path.abspath(args.input_img_path))
        input_stem = os.path.basename(args.input_img_path).rsplit('.', 1)[0]
        same_name_csv = os.path.join(input_dir, input_stem + '.csv')
        with open(same_name_csv, 'w') as f:
            f.write('w,h,angle\n')
            for i in range(len(smooth_path) - 1):
                f.write(f'{smooth_path[i][0]},{smooth_path[i][1]},{angles[i]}\n')
        print(f'Same-name trajectory saved to: {os.path.abspath(same_name_csv)}')

if __name__ == '__main__':
    main()
