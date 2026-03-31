# Align estimated trajectory to ground truth using Umeyama method
def umeyama_alignment(X, Y):
    # X, Y: Nx3 matrices (X: estimated, Y: ground truth)
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    S = np.dot(Xc.T, Yc) / X.shape[0]
    U, D, Vt = np.linalg.svd(S)
    S_mat = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_mat[2, 2] = -1
    R_ = np.dot(U, np.dot(S_mat, Vt))
    var_X = np.var(Xc, axis=0).sum()
    c = 1.0 / var_X * np.trace(np.dot(np.diag(D), S_mat))
    t = mu_Y - c * np.dot(R_, mu_X)
    return c, R_, t
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Helper to read TUM trajectory file (timestamp tx ty tz qx qy qz qw)
def read_trajectory(file_path):
    traj = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            data = line.strip().split()
            if len(data) != 8:
                continue
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, data)
            traj.append({
                'timestamp': timestamp,
                't': np.array([tx, ty, tz]),
                'q': np.array([qx, qy, qz, qw])
            })
    return traj


# Associate trajectories by nearest timestamp
def associate_trajectories(gt_traj, est_traj, max_diff=0.02):
    gt_dict = {x['timestamp']: x for x in gt_traj}
    est_dict = {x['timestamp']: x for x in est_traj}
    matches = []
    gt_times = np.array(list(gt_dict.keys()))
    for est_time in est_dict:
        idx = np.argmin(np.abs(gt_times - est_time))
        if abs(gt_times[idx] - est_time) < max_diff:
            matches.append((gt_times[idx], est_time))
    return matches

# Compute Absolute Trajectory Error (ATE)
def compute_ate(gt_traj, est_traj, matches):
    errors = []
    for gt_time, est_time in matches:
        gt_pose = gt_traj[[x['timestamp'] for x in gt_traj].index(gt_time)]
        est_pose = est_traj[[x['timestamp'] for x in est_traj].index(est_time)]
        error = np.linalg.norm(gt_pose['t'] - est_pose['t'])
        errors.append(error)
    return np.array(errors)

# Compute Relative Pose Error (RPE)
def compute_rpe(gt_traj, est_traj, matches, delta=1):
    errors = []
    for i in range(len(matches) - delta):
        gt1 = gt_traj[[x['timestamp'] for x in gt_traj].index(matches[i][0])]
        gt2 = gt_traj[[x['timestamp'] for x in gt_traj].index(matches[i+delta][0])]
        est1 = est_traj[[x['timestamp'] for x in est_traj].index(matches[i][1])]
        est2 = est_traj[[x['timestamp'] for x in est_traj].index(matches[i+delta][1])]
        gt_rel = gt2['t'] - gt1['t']
        est_rel = est2['t'] - est1['t']
        error = np.linalg.norm(gt_rel - est_rel)
        errors.append(error)
    return np.array(errors)

if __name__ == "__main__":
    gt_traj = read_trajectory("../datasets/TUM/rgbd_dataset_freiburg1_xyz/groundtruth.txt")
    est_traj = read_trajectory("KeyFrameTrajectory.txt")
    matches = associate_trajectories(gt_traj, est_traj)
    # Extract matched positions
    gt_xyz = np.array([gt_traj[[x['timestamp'] for x in gt_traj].index(gt)]["t"] for gt, _ in matches])
    est_xyz = np.array([est_traj[[x['timestamp'] for x in est_traj].index(est)]["t"] for _, est in matches])

    # Align SLAM trajectory to ground truth
    c, R_, t = umeyama_alignment(est_xyz, gt_xyz)
    est_xyz_aligned = (c * (R_ @ est_xyz.T).T) + t

    # Replace est_xyz in error computation with aligned version
    def compute_ate_aligned(gt_xyz, est_xyz_aligned):
        return np.linalg.norm(gt_xyz - est_xyz_aligned, axis=1)

    ate = compute_ate_aligned(gt_xyz, est_xyz_aligned)
    rpe = compute_rpe(gt_traj, est_traj, matches)  # RPE is usually computed before alignment
    print(f"ATE RMSE (aligned): {np.sqrt(np.mean(ate**2)):.4f} m")
    print(f"RPE RMSE: {np.sqrt(np.mean(rpe**2)):.4f} m")

    # Save ATE and RPE to files
    np.savetxt("ate_values.txt", ate, header="ATE (m) (aligned)")
    np.savetxt("rpe_values.txt", rpe, header="RPE (m)")

    # Plot and save ground truth vs aligned SLAM pose
    plt.figure(figsize=(8,6))
    plt.plot(gt_xyz[:,0], gt_xyz[:,1], label='Ground Truth', linewidth=2)
    plt.plot(est_xyz_aligned[:,0], est_xyz_aligned[:,1], label='SLAM (aligned)', linewidth=2)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory: Ground Truth vs SLAM (aligned)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png")
    plt.show()
