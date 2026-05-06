import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.models import load_model

# -----------------------
# Load data
# -----------------------

feature = np.load("feature/featuremap_test.npy")
labels = np.load("feature/labels_test.npy")

# -----------------------
# Load model
# -----------------------

model = load_model("model/MARS.h5", compile=False)

print("Predicting...")
pred = model.predict(feature)

# -----------------------
# reshape（最重要）
# -----------------------

def to_joints(arr):
    return arr.reshape(3,19).T

# -----------------------
# 座標轉換
# -----------------------

def convert(j):

    # ❌ 刪掉這行
    # center = np.mean(j, axis=0)
    # j = j - center

    # 保持原始位置

    x = j[:,0]
    y = j[:,1]   # 高度
    z = j[:,2]

    return np.stack([x,y,z],axis=1)

# -----------------------
# Plot
# -----------------------

fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

def update(frame):

    ax1.clear()
    ax2.clear()
    ax3.clear()

    gt = convert(to_joints(labels[frame]))
    pr = convert(to_joints(pred[frame]))

    radar = feature[frame]

    points = []
    for i in range(8):
        for j in range(8):
            val = radar[i,j,4]
            points.append([i/4-1, val*3, j/4-1])

    points = np.array(points)

    # Radar
    ax1.scatter(points[:,0],points[:,1],points[:,2],c='green')
    ax1.set_title("Radar")

    # Prediction
    ax2.scatter(pr[:,0],pr[:,1],pr[:,2],c='red')
    ax2.set_title("Prediction")

    # Ground Truth
    ax3.scatter(gt[:,0],gt[:,1],gt[:,2],c='blue')
    ax3.set_title("Ground Truth")

    for ax in [ax1,ax2,ax3]:
        ax.set_xlim(-1,1)
        ax.set_ylim(0,3)
        ax.set_zlim(-1,1)
        # ax.view_init(elev=15, azim=-70)

ani = FuncAnimation(fig, update, frames=300, interval=100)

plt.show()