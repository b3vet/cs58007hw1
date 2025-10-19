import csv, math, numpy as np, matplotlib.pyplot as plt

def read_sensor_pair(acc_path, gyro_path):
    def read_csv(p):
        with open(p) as f:
            r = csv.reader(f)
            head = [h.strip().lower() for h in next(r)]
            t_idx = head.index("seconds_elapsed")
            x_idx, y_idx, z_idx = head.index("x"), head.index("y"), head.index("z")
            t,x,y,z=[],[],[],[]
            for row in r:
                try:
                    t.append(float(row[t_idx]))
                    x.append(float(row[x_idx]))
                    y.append(float(row[y_idx]))
                    z.append(float(row[z_idx]))
                except: pass
            return np.array(t),np.array(x),np.array(y),np.array(z)
    ta,ax,ay,az = read_csv(acc_path)
    tg,gx,gy,gz = read_csv(gyro_path)
    # Align by interpolation onto accel timeline
    gx_i = np.interp(ta,tg,gx); gy_i=np.interp(ta,tg,gy); gz_i=np.interp(ta,tg,gz)
    return ta,ax,ay,az,gx_i,gy_i,gz_i

def complementary_filter(acc_path, gyro_path, alpha=0.98, plot=True):
    t,ax,ay,az,gx,gy,gz = read_sensor_pair(acc_path,gyro_path)
    fs = 1/np.median(np.diff(t))
    dt = 1/fs

    pitch, roll = 0.0, 0.0
    pitch_log, roll_log = [], []

    for i in range(len(t)):
        pitch_acc = math.degrees(math.atan2(ay[i], math.sqrt(ax[i]**2 + az[i]**2)))
        roll_acc  = math.degrees(math.atan2(-ax[i], math.sqrt(ay[i]**2 + az[i]**2)))

        pitch_gyro = pitch + math.degrees(gy[i]*dt)
        roll_gyro  = roll  + math.degrees(gx[i]*dt)

        pitch = alpha*pitch_gyro + (1-alpha)*pitch_acc
        roll  = alpha*roll_gyro  + (1-alpha)*roll_acc

        pitch_log.append(pitch)
        roll_log.append(roll)

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(t-t[0], pitch_log, label="Pitch (°)")
        plt.plot(t-t[0], roll_log, label="Roll (°)")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (degrees)")
        plt.title("Pose Estimation via Complementary Filter")
        plt.legend(); plt.tight_layout(); plt.show()

    return {"fs":fs,"pitch":pitch_log,"roll":roll_log}
