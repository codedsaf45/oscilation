"""
mlx_single_monitor.py
=====================
MLX90393 단일 센서 fast monitor용 호스트.
ADS131M08용 single_sensor_monitor.py와 거의 동일하지만
패킷 포맷이 다름:
  ADS:  ID(1) + float×3 = 13 byte
  MLX:  ID(1) + int16×3 = 7 byte
  → int16를 raw LSB로 받고, 호스트에서 정규화 또는 그대로 표시

조작:
  [R]  baseline 재캘리브
  [P]  PSD smoothing 토글
  [Q]  종료
"""

import serial
import time
import threading
import queue
import struct
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


CFG = dict(
    port      = "COM25",        # 본인 포트로
    baudrate  = 115200,

    SENSOR_ID = 0,             # 펌웨어가 1개만 보내므로 0 고정
    AXIS      = "z",           # 'x', 'y', 'z' 중 표시할 축

    buffer_sec       = 4.0,
    fft_window       = 1024,
    fft_hop          = 256,
    expected_fs      = 250.0,  # MLX90393는 ADS보다 느림

    psd_smooth_alpha = 0.2,
    use_smoothing    = True,

    dc_cutoff_hz     = 5.0,
    top_n_peaks      = 3,
    peak_min_db      = -120.0,

    plot_refresh_ms  = 50,
    autoscale_time   = True,

    # MLX raw LSB → µT 변환 (대략값, GAIN/RES 설정에 따라 다름)
    # GAIN=7 (0.625x), RES=0 → 약 0.150 µT/LSB (X,Y), 0.242 (Z)
    # 정확한 값은 데이터시트 표 참조
    lsb_to_uT_xy     = 0.150,
    lsb_to_uT_z      = 0.242,
)

HEADER = 0xAA
FOOTER = 0x55
BYTES_PER_SENSOR_MLX = 7   # ID(1) + int16×3(6)


class MLXReader(threading.Thread):
    def __init__(self, port, baudrate, axis, data_queue, stop_event, cfg):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.axis = axis.lower()
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.cfg = cfg
        self.ser = None
        self.n_ok = 0
        self.n_err = 0

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            print(f"  ✓ Serial open: {self.port}  (axis={self.axis})")
            while not self.stop_event.is_set():
                if not self._sync_header():
                    continue
                self._read_packet()
        except Exception as e:
            print(f"  ✗ Serial error: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def _sync_header(self):
        while not self.stop_event.is_set():
            b = self.ser.read(1)
            if not b:
                return False
            if b[0] == HEADER:
                return True
        return False

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n and not self.stop_event.is_set():
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                continue
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _read_packet(self):
        count_b = self._read_exact(1)
        if count_b is None:
            return
        count = count_b[0]
        if count == 0 or count > 20:
            self.n_err += 1
            return
        payload = self._read_exact(count * BYTES_PER_SENSOR_MLX)
        if payload is None:
            self.n_err += 1
            return
        tail = self._read_exact(2)
        if tail is None:
            self.n_err += 1
            return
        recv_chk = tail[0]
        footer = tail[1]
        if footer != FOOTER:
            self.n_err += 1
            return
        calc = count
        for byte in payload:
            calc ^= byte
        if calc != recv_chk:
            self.n_err += 1
            return

        # 첫 번째 센서만 추출 (펌웨어가 1개만 보냄)
        sid = payload[0]
        x, y, z = struct.unpack_from("<hhh", payload, 1)

        # 축 선택
        if self.axis == "x":
            val = float(x) * self.cfg["lsb_to_uT_xy"]
        elif self.axis == "y":
            val = float(y) * self.cfg["lsb_to_uT_xy"]
        else:  # z
            val = float(z) * self.cfg["lsb_to_uT_z"]

        # µT → mT 통일 (다른 코드와 단위 맞춤)
        val_mT = val * 1e-3

        self.n_ok += 1
        try:
            self.data_queue.put_nowait((time.perf_counter(), val_mT))
        except queue.Full:
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait((time.perf_counter(), val_mT))
            except queue.Empty:
                pass


# =========================================================
# Peak utils (이전 코드와 동일)
# =========================================================
def find_peaks_simple(y, min_distance=2):
    peaks = []
    n = len(y)
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
    if not peaks:
        return np.array([], dtype=int)
    peaks = np.array(peaks, dtype=int)
    if min_distance > 1:
        order = np.argsort(-y[peaks])
        kept = []
        used = np.zeros(n, dtype=bool)
        for idx in peaks[order]:
            lo = max(0, idx - min_distance)
            hi = min(n, idx + min_distance + 1)
            if used[lo:hi].any():
                continue
            kept.append(idx)
            used[lo:hi] = True
        peaks = np.array(sorted(kept), dtype=int)
    return peaks


def parabolic_interp(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(i)
    return i + 0.5 * (y0 - y2) / denom


# =========================================================
# App
# =========================================================
class MLXSingleApp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.axis = cfg["AXIS"]

        self.maxlen = int(cfg["expected_fs"] * cfg["buffer_sec"])
        self.t_buf = deque(maxlen=self.maxlen)
        self.x_buf = deque(maxlen=self.maxlen)

        self.baseline = 0.0
        self.baseline_n = 0
        self.baseline_target = 100
        self.baseline_ready = False

        self.fps_window = deque(maxlen=200)
        self.fs_current = 0.0

        self.samples_since_last_fft = 0
        self.psd_smoothed = None
        self.use_smoothing = cfg["use_smoothing"]

        self.latest_psd_db = None
        self.latest_freqs = None
        self.latest_peaks = []

        self.data_queue = queue.Queue(maxsize=500)
        self.stop_event = threading.Event()
        self.reader = MLXReader(
            cfg["port"], cfg["baudrate"], self.axis,
            self.data_queue, self.stop_event, cfg,
        )

        self.window_func = np.hanning(cfg["fft_window"]).astype(np.float32)
        self.window_norm = float(np.sum(self.window_func ** 2))

    def _update_baseline(self, x):
        if self.baseline_ready:
            return
        self.baseline = (self.baseline * self.baseline_n + x) / (self.baseline_n + 1)
        self.baseline_n += 1
        if self.baseline_n >= self.baseline_target:
            self.baseline_ready = True
            print(f"  ✓ Baseline ready ({self.baseline_n} samples), "
                  f"offset = {self.baseline:+.4f} mT")

    def _consume(self):
        while True:
            try:
                t, x = self.data_queue.get_nowait()
            except queue.Empty:
                break

            self._update_baseline(x)
            x_corr = x - self.baseline

            self.t_buf.append(t)
            self.x_buf.append(x_corr)
            self.fps_window.append(t)
            self.samples_since_last_fft += 1

        if len(self.fps_window) >= 2:
            span = self.fps_window[-1] - self.fps_window[0]
            if span > 0:
                self.fs_current = (len(self.fps_window) - 1) / span

        if self.fs_current > 0:
            target = int(self.fs_current * self.cfg["buffer_sec"])
            target = max(target, self.cfg["fft_window"] * 4)
            if target != self.t_buf.maxlen:
                self.t_buf = deque(self.t_buf, maxlen=target)
                self.x_buf = deque(self.x_buf, maxlen=target)

    def _compute_psd(self):
        N = self.cfg["fft_window"]
        if len(self.x_buf) < N:
            return None, None
        data = np.array(list(self.x_buf)[-N:], dtype=np.float32)
        data = data - data.mean()
        data *= self.window_func
        spec = np.fft.rfft(data)
        psd = (np.abs(spec) ** 2) / (self.window_norm + 1e-12)
        psd_db = 10.0 * np.log10(psd + 1e-12).astype(np.float32)

        if self.fs_current > 0:
            freqs = np.fft.rfftfreq(N, d=1.0 / self.fs_current)
        else:
            freqs = np.arange(len(psd_db), dtype=np.float32)

        if self.use_smoothing:
            if self.psd_smoothed is None or self.psd_smoothed.shape != psd_db.shape:
                self.psd_smoothed = psd_db.copy()
            else:
                a = self.cfg["psd_smooth_alpha"]
                self.psd_smoothed = a * psd_db + (1 - a) * self.psd_smoothed
            return self.psd_smoothed, freqs
        else:
            self.psd_smoothed = None
            return psd_db, freqs

    def _find_peaks(self, psd_db, freqs):
        if psd_db is None or freqs is None or len(psd_db) < 4:
            return []
        mask = freqs >= self.cfg["dc_cutoff_hz"]
        if not mask.any():
            return []
        psd_masked = psd_db.copy()
        psd_masked[~mask] = -1e9
        peaks = find_peaks_simple(psd_masked, min_distance=2)
        peaks = peaks[psd_masked[peaks] >= self.cfg["peak_min_db"]]
        if len(peaks) == 0:
            return []
        order = np.argsort(-psd_masked[peaks])
        peaks = peaks[order][:self.cfg["top_n_peaks"]]

        out = []
        for p in peaks:
            p_frac = parabolic_interp(psd_db, int(p))
            if 0 <= p_frac < len(freqs) - 1:
                lo = int(np.floor(p_frac))
                frac = p_frac - lo
                f = freqs[lo] * (1 - frac) + freqs[lo + 1] * frac
            else:
                f = float(freqs[int(p)])
            out.append((float(f), float(psd_db[int(p)])))
        return out

    def setup_plot(self):
        plt.style.use("default")
        self.fig, (self.ax_time, self.ax_psd) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [1, 1]},
        )
        self.fig.subplots_adjust(hspace=0.4, bottom=0.1, top=0.93)

        self.ax_time.set_title(self._title_str())
        self.ax_time.set_xlabel("time (s)")
        self.ax_time.set_ylabel(f"Δ B{self.axis.upper()} (mT)")
        self.ax_time.grid(True, alpha=0.3)
        (self.line_time,) = self.ax_time.plot([], [], lw=1.0, color="C0")

        self.ax_psd.set_title("PSD")
        self.ax_psd.set_xlabel("freq (Hz)")
        self.ax_psd.set_ylabel("dB")
        self.ax_psd.grid(True, alpha=0.3)
        (self.line_psd,) = self.ax_psd.plot([], [], lw=1.2, color="C2")

        self.peak_scatter = self.ax_psd.scatter(
            [], [], s=60, marker="v",
            facecolor="red", edgecolor="black", zorder=5,
        )
        self.peak_text = self.ax_psd.text(
            0.98, 0.95, "", transform=self.ax_psd.transAxes,
            ha="right", va="top", fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="gray", alpha=0.9),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: self._on_quit())

    def _title_str(self):
        if self.latest_peaks:
            f, d = self.latest_peaks[0]
            dom = f"  dom: {f:6.1f} Hz ({d:+5.1f} dB)"
        else:
            dom = "  dom: --"
        sm = "ON" if self.use_smoothing else "OFF"
        return (f"MLX axis={self.axis.upper()}   "
                f"fs = {self.fs_current:6.1f} Hz   "
                f"buf = {len(self.x_buf):5d}   "
                f"smooth = {sm}{dom}")

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            self._on_quit()
        elif k == "r":
            print("  ↻ Recalibrating baseline...")
            self.baseline = 0.0
            self.baseline_n = 0
            self.baseline_ready = False
        elif k == "p":
            self.use_smoothing = not self.use_smoothing
            self.psd_smoothed = None
            print(f"  → PSD smoothing = {self.use_smoothing}")

    def _on_quit(self):
        self.stop_event.set()
        try:
            plt.close("all")
        except Exception:
            pass

    def animate(self, frame):
        self._consume()

        if len(self.x_buf) > 1:
            ts = np.array(self.t_buf)
            xs = np.array(self.x_buf)
            ts = ts - ts[-1]
            self.line_time.set_data(ts, xs)
            self.ax_time.set_xlim(ts[0], 0)
            if self.cfg["autoscale_time"]:
                m = max(0.001, np.max(np.abs(xs)))
                self.ax_time.set_ylim(-m * 1.2, m * 1.2)

        if self.samples_since_last_fft >= self.cfg["fft_hop"]:
            self.samples_since_last_fft = 0
            psd_db, freqs = self._compute_psd()
            if psd_db is not None:
                self.latest_psd_db = psd_db
                self.latest_freqs = freqs
                self.latest_peaks = self._find_peaks(psd_db, freqs)

                self.line_psd.set_data(freqs, psd_db)
                if len(freqs) > 1:
                    self.ax_psd.set_xlim(0, freqs[-1])
                ymax = float(np.max(psd_db)) + 5
                ymin = max(-160, float(np.min(psd_db)) - 5)
                self.ax_psd.set_ylim(ymin, ymax)

                if self.latest_peaks:
                    pf = np.array([p[0] for p in self.latest_peaks])
                    pd = np.array([p[1] for p in self.latest_peaks])
                    self.peak_scatter.set_offsets(np.column_stack([pf, pd]))
                    lines = []
                    for i, (f, d) in enumerate(self.latest_peaks):
                        prefix = "★" if i == 0 else " "
                        lines.append(f"{prefix} {f:6.1f} Hz  {d:+5.1f} dB")
                    self.peak_text.set_text("\n".join(lines))
                else:
                    self.peak_scatter.set_offsets(np.empty((0, 2)))
                    self.peak_text.set_text("(no peak)")

        self.ax_time.set_title(self._title_str())

    def run(self):
        print("=" * 60)
        print(f"  MLX90393 Single-Sensor Monitor  (axis = {self.axis.upper()})")
        print("=" * 60)
        print("  [R] baseline | [P] smoothing | [Q] quit")
        print("=" * 60)

        self.reader.start()
        self.setup_plot()
        anim = FuncAnimation(
            self.fig, self.animate,
            interval=self.cfg["plot_refresh_ms"],
            blit=False, cache_frame_data=False,
        )
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            self.reader.join(timeout=1.0)


if __name__ == "__main__":
    app = MLXSingleApp(CFG)
    app.run()
