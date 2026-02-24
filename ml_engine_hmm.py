import json
import socket
from collections import deque
from datetime import datetime
import math
import numpy as np
from hmmlearn.hmm import GaussianHMM

HOST = "127.0.0.1"
PORT = 5555

# =========================
# Config
# =========================
N_FEATURES_WINDOW = 300   # cuántas barras guardamos para entrenar
MIN_TRAIN = 250           # mínimo para entrenar el HMM
RETRAIN_EVERY = 150       # reentrena cada N barras (simple)
HYST = 0.90               # histéresis (como en tu captura)
FLIP_GUARD_INIT = 2       # evita flip-flop

# thresholds para clasificar (prob del estado dominante)
P_STRONG = 0.80

# =========================
# Buffers
# =========================
last_close = None
rets = deque(maxlen=N_FEATURES_WINDOW)
abs_rets = deque(maxlen=N_FEATURES_WINDOW)
ranges = deque(maxlen=N_FEATURES_WINDOW)

X = deque(maxlen=N_FEATURES_WINDOW)  # features [ret_norm, range_norm, vol_norm]

hmm = None
bars_seen = 0
last_regime = "NO_TRADE"
flip_guard = 0

# mapping states -> regimes (se decide tras entrenar)
state_to_regime = {0: "TRENDING", 1: "MEAN_REVERTING"}

def safe_std(arr):
    if len(arr) < 20:
        return 1e-6
    return float(np.std(arr, ddof=1) + 1e-6)

def compute_features(o, h, l, c):
    global last_close
    rng = max(h - l, 1e-6)

    if last_close is None:
        last_close = c
        return None

    r = c - last_close
    last_close = c

    rets.append(r)
    abs_rets.append(abs(r))
    ranges.append(rng)

    vol = safe_std(np.array(rets))
    avg_rng = float(np.mean(ranges)) if len(ranges) >= 10 else rng

    # features normalizadas
    ret_norm = r / vol
    range_norm = rng / max(avg_rng, 1e-6)
    vol_norm = vol / max(safe_std(np.array(abs_rets)), 1e-6)

    return [ret_norm, range_norm, vol_norm]

def train_hmm():
    global hmm, state_to_regime

    data = np.array(X, dtype=np.float64)
    if len(data) < MIN_TRAIN:
        return False

    # HMM gaussiano 2 estados
    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    model.fit(data)
    model.covars_ = model.covars_ + np.eye(model.covars_.shape[1]) * 1e-2
    hmm = model

    # Decide mapping estado->régimen mirando "varianza" del return_norm
    # Idea: TREND suele tener |ret_norm| más persistente / alta var,
    # MR suele tener menor var y alternancia.
    means = model.means_
    covs = model.covars_
    # feature 0 = ret_norm
    ret_var_state0 = covs[0][0][0]
    ret_var_state1 = covs[1][0][0]

    # estado con mayor var de ret_norm => TRENDING
    if ret_var_state0 >= ret_var_state1:
        state_to_regime = {0: "TRENDING", 1: "MEAN_REVERTING"}
    else:
        state_to_regime = {1: "TRENDING", 0: "MEAN_REVERTING"}

    return True

def hmm_predict_latest():
    # Devuelve (regime, conf, reject, meta)
    if hmm is None or len(X) < MIN_TRAIN:
        return "NO_TRADE", 0.0, True, "warmup"

    data = np.array(X, dtype=np.float64)
    # posterior por estado para cada obs
    post = hmm.predict_proba(data)
    p_last = post[-1]
    state = int(np.argmax(p_last))
    p = float(np.max(p_last))

    regime = state_to_regime[state]

    # conf: prob del estado dominante
    conf = 0.5 + 0.5 * (p ** 0.25)

    # reject si no es fuerte
    reject = (conf < P_STRONG)

    meta = f"state={state} p={conf:.3f}"
    return regime, conf, reject, meta

def apply_hysteresis(regime, conf, reject):
    global last_regime, flip_guard

    # si reject, no cambiamos last_regime (pero sí reportamos NO_TRADE)
    if reject:
        return "NO_TRADE", conf, True, "reject_low_prob"

    if regime != last_regime:
        if flip_guard > 0:
            # mantenemos el anterior un poco
            flip_guard -= 1
            return last_regime, max(0.60, conf - 0.05), False, "flip_guard"
        else:
            flip_guard = FLIP_GUARD_INIT
            last_regime = regime
            return regime, conf, False, "switch"

    return regime, conf, False, "hold"

def main():
    global bars_seen

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[ENGINE:HMM] Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"[ENGINE:HMM] Connected by {addr}")
            buf = b""

            while True:
                data = conn.recv(4096)
                if not data:
                    print("[ENGINE:HMM] Disconnected")
                    break
                buf += data

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        msg = json.loads(line.decode("utf-8"))
                        b = msg["bar"]
                        o = float(b["open"]); h = float(b["high"]); l = float(b["low"]); c = float(b["close"])

                        feats = compute_features(o, h, l, c)
                        if feats is not None:
                            X.append(feats)
                            bars_seen += 1

                        # retrain periódico
                        trained = False
                        if len(X) >= MIN_TRAIN and (bars_seen % RETRAIN_EVERY == 0):
                            trained = train_hmm()

                        regime, conf, reject, meta = hmm_predict_latest()
                        regime2, conf2, reject2, hmeta = apply_hysteresis(regime, conf, reject)

                        # log estilo tu captura
                        # ejemplo: [HMM] state=1 -> MEAN-REVERTING (conf=0.999) [...]
                        print(f"[HMM] {meta} -> {regime2} (conf={conf2:.3f}) hyst={HYST:.2f} ({hmeta}){' [retrained]' if trained else ''}")

                        resp = {
                            "regime": regime2,
                            "conf": round(conf2, 4),
                            "reject": bool(reject2),
                            "reason": f"{meta}|{hmeta}",
                            "hyst": HYST
                        }
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

                    except Exception as e:
                        err = {"regime": "NO_TRADE", "conf": 0.0, "reject": True, "reason": f"error:{e}", "hyst": HYST}
                        conn.sendall((json.dumps(err) + "\n").encode("utf-8"))

if __name__ == "__main__":
    main()