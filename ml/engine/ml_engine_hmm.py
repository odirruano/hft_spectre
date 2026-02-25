import json
import socket
from collections import deque
from datetime import datetime
import numpy as np
from hmmlearn.hmm import GaussianHMM

# XGBoost (ya lo tienes instalado)
from xgboost import XGBClassifier

HOST = "127.0.0.1"
PORT = 5555

# =========================
# Config HMM
# =========================
N_FEATURES_WINDOW = 300   # cuántas barras guardamos para entrenar
MIN_TRAIN = 250           # mínimo para entrenar el HMM
RETRAIN_EVERY = 150       # reentrena cada N barras (simple)
HYST = 0.90               # histéresis
FLIP_GUARD_INIT = 2       # evita flip-flop
P_STRONG = 0.80           # threshold para reject

# =========================
# Config XGBoost (Filtro)
# =========================
USE_XGB = True
XGB_MIN_TRAIN = 800           # mínimo para entrenar XGB (recomendado > HMM)
XGB_RETRAIN_EVERY = 250       # cada N muestras (labels) vuelve a entrenar
XGB_WARMUP_PASS = True        # mientras no esté listo, no bloquea trades

# Label horizon: cuántas barras adelante medimos "edge"
LABEL_H = 3

# “Edge” threshold en unidades normalizadas (ret/vol). Ajustable:
EDGE_RET_NORM = 0.35

# Gate: el C# ya tiene su propio XgbMinProb; aquí lo devolvemos como prob
# y también devolvemos pass basado en un mínimo local (para debug/consistencia).
XGB_PASS_MIN_PROB_LOCAL = 0.55

# =========================
# Buffers HMM
# =========================
last_close = None
rets = deque(maxlen=N_FEATURES_WINDOW)
abs_rets = deque(maxlen=N_FEATURES_WINDOW)
ranges = deque(maxlen=N_FEATURES_WINDOW)

# Features base HMM: [ret_norm, range_norm, vol_norm]
X = deque(maxlen=N_FEATURES_WINDOW)

hmm = None
bars_seen = 0
last_regime = "NO_TRADE"
flip_guard = 0
state_to_regime = {0: "TRENDING", 1: "MEAN_REVERTING"}

# =========================
# Buffers XGBoost
# =========================
# Guardamos features por barra para etiquetar cuando llegue t+H
xgb_feat_queue = deque(maxlen=5000)   # cada item: (feat_vec, close_at_t)
xgb_close_queue = deque(maxlen=5000)  # solo closes para construir label

# dataset acumulado XGB
xgb_X = []
xgb_y = []
xgb_model = None
xgb_ready = False
xgb_samples = 0

def safe_std(arr):
    if len(arr) < 20:
        return 1e-6
    return float(np.std(arr, ddof=1) + 1e-6)

def compute_features(o, h, l, c):
    """
    Features normalizadas para régimen + para XGB.
    """
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

    # base features (las que ya usas)
    ret_norm = r / vol
    range_norm = rng / max(avg_rng, 1e-6)
    vol_norm = vol / max(safe_std(np.array(abs_rets)), 1e-6)

    return [ret_norm, range_norm, vol_norm]

def train_hmm():
    global hmm, state_to_regime

    data = np.array(X, dtype=np.float64)
    if len(data) < MIN_TRAIN:
        return False

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=200,
        random_state=42
    )

    model.fit(data)
    # regularización suave
    model.covars_ = model.covars_ + np.eye(model.covars_.shape[1]) * 1e-2
    hmm = model

    # map states: el estado con mayor var en ret_norm => TRENDING
    covs = model.covars_
    ret_var_state0 = covs[0][0][0]
    ret_var_state1 = covs[1][0][0]

    if ret_var_state0 >= ret_var_state1:
        state_to_regime = {0: "TRENDING", 1: "MEAN_REVERTING"}
    else:
        state_to_regime = {1: "TRENDING", 0: "MEAN_REVERTING"}

    return True

def hmm_predict_latest():
    """
    Devuelve (regime, conf, reject, meta)
    """
    if hmm is None or len(X) < MIN_TRAIN:
        return "NO_TRADE", 0.0, True, "warmup"

    data = np.array(X, dtype=np.float64)
    post = hmm.predict_proba(data)
    p_last = post[-1]
    state = int(np.argmax(p_last))
    p = float(np.max(p_last))

    regime = state_to_regime[state]

    # conf suavizada
    conf = 0.5 + 0.5 * (p ** 0.25)

    reject = (conf < P_STRONG)

    meta = f"state={state} p={conf:.3f}"
    return regime, conf, reject, meta

def apply_hysteresis(regime, conf, reject):
    global last_regime, flip_guard

    if reject:
        return "NO_TRADE", conf, True, "reject_low_prob"

    if regime != last_regime:
        if flip_guard > 0:
            flip_guard -= 1
            return last_regime, max(0.60, conf - 0.05), False, "flip_guard"
        else:
            flip_guard = FLIP_GUARD_INIT
            last_regime = regime
            return regime, conf, False, "switch"

    return regime, conf, False, "hold"

# =========================
# XGBoost helpers
# =========================
def make_xgb_features(base_feats, regime_str, hmm_conf):
    """
    Construye vector final para XGB:
    [ret_norm, range_norm, vol_norm, regime_is_trend, hmm_conf]
    """
    regime_is_trend = 1.0 if regime_str == "TRENDING" else 0.0
    return [
        float(base_feats[0]),
        float(base_feats[1]),
        float(base_feats[2]),
        float(regime_is_trend),
        float(hmm_conf),
    ]

def compute_label_from_future(close_t, close_tH, vol_est):
    """
    Label binario:
    1 si el movimiento futuro (t->t+H) es suficientemente grande vs volatilidad
    """
    delta = close_tH - close_t
    # normalizamos por volatilidad estimada (en puntos)
    denom = max(vol_est, 1e-6)
    delta_norm = abs(delta) / denom
    return 1 if delta_norm >= EDGE_RET_NORM else 0

def train_xgb():
    global xgb_model, xgb_ready

    if len(xgb_X) < XGB_MIN_TRAIN:
        return False

    Xnp = np.array(xgb_X, dtype=np.float32)
    ynp = np.array(xgb_y, dtype=np.int32)

    model = XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )

    model.fit(Xnp, ynp)
    xgb_model = model
    xgb_ready = True
    return True

def xgb_predict_prob(feat_vec):
    """
    Devuelve probabilidad de clase 1 (TRADE)
    """
    if (not USE_XGB) or (xgb_model is None) or (not xgb_ready):
        return 1.0 if XGB_WARMUP_PASS else 0.0

    X1 = np.array([feat_vec], dtype=np.float32)
    p = float(xgb_model.predict_proba(X1)[0, 1])
    return p

def main():
    global bars_seen, xgb_samples

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[ENGINE:HMM+XGB] Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"[ENGINE:HMM+XGB] Connected by {addr}")
            buf = b""

            while True:
                data = conn.recv(4096)
                if not data:
                    print("[ENGINE:HMM+XGB] Disconnected")
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

                        # 1) features base (HMM base)
                        feats = compute_features(o, h, l, c)
                        if feats is not None:
                            X.append(feats)
                            bars_seen += 1

                        # 2) retrain HMM periódicamente
                        trained_hmm = False
                        if len(X) >= MIN_TRAIN and (bars_seen % RETRAIN_EVERY == 0):
                            trained_hmm = train_hmm()

                        # 3) HMM predict + hysteresis
                        regime, conf, reject, meta = hmm_predict_latest()
                        regime2, conf2, reject2, hmeta = apply_hysteresis(regime, conf, reject)

                        # 4) XGB: construir features por barra y etiquetar cuando haya futuro
                        trained_xgb = False
                        xgb_label = "WARMUP"
                        xgb_prob = 1.0 if XGB_WARMUP_PASS else 0.0
                        xgb_pass = True if XGB_WARMUP_PASS else False

                        if feats is not None:
                            # vol_est: usamos std(rets) en puntos (igual que compute_features)
                            vol_est = safe_std(np.array(rets))

                            xgb_feat = make_xgb_features(feats, regime2, conf2)
                            xgb_feat_queue.append((xgb_feat, c, vol_est))
                            xgb_close_queue.append(c)

                            # si ya tenemos t+H, creamos label para el feature de hace H barras
                            if len(xgb_feat_queue) > LABEL_H:
                                feat_old, close_old, vol_old = xgb_feat_queue[-(LABEL_H+1)]
                                close_now = c  # approx close at t+H

                                y = compute_label_from_future(close_old, close_now, vol_old)
                                xgb_X.append(feat_old)
                                xgb_y.append(y)
                                xgb_samples = len(xgb_y)

                                # retrain XGB
                                if USE_XGB and xgb_samples >= XGB_MIN_TRAIN and (xgb_samples % XGB_RETRAIN_EVERY == 0):
                                    trained_xgb = train_xgb()

                        # 5) XGB infer para el último feature (si existe)
                        if USE_XGB and len(xgb_feat_queue) > 0:
                            feat_last, _, _ = xgb_feat_queue[-1]
                            if xgb_ready:
                                xgb_prob = xgb_predict_prob(feat_last)
                                xgb_pass = bool(xgb_prob >= XGB_PASS_MIN_PROB_LOCAL)
                                xgb_label = "TRADE" if xgb_pass else "NO_TRADE"
                            else:
                                # warmup
                                xgb_prob = 1.0 if XGB_WARMUP_PASS else 0.0
                                xgb_pass = True if XGB_WARMUP_PASS else False
                                xgb_label = "WARMUP"

                        # log estilo tu captura
                        print(f"[HMM] {meta} -> {regime2} (conf={conf2:.3f}) hyst={HYST:.2f} ({hmeta}){' [retrained]' if trained_hmm else ''}"
                              f" | [XGB] ready={xgb_ready} p={xgb_prob:.3f} pass={xgb_pass} samples={xgb_samples}{' [retrained]' if trained_xgb else ''}")

                        resp = {
                            "regime": regime2,
                            "conf": round(conf2, 4),
                            "reject": bool(reject2),
                            "reason": f"{meta}|{hmeta}",
                            "hyst": HYST,

                            # XGBoost fields (nuevo)
                            "xgb_ready": bool(xgb_ready),
                            "xgb_prob": round(float(xgb_prob), 4),
                            "xgb_pass": bool(xgb_pass),
                            "xgb_label": str(xgb_label),
                            "xgb_samples": int(xgb_samples),
                        }

                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

                    except Exception as e:
                        err = {
                            "regime": "NO_TRADE",
                            "conf": 0.0,
                            "reject": True,
                            "reason": f"error:{e}",
                            "hyst": HYST,
                            "xgb_ready": False,
                            "xgb_prob": 0.0,
                            "xgb_pass": False,
                            "xgb_label": "ERROR",
                            "xgb_samples": int(xgb_samples),
                        }
                        conn.sendall((json.dumps(err) + "\n").encode("utf-8"))

if __name__ == "__main__":
    main()