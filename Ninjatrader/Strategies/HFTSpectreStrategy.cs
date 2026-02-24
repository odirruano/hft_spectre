#region Using declarations
using System;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Gui.NinjaScript;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
using System.Windows.Media;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class HFTSpectreStrategy : Strategy
    {
        // =========================
        // Inputs - ML
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "ML Host", Order = 1, GroupName = "ML")]
        public string MlHost { get; set; } = "127.0.0.1";

        [NinjaScriptProperty]
        [Display(Name = "ML Port", Order = 2, GroupName = "ML")]
        public int MlPort { get; set; } = 5555;

        [NinjaScriptProperty]
        [Display(Name = "Send Every N Bars", Order = 3, GroupName = "ML")]
        public int SendEveryNBars { get; set; } = 1;

        [NinjaScriptProperty]
        [Display(Name = "Min Confidence (HMM)", Order = 4, GroupName = "ML")]
        public double MinConfidence { get; set; } = 0.60; // más trades

        // ===== XGBoost filter (nuevo) =====
        [NinjaScriptProperty]
        [Display(Name = "Use XGBoost Filter", Order = 10, GroupName = "ML")]
        public bool UseXgbFilter { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "XGB Min Prob", Order = 11, GroupName = "ML")]
        public double XgbMinProb { get; set; } = 0.55;

        // =========================
        // Inputs - Trading Control
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Enable Trading (Live Orders)", Order = 1, GroupName = "Trading")]
        public bool EnableTrading { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name = "Arm Long", Order = 2, GroupName = "Trading")]
        public bool ArmLong { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Arm Short", Order = 3, GroupName = "Trading")]
        public bool ArmShort { get; set; } = true;

        // =========================
        // Inputs - Session (optional trade window)
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Use Trade Window (Start/End)", Order = 9, GroupName = "Session")]
        public bool UseTradeWindow { get; set; } = false; // operar siempre

        [NinjaScriptProperty]
        [Display(Name = "Trade Start (HHmmss)", Order = 10, GroupName = "Session")]
        public int TradeStart { get; set; } = 73000;

        [NinjaScriptProperty]
        [Display(Name = "Trade End (HHmmss)", Order = 11, GroupName = "Session")]
        public int TradeEnd { get; set; } = 125000;

        [NinjaScriptProperty]
        [Display(Name = "Flatten Time (HHmmss)", Order = 12, GroupName = "Session")]
        public int FlattenTime { get; set; } = 125900;

        // =========================
        // Inputs - Daily Pause around session close/open
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Use Daily Pause (before end / after start)", Order = 20, GroupName = "Session")]
        public bool UseDailyPause { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Flatten Minutes Before Session End", Order = 21, GroupName = "Session")]
        public int FlattenMinsBeforeEnd { get; set; } = 15;

        [NinjaScriptProperty]
        [Display(Name = "Resume Minutes After Session Start", Order = 22, GroupName = "Session")]
        public int ResumeMinsAfterStart { get; set; } = 15;

        // =========================
        // Inputs - Risk
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Cooldown Bars", Order = 30, GroupName = "Risk")]
        public int CooldownBars { get; set; } = 3;

        [NinjaScriptProperty]
        [Display(Name = "Max Trades per Session", Order = 31, GroupName = "Risk")]
        public int MaxTradesPerSession { get; set; } = 10;

        // =========================
        // Inputs - Orders
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Qty", Order = 40, GroupName = "Orders")]
        public int Qty { get; set; } = 1;

        [NinjaScriptProperty]
        [Display(Name = "Stop Loss (ticks)", Order = 41, GroupName = "Orders")]
        public int StopLossTicks { get; set; } = 40;   // tu default sugerido

        [NinjaScriptProperty]
        [Display(Name = "Take Profit (ticks)", Order = 42, GroupName = "Orders")]
        public int TakeProfitTicks { get; set; } = 80; // tu default sugerido

        // =========================
        // Inputs - Signals
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Trend Breakout Lookback (bars)", Order = 50, GroupName = "Signals")]
        public int TrendLookback { get; set; } = 10;

        [NinjaScriptProperty]
        [Display(Name = "Trend Range Mult (current > avg*mult)", Order = 51, GroupName = "Signals")]
        public double TrendRangeMult { get; set; } = 1.15;

        [NinjaScriptProperty]
        [Display(Name = "Mean EMA Length", Order = 60, GroupName = "Signals")]
        public int MeanEmaLen { get; set; } = 50;

        [NinjaScriptProperty]
        [Display(Name = "Mean ATR Length", Order = 61, GroupName = "Signals")]
        public int MeanAtrLen { get; set; } = 14;

        [NinjaScriptProperty]
        [Display(Name = "Mean Deviation (ATR mult)", Order = 62, GroupName = "Signals")]
        public double MeanAtrMult { get; set; } = 0.8;

        // =========================
        // Visual
        // =========================
        [NinjaScriptProperty]
        [Display(Name = "Plot TP/SL Lines", Order = 80, GroupName = "Visual")]
        public bool PlotLevels { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name = "Level Line Length (bars)", Order = 81, GroupName = "Visual")]
        public int LevelLineBars { get; set; } = 12;

        [NinjaScriptProperty]
        [Display(Name = "Max Plotted Signals", Order = 82, GroupName = "Visual")]
        public int MaxPlottedSignals { get; set; } = 80;

        private int plotCounter = 0;

        // =========================
        // Networking fields
        // =========================
        private TcpClient client;
        private NetworkStream stream;
        private readonly object netLock = new object();
        private StringBuilder recvBuffer = new StringBuilder();

        // =========================
        // Last ML state (HMM)
        // =========================
        private string lastRegime = "NO_TRADE";
        private double lastConf = 0.0;
        private bool lastReject = true;

        // =========================
        // Last ML state (XGB)
        // =========================
        private bool lastXgbPass = true;          // default PASS (compat)
        private double lastXgbProb = 1.0;         // default high
        private string lastXgbLabel = "PASS";     // optional
        private string lastXgbLineSig = "";       // to avoid spam prints

        // =========================
        // Session / risk state
        // =========================
        private int lastEntryBar = -999999;
        private int tradesThisSession = 0;
        private DateTime currentSessionDate = Core.Globals.MinDate;

        // Trading hours session iterator
        private SessionIterator sessionIterator;
        private DateTime sessionBegin = Core.Globals.MinDate;
        private DateTime sessionEnd = Core.Globals.MinDate;
        private DateTime lastPrintedSessionEnd = Core.Globals.MinDate;

        // Indicators
        private EMA emaMean;
        private ATR atrMean;
        private SMA avgRange;
        private Series<double> barRange;

        // =========================
        // NT Lifecycle
        // =========================
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "HFTSpectreStrategy";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 5;
                IsUnmanaged = false;
            }
            else if (State == State.DataLoaded)
            {
                sessionIterator = new SessionIterator(Bars);
                TryConnect();

                emaMean = EMA(MeanEmaLen);
                atrMean = ATR(MeanAtrLen);

                barRange = new Series<double>(this);
                avgRange = SMA(barRange, 20);
            }
            else if (State == State.Configure)
            {
                SetStopLoss(CalculationMode.Ticks, StopLossTicks);
                SetProfitTarget(CalculationMode.Ticks, TakeProfitTicks);
            }
            else if (State == State.Terminated)
            {
                Disconnect();
            }
        }

        // =========================
        // Helpers: Session logic
        // =========================
        private void ResetSessionCountersIfNeeded()
        {
            DateTime d = Time[0].Date;
            if (d != currentSessionDate)
            {
                currentSessionDate = d;
                tradesThisSession = 0;
            }
        }

        private bool InTradeWindow()
        {
            if (!UseTradeWindow)
                return true;

            int t = ToTime(Time[0]);
            return t >= TradeStart && t <= TradeEnd;
        }

        private bool PastFlattenTime()
        {
            if (!UseTradeWindow)
                return false;

            int t = ToTime(Time[0]);
            return t >= FlattenTime;
        }

        private bool CooldownOk()
        {
            return (CurrentBar - lastEntryBar) >= CooldownBars;
        }

        private void UpdateSessionWindow()
        {
            if (sessionIterator == null)
                return;

            sessionIterator.GetNextSession(Time[0], true);
            sessionBegin = sessionIterator.ActualSessionBegin;
            sessionEnd = sessionIterator.ActualSessionEnd;

            if (sessionEnd != lastPrintedSessionEnd)
            {
                lastPrintedSessionEnd = sessionEnd;
                Print($"[SESSION] Begin={sessionBegin:yyyy-MM-dd HH:mm:ss} End={sessionEnd:yyyy-MM-dd HH:mm:ss} Now={Time[0]:yyyy-MM-dd HH:mm:ss}");
            }
        }

        private bool InDailyPauseWindow()
        {
            if (!UseDailyPause)
                return false;

            UpdateSessionWindow();
            if (sessionBegin == Core.Globals.MinDate || sessionEnd == Core.Globals.MinDate)
                return false;

            DateTime pauseStart = sessionEnd.AddMinutes(-FlattenMinsBeforeEnd);
            DateTime pauseEndAtStart = sessionBegin.AddMinutes(ResumeMinsAfterStart);

            bool nearEnd = Time[0] >= pauseStart && Time[0] <= sessionEnd;
            bool nearStart = Time[0] >= sessionBegin && Time[0] <= pauseEndAtStart;

            return nearEnd || nearStart;
        }

        // =========================
        // Visual: plot proposed TP/SL
        // =========================
        private void PlotTradeLevels(string tagBase, double entryPrice, bool isLong)
        {
            if (!PlotLevels) return;

            double sl = isLong
                ? entryPrice - (StopLossTicks * TickSize)
                : entryPrice + (StopLossTicks * TickSize);

            double tp = isLong
                ? entryPrice + (TakeProfitTicks * TickSize)
                : entryPrice - (TakeProfitTicks * TickSize);

            plotCounter++;
            int id = plotCounter;

            string tpTag = $"{tagBase}_TP_{id}";
            string slTag = $"{tagBase}_SL_{id}";

            int len = Math.Max(2, LevelLineBars);

            Draw.Line(this, tpTag, false, len, tp, 0, tp, Brushes.Lime, null, 2);
            Draw.Line(this, slTag, false, len, sl, 0, sl, Brushes.Red, null, 2);

            int killId = id - MaxPlottedSignals;
            if (killId > 0)
            {
                RemoveDrawObject($"{tagBase}_TP_{killId}");
                RemoveDrawObject($"{tagBase}_SL_{killId}");
            }
        }

        // =========================
        // Networking helpers
        // =========================
        private void TryConnect()
        {
            try
            {
                if (client != null && client.Connected && stream != null)
                    return;

                Disconnect();

                client = new TcpClient();
                client.NoDelay = true;
                client.Connect(MlHost, MlPort);
                stream = client.GetStream();

                Print($"[ML] Connected to {MlHost}:{MlPort}");
            }
            catch (Exception e)
            {
                // No spamear: solo debug mínimo
                // Print($"[ML] Connect failed: {e.Message}");
                client = null;
                stream = null;
            }
        }

        private void Disconnect()
        {
            try { stream?.Close(); } catch { }
            try { client?.Close(); } catch { }
            stream = null;
            client = null;
        }

        private string SendAndReceiveLine(string message)
        {
            lock (netLock)
            {
                try
                {
                    if (stream == null)
                        return null;

                    byte[] data = Encoding.UTF8.GetBytes(message);
                    stream.Write(data, 0, data.Length);
                    stream.Flush();

                    // Read until newline
                    while (true)
                    {
                        if (stream.DataAvailable)
                        {
                            byte[] buf = new byte[4096];
                            int n = stream.Read(buf, 0, buf.Length);
                            if (n <= 0) return null;

                            recvBuffer.Append(Encoding.UTF8.GetString(buf, 0, n));

                            int idx = recvBuffer.ToString().IndexOf('\n');
                            if (idx >= 0)
                            {
                                string line = recvBuffer.ToString(0, idx).Trim();
                                recvBuffer.Remove(0, idx + 1);
                                return line;
                            }
                        }
                        else
                        {
                            // evita bloquear duro
                            return null;
                        }
                    }
                }
                catch
                {
                    Disconnect();
                    return null;
                }
            }
        }

        // =========================
        // Minimal JSON extractors (regex)
        // =========================
        private string ExtractJsonString(string json, string key)
        {
            if (string.IsNullOrEmpty(json)) return null;
            var m = Regex.Match(json, $"\"{Regex.Escape(key)}\"\\s*:\\s*\"(.*?)\"");
            return m.Success ? m.Groups[1].Value : null;
        }

        private double ExtractJsonDouble(string json, string key, double defaultValue = double.NaN)
        {
            if (string.IsNullOrEmpty(json)) return defaultValue;
            var m = Regex.Match(json, $"\"{Regex.Escape(key)}\"\\s*:\\s*([-0-9\\.eE]+)");
            if (m.Success && double.TryParse(m.Groups[1].Value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double v))
                return v;
            return defaultValue;
        }

        private bool ExtractJsonBool(string json, string key, bool defaultValue = false)
        {
            if (string.IsNullOrEmpty(json)) return defaultValue;
            var m = Regex.Match(json, $"\"{Regex.Escape(key)}\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
            if (m.Success)
                return m.Groups[1].Value.Equals("true", StringComparison.OrdinalIgnoreCase);
            return defaultValue;
        }

        // =========================
        // Main loop
        // =========================
        protected override void OnBarUpdate()
        {
            if (CurrentBar < 20)
                return;

            ResetSessionCountersIfNeeded();

            // 0) Daily pause around session close/open (lo que ya te funciona)
            if (InDailyPauseWindow())
            {
                Print($"[SESSION] DailyPause active. Blocking trades. Now={Time[0]:yyyy-MM-dd HH:mm:ss} Begin={sessionBegin:HH:mm:ss} End={sessionEnd:HH:mm:ss}");

                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    ExitLong("DailyPauseFlatten", "");
                    ExitShort("DailyPauseFlatten", "");
                }
                return;
            }

            // 1) Optional manual flatten time (only if UseTradeWindow)
            if (PastFlattenTime())
            {
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    ExitLong("Flatten", "");
                    ExitShort("Flatten", "");
                }
                return;
            }

            // update range series for Trend filter
            barRange[0] = High[0] - Low[0];

            // Ensure connection
            if (stream == null || client == null || !client.Connected)
            {
                TryConnect();
                if (stream == null) return;
            }

            // Send once per bar (stable ML)
            if (!IsFirstTickOfBar)
                return;

            if (SendEveryNBars > 1 && (CurrentBar % SendEveryNBars) != 0)
                return;

            // Build message (lo que ya usas)
            double bid = GetCurrentBid();
            double ask = GetCurrentAsk();

            string ts = DateTime.UtcNow.ToString("o");
            string barTime = Time[0].ToString("o");

            string msg =
                "{" +
                $"\"ts\":\"{ts}\"," +
                $"\"symbol\":\"{Instrument.FullName}\"," +
                "\"bar\":{" +
                $"\"time\":\"{barTime}\"," +
                $"\"open\":{Open[0].ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"high\":{High[0].ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"low\":{Low[0].ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"close\":{Close[0].ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"volume\":{Volume[0]}" +
                "}," +
                $"\"bid\":{bid.ToString(System.Globalization.CultureInfo.InvariantCulture)}," +
                $"\"ask\":{ask.ToString(System.Globalization.CultureInfo.InvariantCulture)}" +
                "}\n";

            string resp = SendAndReceiveLine(msg);
            if (string.IsNullOrEmpty(resp))
                return;

            // ===== Parse HMM =====
            string regime = ExtractJsonString(resp, "regime");
            double conf = ExtractJsonDouble(resp, "conf", double.NaN);
            bool reject = ExtractJsonBool(resp, "reject", true);

            if (!string.IsNullOrEmpty(regime))
            {
                lastRegime = regime;
                lastConf = double.IsNaN(conf) ? lastConf : conf;
                lastReject = reject;

                if (!reject)
                    Print($"From ML(HMM): {lastRegime} (conf={lastConf:0.00})");
                else
                    Print($"From ML(HMM): REJECT -> {lastRegime} (conf={lastConf:0.00})");
            }

            // ===== Parse XGB (si viene; si no viene, PASS por compatibilidad) =====
            // Esperado en respuesta python:
            // "xgb_pass": true/false
            // "xgb_prob": 0..1
            // "xgb_label": "TRADE"/"NO_TRADE" (opcional)
            bool xgbPass = ExtractJsonBool(resp, "xgb_pass", true);
            double xgbProb = ExtractJsonDouble(resp, "xgb_prob", 1.0);
            string xgbLabel = ExtractJsonString(resp, "xgb_label");
            if (string.IsNullOrEmpty(xgbLabel)) xgbLabel = xgbPass ? "PASS" : "BLOCK";

            lastXgbPass = xgbPass;
            lastXgbProb = xgbProb;
            lastXgbLabel = xgbLabel;

            // evita spam: imprime solo si cambia la firma
            string sig = $"{lastXgbPass}_{lastXgbProb:0.000}_{lastXgbLabel}";
            if (sig != lastXgbLineSig)
            {
                lastXgbLineSig = sig;
                Print($"From ML(XGB): pass={lastXgbPass} prob={lastXgbProb:0.00} label={lastXgbLabel}");
            }

            // ====== Signal intent by regime (igual que venías, sin “vela reversa” forzada) ======
            bool wantLong = false;
            bool wantShort = false;

            // TRENDING signals: breakout + range expansion
            if (lastRegime == "TRENDING")
            {
                if (ArmLong)
                {
                    double hh = MAX(High, TrendLookback)[1];
                    double curRange = High[0] - Low[0];
                    double ar = avgRange[0];

                    if (Close[0] > hh && curRange > ar * TrendRangeMult)
                        wantLong = true;
                }

                if (ArmShort)
                {
                    double ll = MIN(Low, TrendLookback)[1];
                    double curRange = High[0] - Low[0];
                    double ar = avgRange[0];

                    if (Close[0] < ll && curRange > ar * TrendRangeMult)
                        wantShort = true;
                }
            }

            // MEAN_REVERTING signals: re-entry to EMA band (más frecuente)
            if (lastRegime == "MEAN_REVERTING")
            {
                double mean = emaMean[0];
                double dev = atrMean[0] * MeanAtrMult;

                double lower = mean - dev;
                double upper = mean + dev;

                if (ArmLong && Close[1] < lower && Close[0] >= lower)
                    wantLong = true;

                if (ArmShort && Close[1] > upper && Close[0] <= upper)
                    wantShort = true;
            }

            // ===== Hard filters =====
            bool xgbOk = (!UseXgbFilter) || (lastXgbPass && lastXgbProb >= XgbMinProb);

            bool canTrade =
                InTradeWindow()
                && CooldownOk()
                && tradesThisSession < MaxTradesPerSession
                && !lastReject
                && lastConf >= MinConfidence
                && xgbOk;

            // Plot potentials (aunque EnableTrading esté apagado)
            if (canTrade && Position.MarketPosition == MarketPosition.Flat)
            {
                if (wantLong)
                    PlotTradeLevels("SIG_LONG", Close[0], true);
                else if (wantShort)
                    PlotTradeLevels("SIG_SHORT", Close[0], false);
            }

            // ===== Execution (only if enabled) =====
            if (EnableTrading && canTrade && Position.MarketPosition == MarketPosition.Flat)
            {
                if (wantLong)
                {
                    EnterLong(Qty, "ML_Long");
                    lastEntryBar = CurrentBar;
                    tradesThisSession++;
                }
                else if (wantShort)
                {
                    EnterShort(Qty, "ML_Short");
                    lastEntryBar = CurrentBar;
                    tradesThisSession++;
                }
            }
        }
    }
}