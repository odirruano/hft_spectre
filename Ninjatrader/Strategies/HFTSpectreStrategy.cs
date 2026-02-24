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
using NinjaTrader.Gui;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class HFTSpectreStrategy : Strategy
    {
        // =========================
        // Inputs - ML
        // =========================
        [NinjaScriptProperty]
        [Display(Name="ML Host", Order=1, GroupName="ML")]
        public string MlHost { get; set; } = "127.0.0.1";

        [NinjaScriptProperty]
        [Display(Name="ML Port", Order=2, GroupName="ML")]
        public int MlPort { get; set; } = 5555;

        [NinjaScriptProperty]
        [Display(Name="Send Every N Bars", Order=3, GroupName="ML")]
        public int SendEveryNBars { get; set; } = 1;

        [NinjaScriptProperty]
        [Display(Name="Min Confidence", Order=4, GroupName="ML")]
        public double MinConfidence { get; set; } = 0.60; // más trades

        // =========================
        // Inputs - Trading Control
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Enable Trading (Live Orders)", Order=1, GroupName="Trading")]
        public bool EnableTrading { get; set; } = false;

        [NinjaScriptProperty]
        [Display(Name="Arm Long", Order=2, GroupName="Trading")]
        public bool ArmLong { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name="Arm Short", Order=3, GroupName="Trading")]
        public bool ArmShort { get; set; } = true;

        // =========================
        // Inputs - Session (optional trade window)
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Use Trade Window (Start/End)", Order=9, GroupName="Session")]
        public bool UseTradeWindow { get; set; } = false; // operar siempre

        [NinjaScriptProperty]
        [Display(Name="Trade Start (HHmmss)", Order=10, GroupName="Session")]
        public int TradeStart { get; set; } = 73000;

        [NinjaScriptProperty]
        [Display(Name="Trade End (HHmmss)", Order=11, GroupName="Session")]
        public int TradeEnd { get; set; } = 125000;

        [NinjaScriptProperty]
        [Display(Name="Flatten Time (HHmmss)", Order=12, GroupName="Session")]
        public int FlattenTime { get; set; } = 125900;

        // =========================
        // Inputs - Daily Pause around session close/open
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Use Daily Pause (before end / after start)", Order=20, GroupName="Session")]
        public bool UseDailyPause { get; set; } = true;

        [NinjaScriptProperty]
        [Display(Name="Flatten Minutes Before Session End", Order=21, GroupName="Session")]
        public int FlattenMinsBeforeEnd { get; set; } = 15;

        [NinjaScriptProperty]
        [Display(Name="Resume Minutes After Session Start", Order=22, GroupName="Session")]
        public int ResumeMinsAfterStart { get; set; } = 15;

        // =========================
        // Inputs - Risk
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Cooldown Bars", Order=30, GroupName="Risk")]
        public int CooldownBars { get; set; } = 2; // más fluido

        [NinjaScriptProperty]
        [Display(Name="Max Trades per Session", Order=31, GroupName="Risk")]
        public int MaxTradesPerSession { get; set; } = 50;

        // =========================
        // Inputs - Orders (NQ defaults)
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Qty", Order=40, GroupName="Orders")]
        public int Qty { get; set; } = 1;

        [NinjaScriptProperty]
        [Display(Name="Max Qty", Order=41, GroupName="Orders")]
        public int MaxQty { get; set; } = 3;

        [NinjaScriptProperty]
        [Display(Name="Stop Loss (ticks)", Order=42, GroupName="Orders")]
        public int StopLossTicks { get; set; } = 80;   // NQ ~ 20 pts (4 ticks/pt => 80 ticks)

        [NinjaScriptProperty]
        [Display(Name="Take Profit (ticks)", Order=43, GroupName="Orders")]
        public int TakeProfitTicks { get; set; } = 120; // NQ ~ 30 pts

        // =========================
        // Inputs - Signals
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Trend Breakout Lookback (bars)", Order=50, GroupName="Signals")]
        public int TrendLookback { get; set; } = 5; // más entradas

        [NinjaScriptProperty]
        [Display(Name="Mean EMA Length", Order=60, GroupName="Signals")]
        public int MeanEmaLen { get; set; } = 50;

        [NinjaScriptProperty]
        [Display(Name="Mean ATR Length", Order=61, GroupName="Signals")]
        public int MeanAtrLen { get; set; } = 14;

        [NinjaScriptProperty]
        [Display(Name="Mean Deviation (ATR mult)", Order=62, GroupName="Signals")]
        public double MeanAtrMult { get; set; } = 0.6; // más entradas

        // =========================
        // Visual
        // =========================
        [NinjaScriptProperty]
        [Display(Name="Plot TP/SL Lines", Order=80, GroupName="Visual")]
        public bool PlotLevels { get; set; } = true;
		
		[NinjaScriptProperty]
		[Display(Name="Level Line Length (bars)", Order=81, GroupName="Visual")]
		public int LevelLineBars { get; set; } = 12;
		
		[NinjaScriptProperty]
		[Display(Name="Max Plotted Signals", Order=82, GroupName="Visual")]
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
        // Last ML state
        // =========================
        private string lastRegime = "NO_TRADE";
        private double lastConf = 0.0;
        private bool lastReject = true;

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

        // =========================
        // Trading hours session window
        // =========================
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
		
		    // contador incremental para tags únicos
		    plotCounter++;
		    int id = plotCounter;
		
		    string tpTag = $"{tagBase}_TP_{id}";
		    string slTag = $"{tagBase}_SL_{id}";
		
		    int len = Math.Max(2, LevelLineBars); // mínimo 2 barras
		
		    // Segmentos cortos: desde barra actual (barsAgo=0) hasta len barras atrás (barsAgo=len)
		    Draw.Line(this, tpTag, false, len, tp, 0, tp, Brushes.Lime, DashStyleHelper.Solid, 2);
			Draw.Line(this, slTag, false, len, sl, 0, sl, Brushes.Red,  DashStyleHelper.Solid, 2);
		
		    // Limpieza: borrar dibujos viejos para evitar “desastre”
		    int killId = id - MaxPlottedSignals;
		    if (killId > 0)
		    {
		        RemoveDrawObject($"{tagBase}_TP_{killId}");
		        RemoveDrawObject($"{tagBase}_SL_{killId}");
		    }
		}

        // =========================
        // Networking
        // =========================
        private void TryConnect()
        {
            try
            {
                client = new TcpClient();
                client.NoDelay = true;
                client.Connect(MlHost, MlPort);
                stream = client.GetStream();
                Print($"[ML] Connected to {MlHost}:{MlPort}");
            }
            catch (Exception ex)
            {
                Print($"[ML] Connect failed: {ex.Message}");
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
                    if (stream == null || client == null || !client.Connected)
                        return null;

                    byte[] data = Encoding.UTF8.GetBytes(message);
                    stream.Write(data, 0, data.Length);

                    byte[] buf = new byte[4096];
                    int bytes = stream.Read(buf, 0, buf.Length);
                    if (bytes <= 0) return null;

                    string chunk = Encoding.UTF8.GetString(buf, 0, bytes);
                    recvBuffer.Append(chunk);

                    int nl = recvBuffer.ToString().IndexOf('\n');
                    if (nl >= 0)
                    {
                        string line = recvBuffer.ToString(0, nl).Trim();
                        recvBuffer.Remove(0, nl + 1);
                        return line;
                    }
                }
                catch (Exception ex)
                {
                    Print($"[ML] IO error: {ex.Message}");
                    Disconnect();
                }

                return null;
            }
        }

        // =========================
        // Simple JSON extractors
        // =========================
        private string ExtractJsonString(string json, string key)
        {
            try
            {
                var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*\"([^\"]*)\"");
                return m.Success ? m.Groups[1].Value : null;
            }
            catch { return null; }
        }

        private double ExtractJsonDouble(string json, string key)
        {
            try
            {
                var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*([0-9\\.]+)");
                if (!m.Success) return 0.0;

                double v;
                return double.TryParse(m.Groups[1].Value, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out v) ? v : 0.0;
            }
            catch { return 0.0; }
        }

        private bool ExtractJsonBool(string json, string key)
        {
            try
            {
                var m = Regex.Match(json, $"\"{key}\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
                if (!m.Success) return true;
                return m.Groups[1].Value.ToLowerInvariant() == "true";
            }
            catch { return true; }
        }

        // =========================
        // Main loop
        // =========================
        protected override void OnBarUpdate()
        {
            if (CurrentBar < 50)
                return;

            ResetSessionCountersIfNeeded();

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

            // 2) Daily pause window
            if (InDailyPauseWindow())
            {
                Print($"[SESSION] Daily pause window ACTIVE. Blocking trades. Now={Time[0]:yyyy-MM-dd HH:mm:ss} Begin={sessionBegin:HH:mm:ss} End={sessionEnd:HH:mm:ss}");

                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    ExitLong("DailyPauseFlatten", "");
                    ExitShort("DailyPauseFlatten", "");
                }
                return;
            }

            // Ensure connection
            if (stream == null || client == null || !client.Connected)
            {
                TryConnect();
                if (stream == null) return;
            }

            // Send once per bar
            if (!IsFirstTickOfBar)
                return;

            if (SendEveryNBars > 1 && (CurrentBar % SendEveryNBars) != 0)
                return;

            // Build message
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
                $"\"open\":{Open[0]}," +
                $"\"high\":{High[0]}," +
                $"\"low\":{Low[0]}," +
                $"\"close\":{Close[0]}," +
                $"\"volume\":{Volume[0]}" +
                "}," +
                $"\"bid\":{bid}," +
                $"\"ask\":{ask}" +
                "}\n";

            string resp = SendAndReceiveLine(msg);
            if (string.IsNullOrEmpty(resp))
                return;

            // Parse response
            string regime = ExtractJsonString(resp, "regime");
            double conf = ExtractJsonDouble(resp, "conf");
            bool reject = ExtractJsonBool(resp, "reject");

            if (!string.IsNullOrEmpty(regime))
            {
                lastRegime = regime;
                lastConf = conf;
                lastReject = reject;

                if (!reject)
                    Print($"From ML: {lastRegime} (conf={lastConf:0.00})");
                else
                    Print($"From ML: REJECT -> {lastRegime} (conf={lastConf:0.00})");
            }

            // ====== Signals ======
            bool wantLong = false;
            bool wantShort = false;

            // TRENDING: breakout by High/Low with 1 tick confirmation
            if (lastRegime == "TRENDING")
            {
                double hh = MAX(High, TrendLookback)[1];
                double ll = MIN(Low, TrendLookback)[1];
                double oneTick = TickSize;

                if (ArmLong && High[0] >= hh + oneTick)
                    wantLong = true;

                if (ArmShort && Low[0] <= ll - oneTick)
                    wantShort = true;
            }

            // MEAN_REVERTING: re-entry to EMA band (more frequent)
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

            // Plot proposed levels (like original system)
            if (wantLong)  PlotTradeLevels("SIG_LONG", Close[0], true);
            if (wantShort) PlotTradeLevels("SIG_SHORT", Close[0], false);

            // Hard filters
            bool canTrade = InTradeWindow()
                            && CooldownOk()
                            && tradesThisSession < MaxTradesPerSession
                            && !lastReject
                            && lastConf >= MinConfidence;

            int qtyToUse = Math.Max(1, Math.Min(Qty, MaxQty));

            // Execute
            if (EnableTrading && canTrade && Position.MarketPosition == MarketPosition.Flat)
            {
                if (wantLong)
                {
                    EnterLong(qtyToUse, "ML_Long");
                    lastEntryBar = CurrentBar;
                    tradesThisSession++;
                }
                else if (wantShort)
                {
                    EnterShort(qtyToUse, "ML_Short");
                    lastEntryBar = CurrentBar;
                    tradesThisSession++;
                }
            }
        }
    }
}