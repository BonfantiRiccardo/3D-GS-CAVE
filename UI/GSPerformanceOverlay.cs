using System.Text;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Lightweight IMGUI performance overlay for Gaussian Splatting.
    /// Displays FPS, frame time, splat counts, GPU memory estimates and other
    /// useful diagnostics. Stats refresh once per second to avoid noise.
    ///
    /// Usage: Attach this component to any GameObject in the scene (e.g. Main Camera
    /// or a dedicated "PerformanceStats" object).
    /// </summary>
    public class GSPerformanceOverlay : MonoBehaviour
    {
        // ── Inspector settings ──────────────────────────────────────────

        [Header("Display")]
        [Tooltip("How often (in seconds) the stats text is rebuilt.")]
        [Range(0.1f, 5f)]
        public float refreshInterval = 1.0f;

        [Tooltip("Corner of the screen where the overlay is drawn.")]
        public ScreenCorner anchor = ScreenCorner.TopLeft;

        [Tooltip("Toggle the overlay on/off at runtime with this key.")]
        public KeyCode toggleKey = KeyCode.F1;

        [Tooltip("Show the overlay on start.")]
        public bool showOnStart = true;

        [Header("Appearance")]
        [Tooltip("Font size in pixels.")]
        [Range(10, 32)]
        public int fontSize = 14;

        [Tooltip("Text color.")]
        public Color textColor = Color.white;

        [Tooltip("Background tint (set alpha for transparency).")]
        public Color backgroundColor = new Color(0f, 0f, 0f, 0.65f);

        [Tooltip("Padding inside the overlay box (pixels).")]
        public int padding = 8;

        // ── Enums ───────────────────────────────────────────────────────

        public enum ScreenCorner
        {
            TopLeft,
            TopRight,
            BottomLeft,
            BottomRight
        }

        // ── Private state ───────────────────────────────────────────────

        private bool visible;
        private float timer;
        private int frameCount;
        private float fps;
        private float frameTimeMs;
        private string statsText = string.Empty;
        private GUIStyle boxStyle;
        private GUIStyle labelStyle;
        private Texture2D bgTexture;

        // Reusable string builder to avoid GC allocs every refresh
        private readonly StringBuilder sb = new StringBuilder(512);

        // ── Unity lifecycle ─────────────────────────────────────────────

        void Start()
        {
            visible = showOnStart;
        }

        void Update()
        {
            // Toggle visibility
            if (Input.GetKeyDown(toggleKey))
                visible = !visible;

            if (!visible) return;

            frameCount++;
            timer += Time.unscaledDeltaTime;

            if (timer >= refreshInterval)
            {
                fps = frameCount / timer;
                frameTimeMs = (timer / frameCount) * 1000f;
                frameCount = 0;
                timer = 0f;
                RebuildStatsText();
            }
        }

        void OnGUI()
        {
            if (!visible || string.IsNullOrEmpty(statsText)) return;

            EnsureStyles();

            // Measure text and position the box
            GUIContent content = new GUIContent(statsText);
            Vector2 size = labelStyle.CalcSize(content);
            float boxW = size.x + padding * 2;
            float boxH = size.y + padding * 2;

            float x = 0f, y = 0f;
            switch (anchor)
            {
                case ScreenCorner.TopLeft:
                    x = padding; y = padding;
                    break;
                case ScreenCorner.TopRight:
                    x = Screen.width - boxW - padding; y = padding;
                    break;
                case ScreenCorner.BottomLeft:
                    x = padding; y = Screen.height - boxH - padding;
                    break;
                case ScreenCorner.BottomRight:
                    x = Screen.width - boxW - padding; y = Screen.height - boxH - padding;
                    break;
            }

            Rect boxRect = new Rect(x, y, boxW, boxH);
            GUI.Box(boxRect, GUIContent.none, boxStyle);
            Rect labelRect = new Rect(x + padding, y + padding, size.x, size.y);
            GUI.Label(labelRect, content, labelStyle);
        }

        void OnDestroy()
        {
            if (bgTexture != null)
            {
                Destroy(bgTexture);
                bgTexture = null;
            }
        }


        // Helpers

        /// <summary>
        /// Rebuilds the multi-line stats string from current data.
        /// </summary>
        private void RebuildStatsText()
        {
            sb.Clear();

            // FPS & frame time
            sb.AppendLine($"FPS: {fps:F1}  ({frameTimeMs:F1} ms)");

            // Gaussian Splatting stats
            var components = GSManager.Components;
            int componentCount = components?.Count ?? 0;
            int totalSplats = 0;
            int totalActiveSplats = 0;
            long totalMemoryEstimate = 0;
            int maxSHBands = 0;

            for (int i = 0; i < componentCount; i++)
            {
                var comp = components[i];
                if (comp == null || comp.gsAsset == null) continue;

                totalSplats += comp.gsAsset.splatCount;
                totalActiveSplats += comp.ActiveSplatCount;
                totalMemoryEstimate += comp.gsAsset.EstimatedTotalDataSize;
                int bands = comp.ShBandsNumber;
                if (bands > maxSHBands) maxSHBands = bands;
            }

            sb.AppendLine($"GS Objects: {componentCount}");
            sb.AppendLine($"Splats (active/total): {FormatCount(totalActiveSplats)} / {FormatCount(totalSplats)}");
            sb.AppendLine($"SH Bands: {maxSHBands}");
            sb.AppendLine($"GPU Data (est.): {FormatBytes(totalMemoryEstimate)}");

            // System info
            sb.AppendLine($"Resolution: {Screen.width}x{Screen.height}");
            sb.Append($"GPU: {SystemInfo.graphicsDeviceName}");

            statsText = sb.ToString();
        }

        /// <summary>
        /// Format a large number with K/M suffixes for readability.
        /// </summary>
        private static string FormatCount(int count)
        {
            if (count >= 1_000_000)
                return $"{count / 1_000_000f:F2}M";
            if (count >= 1_000)
                return $"{count / 1_000f:F1}K";
            return count.ToString();
        }

        /// <summary>
        /// Format bytes into a human-readable string (KB, MB, GB).
        /// </summary>
        private static string FormatBytes(long bytes)
        {
            if (bytes >= 1L << 30) return $"{bytes / (double)(1L << 30):F1} GB";
            if (bytes >= 1L << 20) return $"{bytes / (double)(1L << 20):F1} MB";
            if (bytes >= 1L << 10) return $"{bytes / (double)(1L << 10):F1} KB";
            return $"{bytes} B";
        }

        /// <summary>
        /// Lazily create GUIStyles so we only allocate once.
        /// </summary>
        private void EnsureStyles()
        {
            if (labelStyle != null) return;

            // Background texture
            bgTexture = new Texture2D(1, 1);
            bgTexture.SetPixel(0, 0, backgroundColor);
            bgTexture.Apply();

            boxStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = bgTexture }
            };

            labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = this.fontSize,
                fontStyle = FontStyle.Bold,
                normal = { textColor = textColor },
                richText = false,
                wordWrap = false
            };
        }
    }
}
