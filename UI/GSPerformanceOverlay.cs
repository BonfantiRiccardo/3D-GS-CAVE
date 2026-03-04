using System.Text;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Lightweight UI performance overlay for Gaussian Splatting.
    /// Displays FPS, frame time, splat counts, GPU memory estimates and other
    /// useful diagnostics. Stats refresh once per second to avoid noise.
    /// </summary>
    [DefaultExecutionOrder(10000)]
    [DisallowMultipleComponent]
    public class GSPerformanceOverlay : MonoBehaviour
    {
        //  Inspector settings

        [Header("Display")]
        [Tooltip("How often (in seconds) the stats text is rebuilt.")]
        [Range(0.1f, 5f)]
        public float refreshInterval = 1.0f;

        [Tooltip("Render the overlay only in the left eye area (left half of a side-by-side stereo view).")]
        public bool leftEyeOnly = false;

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


       
        public enum ScreenCorner
        {
            TopLeft,
            TopRight,
            BottomLeft,
            BottomRight
        }

        //  Private state 

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

        //  Unity lifecycle 

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsureOverlayExists()
        {
            if (!Application.isPlaying) return;

            // Bootstrap in scenes where the overlay component was not added.
            var existing = FindObjectsByType<GSPerformanceOverlay>(
                FindObjectsInactive.Include,
                FindObjectsSortMode.None);
            if (existing != null && existing.Length > 0) return;

            var go = new GameObject(nameof(GSPerformanceOverlay));
            go.AddComponent<GSPerformanceOverlay>();
        }

        void Start()
        {
            visible = showOnStart;
            RebuildStatsText();
        }

        void Update()
        {
            HandleToggleInput();

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
            HandleToggleKeyEvent();

            if (!visible) return;

            if (string.IsNullOrEmpty(statsText))
                RebuildStatsText();

            EnsureStyles();
            GUI.depth = -10000;

            Rect viewportRect = GetViewportRect();
            GUI.BeginGroup(viewportRect);

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
                    x = viewportRect.width - boxW - padding; y = padding;
                    break;
                case ScreenCorner.BottomLeft:
                    x = padding; y = viewportRect.height - boxH - padding;
                    break;
                case ScreenCorner.BottomRight:
                    x = viewportRect.width - boxW - padding; y = viewportRect.height - boxH - padding;
                    break;
            }

            // Clamp to viewport in case very long lines exceed expected width.
            x = Mathf.Clamp(x, 0f, Mathf.Max(0f, viewportRect.width - boxW));
            y = Mathf.Clamp(y, 0f, Mathf.Max(0f, viewportRect.height - boxH));

            Rect boxRect = new Rect(x, y, boxW, boxH);
            GUI.Box(boxRect, GUIContent.none, boxStyle);
            Rect labelRect = new Rect(x + padding, y + padding, size.x, size.y);
            GUI.Label(labelRect, content, labelStyle);

            GUI.EndGroup();
        }

        private void HandleToggleInput()
        {
#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(toggleKey))
                visible = !visible;
#endif
        }

        private void HandleToggleKeyEvent()
        {
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
            var evt = Event.current;
            if (evt != null && evt.type == EventType.KeyDown && evt.keyCode == toggleKey)
            {
                visible = !visible;
                evt.Use();
            }
#endif
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

            // Chunked Gaussian Splatting stats
            var components = ChunkedGSManager.Components;
            int componentCount = components?.Count ?? 0;
            int totalSplats = 0;
            int totalActiveSplats = 0;
            int totalLoadedChunks = 0;
            int totalChunks = 0;
            long totalMemoryEstimate = 0;
            int maxSHBands = 0;
            int totalLoads = 0;
            int totalEvicts = 0;
            int totalPending = 0;

            for (int i = 0; i < componentCount; i++)
            {
                var comp = components[i];
                if (comp == null || comp.asset == null) continue;

                totalSplats += comp.asset.totalSplatCount;
                totalActiveSplats += comp.ActiveSplatCount;
                totalLoadedChunks += comp.LoadedChunkCount;
                totalChunks += comp.TotalChunkCount;
                totalMemoryEstimate += comp.EstimatedGPUMemoryBytes;
                totalLoads += comp.LoadCount;
                totalEvicts += comp.EvictCount;
                totalPending += comp.PendingReadCount;

                int bands = comp.ShBandsNumber;
                if (bands > maxSHBands) maxSHBands = bands;
            }

            sb.AppendLine($"GS Objects: {componentCount}");
            sb.AppendLine($"Chunks (loaded/total): {totalLoadedChunks} / {totalChunks}");
            sb.AppendLine($"Splats (active/total): {FormatCount(totalActiveSplats)} / {FormatCount(totalSplats)}");
            sb.AppendLine($"Streaming: {totalLoads} loads  {totalEvicts} evicts  {totalPending} pending");
            sb.AppendLine($"SH Bands: {maxSHBands}");
            sb.AppendLine($"GPU Alloc (est.): {FormatBytes(totalMemoryEstimate)}");

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

        /// <summary>
        /// Returns the GUI viewport where the overlay should be drawn.
        /// </summary>
        private Rect GetViewportRect()
        {
            if (!leftEyeOnly)
                return new Rect(0f, 0f, Screen.width, Screen.height);

            return new Rect(0f, 0f, Screen.width * 0.5f, Screen.height);
        }
    }
}
