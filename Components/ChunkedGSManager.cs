using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Manager class to keep track of all ChunkedGSComponent instances in the scene.
    /// </summary>
    public class ChunkedGSManager : MonoBehaviour
    {
        private static readonly List<ChunkedGSComponent> components = new List<ChunkedGSComponent>();

        /// <summary>
        /// Registers a ChunkedGSComponent with the manager.
        /// </summary>
        public static void Register(ChunkedGSComponent component)
        {
            if (!components.Contains(component))
            {
                components.Add(component);
            }
        }

        /// <summary>
        /// Unregisters a ChunkedGSComponent from the manager.
        /// </summary>
        public static void Unregister(ChunkedGSComponent component)
        {
            if (components.Contains(component))
            {
                components.Remove(component);
            }
        }

        /// <summary>
        /// Gets a read only list of all registered ChunkedGSComponent instances.
        /// </summary>
        public static IReadOnlyList<ChunkedGSComponent> Components => components;
    }
}
