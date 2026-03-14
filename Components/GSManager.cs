using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting
{
    /// <summary>
    /// Manager class to keep track of all GSComponent instances in the scene.
    /// </summary>
    public class GSManager : MonoBehaviour
    {
        private static readonly List<GSComponent> components = new List<GSComponent>();

        /// <summary>
        /// Registers a GSComponent with the manager.
        /// </summary>
        public static void Register(GSComponent component)
        {
            if (!components.Contains(component))
            {
                components.Add(component);
            }
        }

        /// <summary>
        /// Unregisters a GSComponent from the manager.
        /// </summary>
        public static void Unregister(GSComponent component)
        {
            if (components.Contains(component))
            {
                components.Remove(component);
            }
        }

        /// <summary>
        /// Gets a read-only list of all registered GSComponent instances.
        /// </summary>
        public static IReadOnlyList<GSComponent> Components => components;      // Expose as read-only list of components

    }
}
