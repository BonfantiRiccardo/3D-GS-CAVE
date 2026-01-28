using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting
{
    public class GSManager : MonoBehaviour
    {
        private static readonly List<GSComponent> components = new List<GSComponent>();

        /**
        * Registers a GSComponent with the manager.
        */
        public static void Register(GSComponent component)
        {
            if (!components.Contains(component))
            {
                components.Add(component);
            }
        }

        /**
        * Unregisters a GSComponent from the manager.
        */
        public static void Unregister(GSComponent component)
        {
            if (components.Contains(component))
            {
                components.Remove(component);
            }
        }

        public static IReadOnlyList<GSComponent> Components => components;      // Expose as read-only list of components

/*
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
*/
    }
}
