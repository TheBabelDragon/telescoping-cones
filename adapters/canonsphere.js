/** Canonsphere adapter. Core never depends on Canonsphere.
 *  Sphere → stereograph → nested cone caps. Witness-only.
 */
import { stereographic, capRadiusOnPlane, TelescopingCone, PROTOCOL_VERSION } from "../js/index.js";

export function fromCanonicalState(state, axis, aperture = 0.5) {
  const points = state.points ?? [];
  const uv = stereographic(points, axis);
  const cone = new TelescopingCone({ center: axis, radius: aperture * 1.15, aperture });
  const collapsed = cone.collapse(state.features ?? points, points);
  return {
    version: PROTOCOL_VERSION,
    source: { kind: "canonsphere", id: state.source?.id },
    stereograph: uv,
    cap: capRadiusOnPlane(aperture),
    collapsed: collapsed.state,
    entropy: collapsed.entropy,
  };
}
