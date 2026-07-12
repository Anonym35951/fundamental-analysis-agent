import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { createGlossyBlackMatcap } from "./glossyMatcap";
import { useThemeMode } from "../ui/useThemeMode";
import { FLY_IN_DURATION, FLY_OUT_DURATION } from "./introTiming";

// Ring band geometry: a real finger-ring band has a FLAT, wide surface with
// a thin edge — not a round wire/tube cross-section like a torus. We build it
// by revolving a thin, chamfered rectangle (a LatheGeometry) instead of using
// a circular torus tube, so the broad polished band face is visible and the
// rim reads as a thin edge with a bright specular line (from the chamfers).
const RING_RADIUS = 2.7;
const BAND_WIDTH = 1.15; // visible broad face (along the finger axis)
const BAND_THICKNESS = 0.06; // radial wall = the thin edge
const BAND_CHAMFER = 0.03; // small bevel so edges catch a crisp light line
const ARC = Math.PI;
const LATHE_SEGMENTS = 200;

function buildBandProfile(): THREE.Vector2[] {
  const ro = RING_RADIUS + BAND_THICKNESS / 2;
  const ri = RING_RADIUS - BAND_THICKNESS / 2;
  const wt = BAND_WIDTH / 2;
  const c = BAND_CHAMFER;
  // Rounded-rectangle cross-section (x = radius from axis, y = band width):
  // broad outer/inner walls + thin top/bottom edges, corners chamfered.
  return [
    new THREE.Vector2(ri + c, -wt),
    new THREE.Vector2(ro - c, -wt),
    new THREE.Vector2(ro, -wt + c),
    new THREE.Vector2(ro, wt - c),
    new THREE.Vector2(ro - c, wt),
    new THREE.Vector2(ri + c, wt),
    new THREE.Vector2(ri, wt - c),
    new THREE.Vector2(ri, -wt + c),
    new THREE.Vector2(ri + c, -wt),
  ];
}

const BAND_PROFILE = buildBandProfile();

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function easeInCubic(t: number): number {
  return t * t * t;
}

export type IntroPhase = "idle" | "in" | "hold" | "out";

type RingProps = {
  matcap: THREE.CanvasTexture;
  phase: IntroPhase;
  phaseStartedAtRef: { current: number };
  start: { x: number; y: number; z: number };
  dock: { x: number; y: number; z: number };
  rotation: { x: number; y: number; z: number };
};

function RingHalf({ matcap, phase, phaseStartedAtRef, start, dock, rotation }: RingProps) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    const group = groupRef.current;
    if (!group) return;

    const elapsed = (performance.now() - phaseStartedAtRef.current) / 1000;

    let t: number;
    if (phase === "idle") {
      t = 0;
    } else if (phase === "in") {
      t = easeOutCubic(Math.min(1, Math.max(0, elapsed / FLY_IN_DURATION)));
    } else if (phase === "hold") {
      t = 1;
    } else {
      // "out": mirrored motion back to the start position.
      t = 1 - easeInCubic(Math.min(1, Math.max(0, elapsed / FLY_OUT_DURATION)));
    }

    group.position.set(
      start.x + (dock.x - start.x) * t,
      start.y + (dock.y - start.y) * t,
      start.z + (dock.z - start.z) * t
    );
  });

  return (
    <group ref={groupRef} position={[start.x, start.y, start.z]}>
      <group rotation={[rotation.x, rotation.y, rotation.z]}>
        {/* LatheGeometry revolves around Y; base-rotate so the ring lies in
            the XY plane (hole facing the camera, like the old torus did). */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <latheGeometry args={[BAND_PROFILE, LATHE_SEGMENTS, 0, ARC]} />
          <meshMatcapMaterial matcap={matcap} side={THREE.DoubleSide} />
        </mesh>
      </group>
    </group>
  );
}

type RingsSceneProps = {
  phase: IntroPhase;
};

function SceneContents({ phase }: RingsSceneProps) {
  const { mode } = useThemeMode();
  const matcap = useMemo(() => createGlossyBlackMatcap(), []);
  // Reset via useEffect (React commit phase), not useFrame (render loop):
  // effects for all components flush before the next animation frame, so
  // every RingHalf's useFrame is guaranteed to see the updated timestamp by
  // the time it reads it. Doing this in useFrame instead raced against each
  // RingHalf's own useFrame subscription — on the exact frame `phase`
  // changed, a ring could read the *previous* phase's stale timestamp,
  // computing a huge `elapsed` and snapping to t=0 for one frame before
  // self-correcting, which was the visible glitch right before fly-out.
  // 0 as a pure placeholder instead of performance.now() here
  // (LAUNCH_AUDIT.md P2-10, react-hooks/purity - calling an impure function
  // during render is flagged even as a useRef initial value) - the mount
  // effect below (dependency array includes the initial `phase`) sets the
  // real timestamp synchronously after commit, before any useFrame callback
  // can read it.
  const phaseStartedAtRef = useRef<number>(0);

  useEffect(() => {
    phaseStartedAtRef.current = performance.now();
  }, [phase]);

  return (
    <>
      <color attach="background" args={[mode === "light" ? "#ffffff" : "#000000"]} />

      {/* The two rings stay on separate Z planes (one in front, one behind)
          for their whole flight, so they never penetrate/z-fight each other —
          they overlap in screen space like the reference, with a clean depth
          gap instead of melding. */}
      <RingHalf
        matcap={matcap}
        phase={phase}
        phaseStartedAtRef={phaseStartedAtRef}
        start={{ x: 9, y: 5.5, z: 0.5 }}
        dock={{ x: 1.15, y: 0.85, z: 0.5 }}
        rotation={{ x: 0.35, y: 0.55, z: -0.55 }}
      />

      <RingHalf
        matcap={matcap}
        phase={phase}
        phaseStartedAtRef={phaseStartedAtRef}
        start={{ x: -9, y: -5.5, z: -0.5 }}
        dock={{ x: -1.15, y: -0.85, z: -0.5 }}
        rotation={{ x: -0.35, y: 0.55 + Math.PI, z: -0.55 + Math.PI }}
      />
    </>
  );
}

/** Two glossy black half-rings that fly in from opposite corners, hold at
 * their docked position, then fly back out in reverse — driven entirely by
 * the `phase` prop ("idle" | "in" | "hold" | "out") rather than a one-shot
 * boolean, so the same component can play forward and backward. Shaded with
 * a procedural matcap (no scene lights needed — the lighting response is
 * baked into the matcap texture by view-space normal). Each ring is a flat,
 * thin-edged band (revolved chamfered rectangle via LatheGeometry), so it
 * reads as a finger ring rather than a round wire. The two rings sit on
 * separate Z planes so they pass cleanly without intersecting. */
export default function RingsScene({ phase }: RingsSceneProps) {
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas camera={{ position: [0, 0, 7], fov: 45 }} dpr={[1, 2]}>
        <SceneContents phase={phase} />
      </Canvas>
    </div>
  );
}
