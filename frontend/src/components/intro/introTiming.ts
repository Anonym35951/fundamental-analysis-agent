// Extracted out of RingsScene.tsx (which pulls in three.js/@react-three/fiber)
// so callers that only need these two numbers - like AppLayout.tsx timing its
// crossfade - don't force-load the 3D scene's chunk just to read a constant.
// See P2-15 (LAUNCH.md): the intro is now lazy-loaded, and this keeps the
// split clean.
export const FLY_IN_DURATION = 2.8;
export const FLY_OUT_DURATION = 2.8;
