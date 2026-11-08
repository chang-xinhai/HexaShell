# 4. Seeing and measuring (RGBD)

The three RGB cameras have aligned optical-axis depth in metres. SAM3 outlines objects in RGB; depth pixels within each outline are back-projected through the calibrated camera into world coordinates. Missing, nonfinite, too-near and too-far readings do not count. Robot pixels are excluded from object measurements. This is not a visual-hull or guessed-height RGB-only measurement.

1. Look with robot.look('all', tile=True), or use grid/crop/zoom. Pixel coordinates always refer to the full original picture.
2. Find objects using robot.find('head', 'red cup') or a pixel, points or box. Choose the correct returned mask and inspect it using robot.look(cam,masks=[mask]). Text prompts find appearance, not ordering or identity.
3. robot.measure(mask), words or (cam,(x,y)) collects visible surface points from allowed cameras. The far side is unseen: a centre can be biased toward cameras by as much as half the object's depth. Transparent, thin and reflective objects may have missing or misleading depth. Read warnings and inspect outlines. Multiple views improve coverage but are not proof of identity.
4. After objects or robot arms move, remeasure. A wrist view nearer or above an object can improve resolution but must keep the complete object visible and avoid occlusion by a hand. Use measured coordinates cautiously for a narrow slot or hole; a mask may include the surrounding fixture rather than the opening.
5. Check target world positions with robot.look('head',marks=[('world',xyz,'target')]) or robot.to_pixel before committing a close motion.

robot.to_world(cam,(x,y)) uses the metric depth at that pixel of the last delivered picture for that camera, not an assumed tabletop. It may use valid neighboring depth samples; it returns no point when no usable local reading exists, and names its source as "the depth picture". A picked pixel showing the robot is warned. Explicit z= uses a plane at that height; pair=(other_cam,(x2,y2)) triangulates the two chosen rays and reports agreement. These explicit modes do not consume depth.

A held object is tracked relative to the hand. Its visible surfaces can be occluded by the gripper, so measure destination geometry before grasping when possible. robot.image still returns RGB only; the depth is used by host-side sensing tools. No filesystem or network depth access is available inside the code sandbox.
